"""Compare calibrated export lighting with native fixed-camera rendering.

Also report wall/floor color against depth-consistent recorded pixels. These
photometric measurements are diagnostics: capture auto-exposure and omitted
objects prevent exact agreement with one shared physical scene.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from export_web import set_camera

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', help='Optional Git revision for old paired RGB diagnostics')
    args = parser.parse_args()
    manifest = json.loads((WEB / 'data/manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = manifest['frames']
    indices = [0, 31, 43, 122, 123, 124, 125, 160, 172]
    parity = []
    photo = []
    sheet = Image.new('RGB', (1024, 404 * len(indices)), 'white')
    labels = ImageDraw.Draw(sheet)
    walls = [i for i in range(model.ngeom) if model.geom(i).name.startswith(
        ('north_wall', 'east_wall', 'west_wall', 'south_wall', 'door_lintel'))]
    floors = [i for i in range(model.ngeom) if model.geom(i).name.startswith('floor')]
    with mujoco.Renderer(model, height=384, width=512) as renderer:
        for row, index in enumerate(indices):
            frame = frames[index]
            transform = np.array(frame['camera_to_world'])
            # Native update_scene owns the camera and headlight here. Compare
            # against the export override at equivalent centered intrinsics.
            camera = model.camera('interior')
            camera.pos[:] = transform[:3, 3]
            mujoco.mju_mat2Quat(camera.quat, transform[:3, :3].flatten())
            camera.fovy[0] = np.rad2deg(2 * np.arctan(384 / frame['intrinsics']['fy']))
            camera.ipd[0] = 0
            mujoco.mj_forward(model, data)
            option = mujoco.MjvOption()
            option.geomgroup[:] = 1
            renderer.update_scene(data, camera='interior', scene_option=option)
            native = renderer.render().copy()
            centered = {**frame, 'intrinsics': {**frame['intrinsics'],
                        'cx': 512, 'cy': 384, 'fx': frame['intrinsics']['fy']}}
            set_camera(renderer, model, data, centered)
            exported = renderer.render().copy()
            delta = np.abs(native.astype(float) - exported.astype(float))
            assert delta.mean() < .05 and np.percentile(delta, 99) <= 1, (index, delta.mean())
            parity.append({'frame': index + 1, 'mean_rgb_difference_255': float(delta.mean()),
                           'p99_rgb_difference_255': float(np.percentile(delta, 99))})

            set_camera(renderer, model, data, frame)
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()[:, :, 0]
            renderer.disable_segmentation_rendering()
            recorded = np.array(Image.open(WEB / frame['rgb']).resize((512, 384)))
            rendered = np.array(Image.open(WEB / frame['render_rgb']))
            raw_depth = np.fromfile(WEB / frame['depth'], dtype='<u2').reshape(192, 256)
            sim_depth = np.fromfile(WEB / frame['render_depth'], dtype='<u2').reshape(192, 256)
            confidence = np.fromfile(WEB / frame['confidence'], dtype='u1').reshape(192, 256)
            aligned = ((raw_depth > 200) & (confidence >= 200) &
                       (np.abs(raw_depth.astype(float) - sim_depth) < 100))
            aligned = aligned.repeat(2, axis=0).repeat(2, axis=1)
            baseline = None
            if args.baseline_ref:
                blob = subprocess.check_output(['git', 'show',
                    f'{args.baseline_ref}:mujoco_scene/web/{frame["render_rgb"]}'], cwd=ROOT.parent)
                baseline = np.array(Image.open(io.BytesIO(blob)))
            stats = {'frame': index + 1}
            for name, ids in [('walls', walls), ('floor', floors)]:
                mask = np.isin(segmentation, ids) & aligned
                if mask.sum() < 100:
                    continue
                values = {'pixels': int(mask.sum()),
                          'recorded_median_rgb': np.median(recorded[mask], axis=0).tolist(),
                          'rendered_median_rgb': np.median(rendered[mask], axis=0).tolist()}
                if baseline is not None:
                    values['baseline_median_rgb'] = np.median(baseline[mask], axis=0).tolist()
                stats[name] = values
            photo.append(stats)
            sheet.paste(Image.fromarray(recorded), (0, row * 404 + 20))
            sheet.paste(Image.fromarray(rendered), (512, row * 404 + 20))
            labels.text((8, row * 404 + 4), f'Recorded / Frame {index + 1:03}', fill='black')
            labels.text((520, row * 404 + 4), 'MuJoCo / shared native and comparison lighting', fill='black')

    # Independent photo annotations catch placement errors, including parallax
    # introduced by the mounts' stand-off from the fitted wall plane.
    annotations = json.loads((ROOT / 'reference/wall_mount_corners.json').read_text())
    frame = frames[annotations['frame'] - 1]
    assert frame['id'] == annotations['frame_id']
    transform = np.array(frame['camera_to_world'])
    intrinsics = frame['intrinsics']
    mounts = []
    for annotation in annotations['mounts']:
        geom = model.geom(annotation['geom'])
        width, depth, height = geom.size
        rotation = data.geom_xmat[geom.id].reshape(3, 3)
        center = data.geom_xpos[geom.id]
        projected = []
        for x, z in [(width, height), (-width, height), (-width, -height), (width, -height)]:
            point = transform[:3, :3].T @ (center + rotation @ [x, depth, z] - transform[:3, 3])
            projected.append([intrinsics['cx'] + intrinsics['fx'] * point[0] / -point[2],
                              intrinsics['cy'] - intrinsics['fy'] * point[1] / -point[2]])
        errors = np.linalg.norm(np.array(projected) - annotation['pixels'], axis=1)
        assert errors.max() < 8, (annotation['geom'], errors)
        mounts.append({'geom': annotation['geom'], 'corner_reprojection_errors_px': errors.tolist()})

    clipping = []
    for frame in frames:
        rgb = np.array(Image.open(WEB / frame['render_rgb']))
        clipping.append(float((rgb.min(axis=2) > 250).mean()))
    assert max(clipping) < .01, 'Rendered white clipping exceeds 1% of a frame'
    report = {'native_camera_parity': parity, 'wall_mount_reprojection': mounts, 'photometry': photo,
              'photometry_mask': 'Rendered surface and recorded confidence >= 200, depth agreement < 0.10 m',
              'baseline_ref': args.baseline_ref, 'rendered_frames_checked': len(frames),
              'maximum_white_clipping_fraction': max(clipping)}
    (ROOT / 'appearance_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    sheet.save(ROOT / 'reference/appearance_comparison.jpg', quality=90)
    print(f'PASS: {len(parity)} native/export camera comparisons; {len(frames)} clipping checks.')
    print(f'Maximum native/export mean RGB difference: {max(p["mean_rgb_difference_255"] for p in parity):.4f}/255')


if __name__ == '__main__':
    main()
