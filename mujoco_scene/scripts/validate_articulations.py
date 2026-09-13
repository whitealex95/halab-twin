"""Check cabinet sweep clearance and drawer containment with native contacts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
data = mujoco.MjData(model)
clearance = []
for asset in ['white_cabinet', 'sink_cabinet', 'black_dresser']:
    parent = model.body(asset).id
    fixed = [g for g in range(model.ngeom) if model.geom_bodyid[g] == parent and model.geom_contype[g]]
    for joint_id in range(model.njnt):
        child = model.jnt_bodyid[joint_id]
        if model.body_parentid[child] != parent or model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            continue
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            assert np.isclose(np.ptp(model.jnt_range[joint_id]), np.pi)
        moving = [g for g in range(model.ngeom) if model.geom_bodyid[g] == child and model.geom_contype[g]]
        minimum = .01
        for qpos in np.linspace(*model.jnt_range[joint_id], 25):
            data.qpos[:] = model.qpos0
            data.qpos[model.jnt_qposadr[joint_id]] = qpos
            mujoco.mj_forward(model, data)
            for geom in moving:
                for other in fixed:
                    distance = mujoco.mj_geomDistance(model, data, geom, other, .01, None)
                    minimum = min(minimum, distance)
        # Check geometry even for parent/child pairs normally excluded from
        # automatic contact generation by MuJoCo.
        assert minimum > .0005, (model.joint(joint_id).name, minimum)
        clearance.append({'joint': model.joint(joint_id).name,
                          'positions_checked': 25, 'minimum_carcass_clearance_m': minimum})

# Add one small test payload per drawer. A lateral impulse should be stopped
# by its side/front/back walls instead of falling through an incomplete tray.
xml = ET.parse(ROOT / 'scene.xml')
world = xml.getroot().find('worldbody')
for index in range(5):
    body = ET.SubElement(world, 'body', name=f'drawer_payload{index}')
    ET.SubElement(body, 'freejoint', name=f'drawer_payload_free{index}')
    ET.SubElement(body, 'geom', type='sphere', size='.012', mass='.015')
probe_model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding='unicode'))
probe_data = mujoco.MjData(probe_model)
containment = []
for direction in [-1, 1]:
    mujoco.mj_resetData(probe_model, probe_data)
    for index in range(5):
        slide = probe_model.joint(f'black_dresser_slide{index + 1}')
        probe_data.qpos[slide.qposadr[0]] = probe_model.jnt_range[slide.id, 1]
    mujoco.mj_forward(probe_model, probe_data)
    for index in range(5):
        tray = probe_model.geom(f'black_dresser_drawer_tray{index}')
        payload = probe_model.joint(f'drawer_payload_free{index}')
        position = probe_data.geom_xpos[tray.id].copy()
        position[2] += tray.size[2] + .016
        probe_data.qpos[payload.qposadr[0]:payload.qposadr[0] + 3] = position
        probe_data.qvel[payload.dofadr[0]:payload.dofadr[0] + 2] = [direction * 1.8, direction * 1.8]
    mujoco.mj_forward(probe_model, probe_data)
    for _ in range(1000):
        mujoco.mj_step(probe_model, probe_data)
    assert not any(w.number for w in probe_data.warning)
    mujoco.mj_forward(probe_model, probe_data)
    for index in range(5):
        tray = probe_model.geom(f'black_dresser_drawer_tray{index}')
        payload = probe_model.body(f'drawer_payload{index}')
        relative = (probe_data.geom_xmat[tray.id].reshape(3, 3).T @
                    (probe_data.xpos[payload.id] - probe_data.geom_xpos[tray.id]))
        assert abs(relative[0]) < tray.size[0] - .008, (index, relative)
        assert abs(relative[1]) < tray.size[1] - .005, (index, relative)
        assert .012 < relative[2] < .025, (index, relative)
        containment.append({'drawer': index + 1, 'impulse_direction': direction,
                            'payload_position_relative_to_tray_m': relative.tolist()})
report = {'cabinet_doors_open_degrees': 180, 'carcass_clearance': clearance,
          'drawer_payload_containment': containment,
          'scope': 'Own-carcass clearance; nearby furniture can still obstruct native door movement.'}
(ROOT / 'articulation_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS: {len(clearance)} joint sweeps and {len(containment)} drawer payload tests.')
