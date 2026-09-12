#!/usr/bin/env python3
"""Launch the interactive HaLab reconstruction in the native MuJoCo viewer."""
from pathlib import Path
import argparse
import queue
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=None, help='Close after this many wall-clock seconds (smoke test)')
    args = parser.parse_args()
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    events = queue.SimpleQueue()
    movable = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    index = -1
    paused = False
    print('HaLab MuJoCo | Double-click to select; Ctrl + right-drag to pull; Ctrl + left-drag to rotate.')
    print('Space: pause | Backspace: reset | 7: select next asset | 8: push selected asset')
    print('2 / 3: toggle walls | [ / ]: camera views | F8: save state | F9: restore state | Esc: free camera')
    print('Doors and drawers respond directly to mouse forces. F1 opens native viewer help.')
    state_path = ROOT / 'saved_state.npz'
    deadline = time.monotonic() + args.seconds if args.seconds is not None else float('inf')
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 1, 1.1]
            viewer.cam.distance = 4.0
            viewer.cam.azimuth = -90
            viewer.cam.elevation = -8
            viewer.opt.geomgroup[2:4] = 1
            viewer.opt.geomgroup[4] = 0
        next_frame = time.monotonic()
        push_until = 0.
        push_body = 0
        while viewer.is_running() and time.monotonic() < deadline:
            with viewer.lock():
                while not events.empty():
                    key = events.get()
                    if key == 32:
                        paused = not paused
                    elif key == 259:  # GLFW Backspace
                        mujoco.mj_resetData(model, data)
                        push_until = 0
                        mujoco.mj_forward(model, data)
                    elif key == ord('7') and movable:
                        index = (index + 1) % len(movable)
                        viewer.pert.select = movable[index]
                        viewer.pert.localpos[:] = model.body_ipos[movable[index]]
                        print('Selected:', model.body(movable[index]).name)
                    elif key == ord('8') and viewer.pert.select:
                        push_body = int(viewer.pert.select)
                        push_until = data.time + .25
                        paused = False
                    elif key == 297:  # GLFW F8
                        np.savez(state_path, qpos=data.qpos, qvel=data.qvel, time=np.array(data.time))
                        print('Saved:', state_path)
                    elif key == 298 and state_path.exists():
                        with np.load(state_path, allow_pickle=False) as saved:
                            if saved['qpos'].shape != data.qpos.shape or saved['qvel'].shape != data.qvel.shape:
                                print('Saved state does not match this model; ignored.')
                            else:
                                mujoco.mj_resetData(model, data)
                                data.qpos[:] = saved['qpos']; data.qvel[:] = saved['qvel']; data.time = float(saved['time'])
                                push_until = 0
                                mujoco.mj_forward(model, data)
                                print('Restored:', state_path)
                data.qfrc_applied[:] = 0
                if not paused:
                    # Native sync transfers mouse perturbations into xfrc_applied.
                    for _ in range(round(1 / (60 * model.opt.timestep))):
                        data.qfrc_applied[:] = 0
                        if data.time < push_until:
                            force = np.array([model.body_mass[push_body] * 12, 0., 0.])
                            mujoco.mj_applyFT(model, data, force, np.zeros(3), data.xipos[push_body], push_body, data.qfrc_applied)
                        mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)
            viewer.sync()
            next_frame += round(1 / (60 * model.opt.timestep)) * model.opt.timestep
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()

if __name__ == '__main__':
    main()
