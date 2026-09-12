# HaLab raw RGB-D dataset

[Download the original raw capture](https://github.com/whitealex95/halab-twin/releases/latest/download/halab-raw-dataset.zip) · [Archive SHA-256](https://github.com/whitealex95/halab-twin/releases/latest/download/halab-raw-dataset.sha256)

Captured with an iPhone 14 Pro. The capture contains 179 RGB-D keyframes spanning approximately 244.7 seconds, with 14,811 additional raw ARKit pose samples. The archive contains all 983 preserved capture files, a per-file checksum list, and this description.

Unzip in the repository root to reproduce the original paths:

```bash
unzip halab-raw-dataset.zip
```

The original data is under `Polycam_HaLab_WT2_gpt/Images/keyframes/`:

| Folder | Contents |
| --- | --- |
| `images/` | 179 original 1024 × 768 JPEG frames |
| `depth/` | 179 original 256 × 192 uint16 PNG maps, optical-axis depth in millimeters |
| `confidence/` | 179 original 256 × 192 confidence PNG maps; the demo uses values ≥200 |
| `cameras/` | Timestamp-matched JSON intrinsics and camera-to-world matrices |
| `arkit_motion/` | Raw tracking transforms and tracking state |
| `motion/` | Original inertial/motion records |
| `anchor_updates/` | Original capture anchor updates |
| `location/` | Original capture location metadata |

Timestamp filenames and JSON timestamps use the capture clock in microseconds. Camera JSON `fx`, `fy`, `cx`, and `cy` refer to the full RGB resolution. `t_00` through `t_23` are the three rows of a camera-to-world matrix; append `[0,0,0,1]`. Camera coordinates use +X right, +Y up, −Z forward; the raw capture uses Y up. ARKit motion arrays store flattened column-major transforms. The raw values are preserved without corrections or optimization.

The MuJoCo/web scene uses Z up. Its exact transform is recorded in `mujoco_scene/raw_measurements.json`. The projected keyframe poses and raw ARKit trajectory both use that same transform. Raw ARKit samples and keyframe poses are presented as separate trajectories; neither is replaced with a smoothed or optimized path.

To read depth without reducing its bit depth:

```python
from PIL import Image
import numpy as np
z_m = np.asarray(Image.open("path/to/depth.png"), dtype=np.float32) * 0.001
```

The archive intentionally excludes the deleted textured meshes, generated point clouds, RoomPlan/object-detection outputs, Gaussian splats, rendered turntable videos, and their previews. The downloadable files are the original RGB-D/sensor capture, not a reconstruction.

The browser distribution includes original RGB images and lossless millimeter depth/confidence binaries for paired visualization. MuJoCo-rendered RGB-D files are clearly separate in `web/data/rendered/`; they are simulation output, not camera measurements.
