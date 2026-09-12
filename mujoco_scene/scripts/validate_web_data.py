"""Check frame completeness, raw integrity, poses, and optical depth calibration."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];WEB=ROOT/'web';m=json.loads((WEB/'data/manifest.json').read_text())
model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
assert m['scene_sha256']==hashlib.sha256((ROOT/'scene.xml').read_bytes()).hexdigest()
assert len(m['frames'])==179
for frame in m['frames']:
 T=np.array(frame['camera_to_world']);assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
 for key in ['rgb','render_rgb','depth','render_depth','confidence']:assert (WEB/frame[key]).is_file()
 raw=ROOT.parent/'Polycam_HaLab_WT2_gpt/Images/keyframes'
 assert (WEB/frame['rgb']).read_bytes()==(raw/'images'/f"{frame['id']}.jpg").read_bytes()
 assert np.array_equal(np.fromfile(WEB/frame['depth'],dtype='<u2').reshape(192,256),np.asarray(Image.open(raw/'depth'/f"{frame['id']}.png")))
calibration=[]
for index in [0,45,90,135,178]:
 frame=m['frames'][index];T=np.array(frame['camera_to_world']);k=frame['intrinsics'];depth=np.fromfile(WEB/frame['render_depth'],dtype='<u2').reshape(192,256)/1000
 errors=[]
 for v in range(13,192,19):
  for u in range(11,256,23):
   ray_camera=np.array([((u+.5)*4-k['cx'])/k['fx'],-((v+.5)*4-k['cy'])/k['fy'],-1.])
   length=np.linalg.norm(ray_camera);direction=T[:3,:3]@ray_camera/length;geomid=np.zeros(1,dtype=np.int32)
   dist=mujoco.mj_ray(model,data,T[:3,3].copy(),direction,np.ones(6,dtype=np.uint8),1,-1,geomid)
   if dist>0 and depth[v,u]>0:errors.append(abs(dist/length-depth[v,u]))
 median=float(np.median(errors));p90=float(np.percentile(errors,90));assert median<.002 and p90<.008,(index,median,p90)
 calibration.append({'frame':index,'median_ray_vs_render_depth_error_m':median,'p90_error_m':p90,'rays':len(errors)})
checks=json.loads((ROOT/'raw_input_checksums.json').read_text());assert all(hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest()==h for p,h in checks.items())
report={'frames':len(m['frames']),'raw_arkit_poses':len(m['raw_arkit_trajectory']),'original_rgb_and_depth_preserved':True,'raw_source_checksum_count':len(checks),'camera_rotations_valid':True,'calibration':calibration,'mean_frame_depth_mae_m':float(np.mean([f['depth_metrics']['mae_m'] for f in m['frames']]))}
(ROOT/'web_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
