"""Export the MJCF, raw trajectories and calibrated recorded/rendered RGB-D pairs."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,shutil,hashlib
import numpy as np
from PIL import Image
import mujoco
OUT=Path(__file__).resolve().parents[1]
RAW=OUT.parent/'Polycam_HaLab_WT2_gpt/Images/keyframes'
WEB=OUT/'web';DATA=WEB/'data'

def set_camera(renderer,model,data,frame):
 opt=mujoco.MjvOption();opt.geomgroup[:]=1
 renderer.update_scene(data,scene_option=opt)
 T=np.array(frame['camera_to_world']);c=frame['intrinsics'];near=.025;far=40.
 for camera in renderer.scene.camera:
  camera.pos[:]=T[:3,3];camera.forward[:]=-T[:3,2];camera.up[:]=T[:3,1]
  camera.orthographic=0;camera.frustum_near=near;camera.frustum_far=far
  camera.frustum_top=c['cy']/c['fy']*near
  camera.frustum_bottom=-(c['height']-c['cy'])/c['fy']*near
  camera.frustum_center=(c['width']/2-c['cx'])/c['fx']*near
  camera.frustum_width=c['width']/(2*c['fx'])*near
 # Renderer depth conversion uses model clipping distances.
 model.vis.map.znear=near/model.stat.extent;model.vis.map.zfar=far/model.stat.extent

def main():
 DATA.mkdir(parents=True,exist_ok=True)
 for folder in ['recorded','rendered']: (DATA/folder).mkdir(exist_ok=True)
 meta=json.loads((OUT/'raw_measurements.json').read_text());R=np.array(meta['raw_camera_to_scene_rotation']);offset=np.r_[meta['scene_xy_center_before_translation_m'],meta['floor_raw_y_m']]
 def transform(T):
  result=np.eye(4);result[:3,:3]=R@T[:3,:3];result[:3,3]=R@T[:3,3]-offset;return result
 frames=[];files=sorted((RAW/'cameras').glob('*.json'));start=json.loads(files[0].read_text())['timestamp']
 for i,f in enumerate(files):
  c=json.loads(f.read_text());T=np.eye(4);T[:3,:]=[[c[f't_{j}{k}'] for k in range(4)] for j in range(3)]
  frames.append({'index':i,'id':f.stem,'timestamp_us':c['timestamp'],'time_s':(c['timestamp']-start)/1e6,'camera_to_world':transform(T).tolist(),'intrinsics':{k:c[k] for k in ['fx','fy','cx','cy','width','height']},'rgb':f'data/recorded/{f.stem}.jpg','depth':f'data/recorded/{f.stem}.depth.bin','confidence':f'data/recorded/{f.stem}.confidence.bin','render_rgb':f'data/rendered/{f.stem}.jpg','render_depth':f'data/rendered/{f.stem}.depth.bin'})
 # Keep raw tracking samples separate from corrected/optimized reconstructions.
 dense={}
 for f in sorted((RAW/'arkit_motion').glob('*.json')):
  for item in json.loads(f.read_text()):
   if 'transform' in item:
    T=np.array(item['transform']).reshape(4,4).T;dense[item['timestamp']]={'time_s':(item['timestamp']-start)/1e6,'position':transform(T)[:3,3].tolist(),'tracking_state':item.get('trackingState','unknown')}
 trajectory=[dense[k] for k in sorted(dense)]
 model=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
 # Export the actual compiled reference geometry, not a parallel hand-authored model.
 geoms=[]
 types={int(mujoco.mjtGeom.mjGEOM_BOX):'box',int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder',int(mujoco.mjtGeom.mjGEOM_CAPSULE):'capsule'}
 for i in range(model.ngeom):
  kind=types.get(int(model.geom_type[i]));assert kind is not None,(i,model.geom_type[i])
  rgba=model.geom_rgba[i].copy();material=int(model.geom_matid[i])
  if material>=0:
   rgba=model.mat_rgba[material].copy()
   texture_ids=model.mat_texid[material];texture_ids=texture_ids[texture_ids>=0]
   if len(texture_ids):
    tex=int(texture_ids[0]);adr=int(model.tex_adr[tex]);count=int(model.tex_width[tex]*model.tex_height[tex]*model.tex_nchannel[tex])
    pixels=model.tex_data[adr:adr+count].reshape(-1,int(model.tex_nchannel[tex]))
    rgba[:3]*=pixels[:,:3].mean(axis=0)/255
  bid=int(model.geom_bodyid[i]);root=bid
  while model.body_parentid[root]>0:root=int(model.body_parentid[root])
  geoms.append({'id':i,'name':model.geom(i).name,'body':model.body(bid).name,'asset':'room' if root==0 else model.body(root).name,'type':kind,'size':model.geom_size[i].tolist(),'position':data.geom_xpos[i].tolist(),'rotation':data.geom_xmat[i].reshape(3,3).tolist(),'rgba':rgba.tolist(),'group':int(model.geom_group[i])})
 renderer=mujoco.Renderer(model,height=384,width=512)
 # Disable MSAA for depth: multisample resolve can select an off-center subpixel depth.
 model.vis.quality.offsamples=0
 depth_renderer=mujoco.Renderer(model,height=192,width=256)
 try:
  for frame in frames:
   stem=frame['id'];shutil.copyfile(RAW/'images'/f'{stem}.jpg',DATA/'recorded'/f'{stem}.jpg')
   recorded=np.array(Image.open(RAW/'depth'/f'{stem}.png'),dtype='<u2');confidence=np.array(Image.open(RAW/'confidence'/f'{stem}.png'),dtype='u1')
   recorded.tofile(DATA/'recorded'/f'{stem}.depth.bin');confidence.tofile(DATA/'recorded'/f'{stem}.confidence.bin')
   set_camera(renderer,model,data,frame);Image.fromarray(renderer.render()).save(DATA/'rendered'/f'{stem}.jpg',quality=91)
   set_camera(depth_renderer,model,data,frame);depth_renderer.enable_depth_rendering();depth=depth_renderer.render();depth_renderer.disable_depth_rendering()
   valid=np.isfinite(depth)&(depth>.025)&(depth<39.9);millimeters=np.where(valid,np.minimum(depth*1000,65535),0).round().astype('<u2');millimeters.tofile(DATA/'rendered'/f'{stem}.depth.bin')
   mask=(recorded>200)&(recorded<9000)&(confidence>=200)&valid
   error=millimeters.astype(float)/1000-recorded.astype(float)/1000
   frame['depth_metrics']={'valid_pixels':int(mask.sum()),'coverage':float(mask.mean()),'mae_m':float(np.mean(abs(error[mask]))) if mask.any() else None,'rmse_m':float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None}
   if frame['index']%25==0:print(f"Exported {frame['index']+1}/{len(frames)}",flush=True)
 finally:renderer.close();depth_renderer.close()
 manifest={'title':'HaLab · Capture & twin','frame_count':len(frames),'duration_s':frames[-1]['time_s'],'depth_width':256,'depth_height':192,'depth_unit_m':.001,'depth_encoding':'little-endian uint16, optical-axis depth, 0 invalid','confidence_threshold':200,'rgb_render_size':[512,384],'coordinate_convention':'Z up. camera_to_world columns: right, up, backward, position; OpenGL -Z forward.','comparison':'MuJoCo reference state at the recorded pose and intrinsics; no pose optimization.','scene_sha256':hashlib.sha256((OUT/'scene.xml').read_bytes()).hexdigest(),'raw_only':True,'walls':json.loads((OUT/'walls.json').read_text()),'assets':json.loads((OUT/'inventory.json').read_text())['assets'],'geoms':geoms,'frames':frames,'raw_arkit_trajectory':trajectory}
 (DATA/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
 print(f"Ready: {len(frames)} RGB-D pairs, {len(trajectory)} raw tracking poses, {len(geoms)} MuJoCo geoms")
if __name__=='__main__':main()
