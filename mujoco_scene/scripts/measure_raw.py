"""Measure raw depth samples in the capture frame; no meshes or detections are read."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
RAW=OUT.parent/'Polycam_HaLab_WT2_gpt/Images/keyframes'

def unproject(stem,stride=4):
 c=json.loads((RAW/'cameras'/f'{stem}.json').read_text())
 z=np.array(Image.open(RAW/'depth'/f'{stem}.png'),dtype=float)[::stride,::stride]/1000
 conf=np.array(Image.open(RAW/'confidence'/f'{stem}.png'))[::stride,::stride]
 yy,xx=np.mgrid[:192:stride,:256:stride];u=(xx+.5)*4;v=(yy+.5)*4
 cam=np.stack([(u-c['cx'])/c['fx']*z,-(v-c['cy'])/c['fy']*z,-z],axis=-1)
 T=np.array([[c[f't_{i}{j}'] for j in range(4)] for i in range(3)])
 p=cam@T[:,:3].T+T[:,3]
 im=np.asarray(Image.open(RAW/'images'/f'{stem}.jpg').resize((256,192)))[::stride,::stride,:3]/255
 mask=(conf>=200)&(z>.2)&(z<9)
 return p,im,mask

def main():
 ps=[];cs=[];angles=[]
 for f in sorted((RAW/'images').glob('*.jpg')):
  p,c,mask=unproject(f.stem)
  n=np.cross(p[1:,:-1]-p[:-1,:-1],p[:-1,1:]-p[:-1,:-1]);length=np.linalg.norm(n,axis=-1);n/=np.maximum(length[...,None],1e-10)
  good=mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&(np.abs(n[:,:,1])<.08)&(length>.0003)&(length<.12)
  angles.extend(((np.arctan2(n[:,:,2][good],n[:,:,0][good])+np.pi/4)%(np.pi/2)-np.pi/4).tolist())
  ps.append(p[mask]);cs.append(c[mask])
 p=np.concatenate(ps);c=np.concatenate(cs)
 hist,edges=np.histogram(angles,bins=180,range=(-np.pi/4,np.pi/4));ang=(edges[hist.argmax()]+edges[hist.argmax()+1])/2
 floor_values=p[(p[:,1]>-2)&(p[:,1]<-.8),1];hist,edges=np.histogram(floor_values,bins=240,range=(-2,-.8));floor=float((edges[hist.argmax()]+edges[hist.argmax()+1])/2)
 R=np.array([[np.cos(ang),0,np.sin(ang)],[np.sin(ang),0,-np.cos(ang)],[0,1,0]])
 q=p@R.T;q[:,2]-=floor
 lo,hi=np.percentile(q[(q[:,2]>.3)&(q[:,2]<2.4),:2],[1,99],axis=0);center=(lo+hi)/2;q[:,:2]-=center
 report={'source':'Images/keyframes/{images,depth,confidence,cameras} only','depth_scale_m':.001,'depth_pixel_to_rgb_scale':4,'raw_camera_to_scene_rotation':R.tolist(),'scene_xy_center_before_translation_m':center.tolist(),'floor_raw_y_m':floor,'dominant_wall_angle_deg':float(np.degrees(ang)),'room_span_estimate_m':(hi-lo).tolist(),'sample_count':len(q),'method':'Confidence-filtered raw depth backprojection. Dominant horizontal wall-normal direction and modal floor height. No fused surface or automatic object classification.'}
 (OUT/'raw_measurements.json').write_text(json.dumps(report,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(12,10));sel=(q[:,2]>.12)&(q[:,2]<1.7);ax.scatter(q[sel,0],q[sel,1],c=c[sel],s=.45);ax.set_aspect('equal');ax.grid();ax.set_xlabel('Raw-derived scene X (m)');ax.set_ylabel('Raw-derived scene Y (m)');ax.set_title('Direct raw depth samples; no mesh or object detections');fig.savefig(OUT/'reference/raw_depth_plan.png',dpi=150)
 annotations=json.loads((OUT/'manual_photo_samples.json').read_text())
 for entry in annotations.values():
  u,v=entry['rgb_pixel'];points,_,valid=unproject(Path(entry['frame']).stem,1)
  patch=points[v//4-1:v//4+2,u//4-1:u//4+2]
  mask=valid[v//4-1:v//4+2,u//4-1:u//4+2]
  entry['confident_depth_pixels']=int(mask.sum())
  if mask.sum()>=3:
   point=np.median(patch[mask],axis=0)@R.T;point[:2]-=center;point[2]-=floor
   entry['surface_point_m']=point.tolist()
  else:
   entry['surface_point_m']=None
 (OUT/'manual_photo_measurements.json').write_text(json.dumps(annotations,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
