"""Fit room boundary planes directly to confident raw depth observations."""
from pathlib import Path
import json
import numpy as np
from measure_raw import RAW,OUT,unproject

def fit():
 meta=json.loads((OUT/'raw_measurements.json').read_text());R=np.array(meta['raw_camera_to_scene_rotation']);center=np.array(meta['scene_xy_center_before_translation_m'])
 samples=[];ceiling=[]
 for f in sorted((RAW/'images').glob('*.jpg')):
  p,_,valid=unproject(f.stem,4);p=p@R.T;p[:,:,:2]-=center;p[:,:,2]-=meta['floor_raw_y_m']
  n=np.cross(p[1:,:-1]-p[:-1,:-1],p[:-1,1:]-p[:-1,:-1]);norm=np.linalg.norm(n,axis=-1);n/=np.maximum(norm[...,None],1e-9)
  good=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&(np.abs(n[:,:,2])<.12)&(norm>.0003)&(norm<.12)
  samples.append(np.c_[p[:-1,:-1][good],n[good]])
  horizontal=valid[:-1,:-1]&(np.abs(n[:,:,2])>.97)&(p[:-1,:-1,2]>2.5)&(p[:-1,:-1,2]<4)
  ceiling.extend(p[:-1,:-1,2][horizontal].tolist())
 pts=np.concatenate(samples);planes={}
 for name,axis,sign in [('west',0,-1),('east',0,1),('north',1,-1),('south',1,1)]:
  cross=1-axis;limit=4.8 if axis==0 else 3.5
  q=pts[(pts[:,axis]*sign>limit)&(pts[:,2]>.3)&(np.abs(pts[:,3+axis])>.94),:3]
  hist,edges=np.histogram(q[:,axis],bins=100);offset=float((edges[hist.argmax()]+edges[hist.argmax()+1])/2)
  sel=np.abs(q[:,axis]-offset)<.18
  for _ in range(5):
   a,b=np.linalg.lstsq(np.c_[q[sel,cross],np.ones(sel.sum())],q[sel,axis],rcond=None)[0]
   residual=np.abs(q[:,axis]-(a*q[:,cross]+b));sel=residual<.075
  ztop=float(np.percentile(q[sel,2],99))
  planes[name]={'axis':axis,'slope':float(a),'offset':float(b),'inliers':int(sel.sum()),'median_residual_m':float(np.median(residual[sel])),'p95_residual_m':float(np.percentile(residual[sel],95)),'observed_horizontal_range_m':np.percentile(q[sel,cross],[1,99]).tolist(),'observed_top_m':ztop}
 def intersect(xname,yname):
  x,y=planes[xname],planes[yname]
  return np.linalg.solve([[1,-x['slope']],[-y['slope'],1]],[x['offset'],y['offset']]).tolist()
 corners=[intersect('west','north'),intersect('east','north'),intersect('east','south'),intersect('west','south')]
 hist,edges=np.histogram(ceiling,bins=75,range=(2.5,4))
 height=round(float((edges[hist.argmax()]+edges[hist.argmax()+1])/2),2)
 # Do not infer unseen ceiling contours; a level top is a declared approximation.
 result={'source':'Original RGB-D and camera poses only','planes':planes,'corners_xy_m':corners,'height_m':height,'ceiling_horizontal_samples':len(ceiling),'thickness_m':.15,'method':'Robust vertical plane fitting to confidence-filtered raw depth normals; 7.5 cm inlier band. Wall intersections close the room. Wall top uses the modal high horizontal depth surface; unobserved spans and a level ceiling are extrapolated. Thickness and door heights are estimates.', 'doors':[{'name':'room_door1','wall':'north','x_center_m':-3.65,'width_m':.90,'height_m':2.5},{'name':'room_door2','wall':'north','x_center_m':3.825,'width_m':.95,'height_m':2.5}]}
 (OUT/'walls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':fit()
