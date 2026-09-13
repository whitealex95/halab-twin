"""Headless physics regression and rendered preview for the reconstructed scene."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import mujoco
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
m=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
# The three divider instances must have the same geometry and three real hinges.
screen_names=['sofa_screen','center_screen','desk_screen']
panel_shapes=[]
for name in screen_names:
 for leaf in range(4):
  g=m.geom(f'{name}_paper{leaf}')
  panel_shapes.append(m.geom_size[g.id].copy())
 for hinge in range(1,4):
  j=m.joint(f'{name}_hinge{hinge}')
  assert m.jnt_type[j.id] == mujoco.mjtJoint.mjJNT_HINGE
  assert np.allclose(m.jnt_range[j.id],np.deg2rad([-175,175]))
assert all(np.allclose(shape,panel_shapes[0]) for shape in panel_shapes)
assert len([j for j in range(m.njnt) if any(m.joint(j).name.startswith(n+'_hinge') for n in screen_names)])==9
# The platform has four visible wheel assemblies and supports its single box.
assert len([i for i in range(m.ngeom) if m.geom(i).name.startswith('utility_wheel') and m.geom(i).name.endswith('_tire')])==4
cart=m.body('utility_cart').id;box_body=m.body('large_box').id
relative_box=d.xmat[cart].reshape(3,3).T@(d.xpos[box_body]-d.xpos[cart])
assert abs(relative_box[0])<.82/2-.47/2 and abs(relative_box[1])<.64/2-.49/2
bottom=m.geom('large_box_bottom');deck=m.geom('utility_deck_contact')
assert abs((d.geom_xpos[bottom.id,2]-bottom.size[2])-(d.geom_xpos[deck.id,2]+deck.size[2]))<.001
# In the user's reference views the handle is on the image-left end of the cart.
manifest=json.loads((ROOT/'web/data/manifest.json').read_text())
for index in [135,168,171]:
 frame=manifest['frames'][index];T=np.array(frame['camera_to_world']);k=frame['intrinsics']
 def pixel_x(g):
  p=T[:3,:3].T@(d.geom_xpos[m.geom(g).id]-T[:3,3]);return k['cx']+k['fx']*p[0]/-p[2]
 assert pixel_x('utility_handle_grip')<pixel_x('utility_deck')
initial=d.xpos.copy();worst=0.;pairs=[]
for c in d.contact:
 if c.dist<-.003:pairs.append((m.geom(c.geom1).name,m.geom(c.geom2).name,float(c.dist)))
print('Initial penetrations >3mm:',pairs)
for i in range(5000):
 mujoco.mj_step(m,d)
 assert np.isfinite(d.qpos).all(),f'Nonfinite state at step {i}'
 worst=max(worst,float(np.max(np.abs(d.qvel))))
free_ids=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
movement={m.body(m.jnt_bodyid[j]).name:round(float(np.linalg.norm(d.xpos[m.jnt_bodyid[j],:2]-initial[m.jnt_bodyid[j],:2])),4) for j in free_ids}
print('Settling horizontal movement:',movement,'peak speed',worst,'final speed',max(abs(d.qvel)))
print('Warnings:',[(i,int(w.number)) for i,w in enumerate(d.warning) if w.number])
# Force one chair and exercise a cabinet hinge and drawer by generalized force.
checks={}
for name,force,threshold in [('office_chair_free',120,.025),('white_cabinet_left_hinge',-5,.05),('black_dresser_slide1',25,.025)]:
 t=mujoco.MjData(m);t.qpos[:]=d.qpos;mujoco.mj_forward(m,t);j=m.joint(name);adr=j.dofadr[0];qadr=j.qposadr[0];start=t.qpos[qadr]
 for _ in range(350):t.qfrc_applied[adr]=force;mujoco.mj_step(m,t)
 delta=float(t.qpos[qadr]-start);checks[name]=delta;assert abs(delta)>threshold,(name,delta)
# Every divider hinge must respond, including the hidden folded panels.
for name in screen_names:
 for hinge in range(1,4):
  joint_name=f'{name}_hinge{hinge}';j=m.joint(joint_name);adr=j.dofadr[0];qadr=j.qposadr[0]
  changes=[]
  for force in [-6.,6.]:
   t=mujoco.MjData(m);t.qpos[:]=d.qpos;mujoco.mj_forward(m,t);start=t.qpos[qadr]
   for _ in range(250):
    t.qfrc_applied[adr]=force;mujoco.mj_step(m,t)
   assert np.isfinite(t.qpos).all() and not any(w.number for w in t.warning)
   changes.append(float(t.qpos[qadr]-start))
  checks[joint_name]=max(changes,key=abs)
  assert max(abs(x) for x in changes)>.025,(joint_name,changes)
print('Force-response checks:',checks)
assert not any(w.number for w in d.warning)
assert max(movement.values())<.12, 'Objects drift excessively while settling'
assert max(abs(d.qvel))<.2,'Scene does not settle'
assert not pairs,'Initial collision overlaps'
with mujoco.Renderer(m,height=1000,width=1400) as renderer:
 opt=mujoco.MjvOption();opt.geomgroup[2]=0;opt.geomgroup[3]=0
 for name in ['overview','interior','bed_area','sofa_area']:
  opt.geomgroup[2:4] = 0 if name == 'overview' else 1
  renderer.update_scene(d,camera=name,scene_option=opt)
  Image.fromarray(renderer.render()).save(ROOT/f'{name}.png')
report={'simulation_seconds':d.time,'divider_instances':3,'divider_panels':12,'divider_hinges':9,'identical_panel_geometry':True,'free_bodies':len(free_ids),'bodies':m.nbody,'joints':m.njnt,'geoms':m.ngeom,'initial_penetrations_over_3mm':pairs,'settling_xy_m':movement,'force_response':checks,'peak_generalized_speed':worst,'final_max_generalized_speed':float(max(abs(d.qvel))),'warnings':[]}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS; previews and validation.json saved')
