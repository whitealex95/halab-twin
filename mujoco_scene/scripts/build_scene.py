"""Build a metric, editable MuJoCo approximation of the HaLab capture."""
from pathlib import Path
import json, math, xml.etree.ElementTree as ET
import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
inventory = []
def el(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k,v in kw.items()})
def vec(a): return ' '.join(f'{float(x):.6g}' for x in a)
def box(p,n,pos,size,mat='white',mass=None,visual=False,group=0):
    kw=dict(name=n,type='box',pos=vec(pos),size=vec(np.array(size)/2),material=mat,group=group)
    if mass is not None: kw['mass']=mass
    if visual: kw.update(contype=0,conaffinity=0,mass=0)
    return el(p,'geom',**kw)
def cyl(p,n,pos,r,h,mat='metal',mass=.2):
    return el(p,'geom',name=n,type='cylinder',pos=vec(pos),size=vec([r,h/2]),material=mat,mass=mass)
def body(n,x,y,yaw=0,free=True,source='Manual raw-photo interpretation and raw depth samples; approximate dimensions',size=None):
    b=el(world,'body',name=n,pos=vec([x,y,.004 if free else 0]),euler=f'0 0 {yaw}')
    if free: el(b,'freejoint',name=n+'_free')
    inventory.append(dict(name=n,movable=free,position_m=[x,y,0],yaw_deg=yaw,dimensions_m=size,source=source))
    return b

def legs(b,n,w,d,h,mat='metal',thick=.035):
    for i,(x,y) in enumerate([(x,y) for x in [-w/2+thick,w/2-thick] for y in [-d/2+thick,d/2-thick]]):
        box(b,f'{n}_leg{i}',[x,y,h/2],[thick,thick,h],mat,mass=.8)
def table(n,x,y,w,d,h,yaw=0,mat='oak'):
    b=body(n,x,y,yaw,size=[w,d,h]);box(b,n+'_top',[0,0,h-.025],[w,d,.05],mat,mass=9);legs(b,n,w,d,h-.05);return b

def shelf(n,x,y,w,d,h,levels=4,yaw=0,mat='oak'):
    b=body(n,x,y,yaw,size=[w,d,h]);legs(b,n,w,d,h,thick=.025)
    for i,z in enumerate(np.linspace(.1,h-.025,levels)):box(b,f'{n}_shelf{i}',[0,0,z],[w,d,.035],mat,mass=2)
    return b

def chair(n,x,y,yaw=0,swivel=True):
    b=body(n,x,y,yaw,size=[.55,.60,.96]);box(b,n+'_seat',[0,0,.46],[.48,.45,.08],'charcoal',mass=3)
    box(b,n+'_back',[0,.20,.74],[.46,.065,.44],'charcoal',mass=2)
    if swivel:
        cyl(b,n+'_column',[0,0,.24],.036,.38,mass=2)
        for i,a in enumerate(np.arange(5)*2*math.pi/5):
            el(b,'geom',name=f'{n}_spoke{i}',type='capsule',fromto=vec([0,0,.09,.26*math.cos(a),.26*math.sin(a),.055]),size='.018',material='metal',mass='.3')
            el(b,'geom',name=f'{n}_caster{i}',type='sphere',pos=vec([.26*math.cos(a),.26*math.sin(a),.032]),size='.032',material='charcoal',mass='.1',friction='.22 .005 .001')
        for s in [-1,1]:box(b,f'{n}_arm{s}',[s*.27,0,.65],[.045,.33,.045],'charcoal',mass=.4)
    else: legs(b,n,.45,.43,.42)
    return b

def cabinet(n,x,y,w,d,h,yaw=0,mat='white',drawers=False):
    b=body(n,x,y,yaw,size=[w,d,h]);t=.035
    for label,pos,size in [('bottom',[0,0,.05],[w,d,.1]),('top',[0,0,h-t/2],[w,d,t]),('back',[0,d/2-t/2,h/2],[w,t,h]),('left',[-w/2+t/2,0,h/2],[t,d,h]),('right',[w/2-t/2,0,h/2],[t,d,h])]:box(b,n+'_'+label,pos,size,mat,mass=4)
    if drawers:
        for i in range(5):
            z=.14+(h-.16)/5*(i+.5);dh=(h-.16)/5-.015
            dr=el(b,'body',name=f'{n}_drawer{i+1}',pos=vec([0,0,z]));el(dr,'joint',name=f'{n}_slide{i+1}',type='slide',axis='0 -1 0',range=f'0 {d*.65}',damping='8',frictionloss='2')
            box(dr,f'{n}_drawer_front{i}',[0,-d/2,0],[w-.075,.025,dh],mat,mass=1.2)
            box(dr,f'{n}_drawer_tray{i}',[0,0,-dh/2+.015],[w-.09,d-.08,.025],mat,mass=.5)
            box(dr,f'{n}_drawer_handle{i}',[0,-d/2-.028,0],[w*.3,.03,.018],'metal',mass=.1)
    else:
        box(b,n+'_shelf',[0,.01,h*.5],[w-.08,d-.05,.028],mat,mass=2)
        for s,label in [(-1,'left'),(1,'right')]:
            door=el(b,'body',name=n+'_'+label+'_door',pos=vec([s*(w/2-.025),-d/2-.007,.1]))
            el(door,'joint',name=n+'_'+label+'_hinge',type='hinge',axis='0 0 1',range='-105 0' if s==-1 else '0 105',damping='2',frictionloss='.4')
            box(door,n+'_'+label+'_panel',[-s*(w/4-.026),0,(h-.12)/2],[w/2-.025,.025,h-.12],mat,mass=2)
            box(door,n+'_'+label+'_handle',[-s*(w/2-.10),-.032,h*.48],[.022,.04,.15],'brass',mass=.08)
    return b

model=ET.Element('mujoco',model='HaLab — image-based interactive reconstruction')
el(model,'compiler',angle='degree',autolimits='true')
el(model,'option',timestep='.002',integrator='implicitfast',gravity='0 0 -9.81',cone='elliptic',iterations='60')
el(model,'size',njmax='8000',nconmax='2000')
el(model,'statistic',center='0 0 .8',extent='6.8')
v=el(model,'visual');el(v,'global',offwidth='1600',offheight='1200',azimuth='125',elevation='-48');el(v,'headlight',ambient='.6 .6 .6',diffuse='.8 .8 .8',specular='.1 .1 .1');el(v,'rgba',haze='.12 .15 .19 1')
d=el(model,'default');el(d,'geom',friction='.7 .01 .001',solref='.015 1',solimp='.95 .99 .001',condim='4');el(d,'joint',damping='.5')
a=el(model,'asset')
colors={'white':'.86 .85 .79 1','wall':'.75 .73 .65 1','oak':'.56 .40 .23 1','charcoal':'.055 .065 .075 1','metal':'.17 .19 .20 1','sofa':'.30 .34 .34 1','cushion':'.39 .43 .42 1','brass':'.53 .44 .24 1','screenframe':'.27 .075 .055 1','paper':'.9 .9 .84 1','display':'.12 .32 .43 1','orange':'.96 .17 .025 1','blue':'.035 .12 .6 1'}
for n,c in colors.items():el(a,'material',name=n,rgba=c,specular='.15',shininess='.1')
el(a,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.25 .29 .34',rgb2='.075 .095 .12',width='512',height='3072')
el(a,'texture',name='carpet_tex',type='2d',builtin='checker',rgb1='.20 .21 .21',rgb2='.215 .225 .22',width='128',height='128')
el(a,'material',name='carpet',texture='carpet_tex',texrepeat='1 1',texuniform='true',reflectance='0')
world=el(model,'worldbody')
el(world,'light',pos='0 0 6',dir='0 0 -1',diffuse='.7 .7 .7',castshadow='false',directional='true')
walls=json.loads((OUT/'walls.json').read_text())
corners=np.array(walls['corners_xy_m']);height=walls['height_m'];thickness=walls['thickness_m']
lo=corners.min(0);hi=corners.max(0);mid=(lo+hi)/2
box(world,'floor',[*mid,-.08],[*(hi-lo+.3),.16],'carpet')
box(world,'ceiling',[*mid,height+.06],[*(hi-lo+.3),.12],'wall',group=4)
def wall_segment(name,p0,p1,z0=0,z1=None,group=2):
    p0=np.array(p0);p1=np.array(p1);direction=p1-p0;length=np.linalg.norm(direction)
    outward=np.array([direction[1],-direction[0]])/length
    center=(p0+p1)/2+outward*thickness/2;z1=height if z1 is None else z1
    geom=box(world,name,[*center,(z0+z1)/2],[length,thickness,z1-z0],'wall',group=group)
    geom.set('euler',f'0 0 {np.degrees(np.arctan2(direction[1],direction[0]))}')
wall_segment('east_wall',corners[1],corners[2],group=3)
wall_segment('south_wall',corners[2],corners[3],group=3)
wall_segment('west_wall',corners[3],corners[0],group=2)
north=walls['planes']['north']
def north_point(x): return np.array([x,north['slope']*x+north['offset']])
left=float(corners[0,0]);yaw=math.degrees(math.atan(north['slope']))
for i,door in enumerate(walls['doors']):
    x=door['x_center_m'];w=door['width_m'];h=door['height_m'];start=x-w/2;end=x+w/2
    wall_segment(f'north_wall_{i}',north_point(left),north_point(start))
    wall_segment(f'door_lintel{i}',north_point(start),north_point(end),z0=h)
    pos=north_point(start)+np.array([-north['slope'],1])*.035
    b=el(world,'body',name=door['name'],pos=vec([*pos,0]),euler=f'0 0 {yaw}')
    el(b,'joint',name=f'room_door_hinge{i+1}',type='hinge',axis='0 0 1',range='0 100',damping='4',frictionloss='1')
    box(b,f'room_door_panel{i}',[w/2,0,h/2],[w-.035,.045,h],'oak',mass=20)
    box(b,f'room_door_handle{i}',[w-.14,.06,1.05],[.13,.08,.025],'metal',mass=.15)
    left=end
wall_segment('north_wall_2',north_point(left),corners[1])
# Large objects re-estimated from manual_photo_samples and direct raw depth plan.
b=body('sofa',-1.55,2.30,0,size=[2.2,.9,.9])
box(b,'sofa_base',[0,0,.24],[2.2,.9,.30],'sofa',mass=28)
box(b,'sofa_back',[0,.34,.64],[2.2,.17,.44],'sofa',mass=12)
for s in [-1,1]:box(b,f'sofa_arm{s}',[s*1.015,0,.48],[.155,.9,.40],'sofa',mass=4)
for i in range(3):
    box(b,f'sofa_seat_cushion{i}',[(i-1)*.62,-.04,.45],[.6,.63,.12],'cushion',mass=2)
    box(b,f'sofa_back_cushion{i}',[(i-1)*.62,.235,.68],[.6,.12,.35],'cushion',mass=1)
legs(b,'sofa',2.0,.70,.1,thick=.05)
b=body('bed',4.12,-1.35,90,size=[1.96,1.06,.56])
box(b,'bed_plinth',[0,0,.22],[1.85,.99,.40],'oak',mass=32)
box(b,'mattress',[0,0,.48],[1.96,1.06,.16],'white',mass=8)
for i in [-1,1]:box(b,f'bed_drawer_face{i}',[i*.45,-.5,.22],[.86,.018,.3],'oak',visual=True)
# Separate pillow can be picked up and moved; no tiny clutter.
p=body('pillow',4.12,-1.93,90,size=[.55,.38,.13]);box(p,'pillow_geom',[0,0,.645],[.55,.38,.13],'paper',mass=.7)
desk=table('desk',4.80,.95,1.30,.72,.76,90)
box(desk,'desktop_monitor_stand',[0,.10,.86],[.20,.16,.16],'metal',mass=1)
box(desk,'desktop_monitor',[0,.14,1.12],[.49,.045,.31],'charcoal',mass=2)
box(desk,'desktop_monitor_screen',[0,.113,1.12],[.45,.008,.27],'display',visual=True)
chair('office_chair',4.12,.30,90)
chair('visitor_chair',-4.30,-3.10,180,False)
cabinet('white_cabinet',1.55,2.03,.85,.42,1.70,0)
cabinet('black_dresser',.85,2.04,.48,.42,1.18,0,mat='charcoal',drawers=True)
cabinet('gray_sideboard',-3.72,.77,.95,.65,.85,90,mat='sofa')
shelf('equipment_rack',.85,-3.77,.72,.43,.91,3,mat='charcoal')
shelf('kitchen_cart',.03,-3.77,.72,.49,.91,2,mat='white')
# Large stove top provides visual identity without modeling cables or small tools.
k=world.find("body[@name='kitchen_cart']")
box(k,'stovetop',[0,0,.96],[.72,.49,.07],'metal',mass=2)
for i,x in enumerate([-.18,.18]):cyl(k,f'burner{i}',[x,0,1.005],.105,.012,'charcoal',.1)
table('low_bench',-.92,-3.79,1.0,.40,.45,mat='charcoal')
shelf('tall_shelf',-2.12,-3.79,.73,.43,1.95,5,mat='oak')
# TV on its own mobile stand.
b=body('mobile_display',-4.0,-.95,90,size=[1.25,.68,1.9])
for s in [-1,1]:box(b,f'tv_foot{s}',[s*.42,0,.055],[.09,.70,.11],'metal',mass=6)
cyl(b,'tv_column',[0,0,.85],.045,1.5,mass=8)
box(b,'tv_frame',[0,0,1.58],[1.23,.07,.73],'charcoal',mass=7)
box(b,'tv_screen',[0,-.039,1.58],[1.16,.008,.65],'display',visual=True)
table('display_table',-4.38,-2.18,1.12,.61,.73,90,mat='white')
chair('display_chair',-3.48,-.83,-90,False)
shelf('bedside_cart',4.12,-2.73,.52,.48,.70,2,mat='oak')
shelf('utility_cart',4.46,-3.60,.67,.58,.62,2,mat='white')
b=body('large_box',4.46,-3.60,size=[.44,.38,.30]);box(b,'large_box_geom',[0,0,.80],[.44,.38,.30],'oak',mass=1.5)
# Three instances of ONE four-panel divider. Only poses and folds differ.
SCREEN_WIDTH = .46
SCREEN_HEIGHT = 1.80
SCREEN_INSTANCES = [
    ('sofa_screen', -3.80, 1.45, 50, [20, -25, 20]),
    ('center_screen', -.36, 1.85, 0, [35, 145, -35]),
    ('desk_screen', 2.05, 2.12, -15, [-30, 40, -30]),
]
def screen(n,x,y,yaw,folds):
    width=SCREEN_WIDTH
    root=body(n,x,y,yaw,free=False,size=[4*width,.035,SCREEN_HEIGHT],
              source='User: three identical dividers, four panels and three hinges each; raw photos for pose')
    inventory[-1].update(design='four_panel_divider',panels=4,hinges=3,
                         panel_width_m=width,panel_height_m=SCREEN_HEIGHT,
                         initial_hinge_angles_deg=folds,base_anchored=True)
    cur=root
    for i in range(4):
        if i:
            angle=folds[i-1]
            cur=el(cur,'body',name=f'{n}_leaf{i}',pos=f'{width} 0 0',euler=f'0 0 {angle}')
            # ref + body rotation make qpos equal the actual relative fold angle.
            el(cur,'joint',name=f'{n}_hinge{i}',type='hinge',axis='0 0 1',
               ref=angle,range='-175 175',damping='3',frictionloss='.6')
        box(cur,f'{n}_paper{i}',[width/2,0,.94],[width-.035,.025,1.69],'paper',mass=1.1)
        for side in [0,width]:
            box(cur,f'{n}_frame{i}_{side}',[side,0,.92],[.025,.035,1.80],'screenframe',mass=.5)
        for j,z in enumerate(np.linspace(.10,1.8,10)):
            box(cur,f'{n}_rail{i}_{j}',[width/2,-.022,z],[width,.016,.012],'screenframe',visual=True)
        for j,xx in enumerate(np.linspace(.10,width-.1,4)):
            box(cur,f'{n}_grid{i}_{j}',[xx,-.022,.94],[.009,.016,1.70],'screenframe',visual=True)
    return root
for args in SCREEN_INSTANCES:
    screen(*args)
# Large boards are attached to the fitted inner wall surfaces.
for i,(x,w) in enumerate([(1.9,1.2),(-.8,1.2)]):
    p=north_point(x)+np.array([0,.025])
    g=box(world,f'north_board{i}',[*p,2.03],[w,.035,.85],'paper',visual=True,group=2)
    g.set('euler',f'0 0 {yaw}')
east=walls['planes']['east'];g=box(world,'east_whiteboard',[east['offset']-.025,-.15,1.95],[.035,3.9,1.08],'paper',visual=True,group=3)
g.set('euler',f'0 0 {-math.degrees(math.atan(east["slope"]))}')
# Fixed camera views (camera local -Z points toward target).
def camera(n,pos,target):
    z=np.array(pos)-target;z=z/np.linalg.norm(z);x=np.cross([0,0,1],z);x=x/np.linalg.norm(x);y=np.cross(z,x)
    el(world,'camera',name=n,pos=vec(pos),xyaxes=vec([*x,*y]),fovy='48' if n == 'overview' else '58')
camera('overview',[-10,-11,12],[0,0,.4]);camera('interior',[0,-3.65,1.8],[0,1.1,.85]);camera('bed_area',[1,-1.2,2.3],[4.4,-1,.7]);camera('sofa_area',[-1,-1,2.2],[-1.6,2.3,.65])
contact=el(model,'contact')
for n,*_ in SCREEN_INSTANCES:
    for i in range(3):
        el(contact,'exclude',body1=n if i == 0 else f'{n}_leaf{i}',body2=f'{n}_leaf{i+1}')
ET.indent(model);ET.ElementTree(model).write(OUT/'scene.xml',encoding='unicode',xml_declaration=True)
measurements=json.loads((OUT/'raw_measurements.json').read_text())
(OUT/'inventory.json').write_text(json.dumps({
    'units':'meters; kilograms; seconds',
    'coordinates':measurements,
    'method':'Manual raw-photo interpretation; positions and dimensions estimated from raw RGB/depth and camera poses. No prebuilt reconstruction, object detection, or mesh input.',
    'assets':inventory},indent=2)+'\n')
print(f'Built scene.xml and inventory.json ({len(inventory)} assets; 3 identical four-panel dividers)')
