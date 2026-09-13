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
    b=body(n,x,y,yaw,size=[.60,.62,1.04])
    rounded(b,n+'_seat',[0,0,.46],[.48,.45,.07],.03,'charcoal',mass=3)
    if swivel:
        rounded(b,n+'_back',[0,.21,.79],[.46,.055,.49],.024,'charcoal',mass=1)
        rounded(b,n+'_mesh_back',[0,.175,.79],[.395,.012,.405],.005,'sofa',mass=.2)
    else:
        box(b,n+'_back',[0,.20,.75],[.43,.025,.15],'charcoal',mass=1)
        for side in [-1,1]:capsule(b,f'{n}_backpost{side}',[side*.20,.20,.41],[side*.20,.20,.82],.012,'metal')
    if swivel:
        cyl(b,n+'_column',[0,0,.24],.036,.38,mass=2)
        for i,a in enumerate(np.arange(5)*2*math.pi/5):
            el(b,'geom',name=f'{n}_spoke{i}',type='capsule',fromto=vec([0,0,.09,.26*math.cos(a),.26*math.sin(a),.055]),size='.018',material='metal',mass='.3')
            el(b,'geom',name=f'{n}_caster{i}',type='sphere',pos=vec([.26*math.cos(a),.26*math.sin(a),.032]),size='.032',material='charcoal',mass='.1',friction='.22 .005 .001')
        for s in [-1,1]:box(b,f'{n}_arm{s}',[s*.27,0,.65],[.045,.33,.045],'charcoal',mass=.4)
    else: legs(b,n,.45,.43,.42)
    return b

def capsule(b,n,p0,p1,r,mat='metal',mass=.15,visual=False):
    kw=dict(name=n,type='capsule',fromto=vec([*p0,*p1]),size=r,material=mat,mass=mass)
    if visual: kw.update(contype=0,conaffinity=0,mass=0)
    return el(b,'geom',**kw)

def mesh_geom(b,n,vertices,faces,pos,mat,mass=1,visual=False,uv=None):
    kw=dict(name=n+'_mesh',vertex=vec(np.asarray(vertices).ravel()),face=' '.join(str(int(i)) for i in np.asarray(faces).ravel()))
    if uv is not None:kw['texcoord']=vec(np.asarray(uv).ravel())
    el(a,'mesh',**kw)
    kw=dict(name=n,type='mesh',mesh=n+'_mesh',pos=vec(pos),material=mat,mass=mass)
    if visual:kw.update(contype=0,conaffinity=0,mass=0)
    return el(b,'geom',**kw)

def rounded(b,n,pos,size,r,mat,mass=1,visual=False):
    # Closed, convex rounded cuboid; vertices are shared across all six faces.
    half=np.array(size)/2;core=half-r;vertices=[];faces=[];lookup={}
    coords=[np.unique(np.r_[-h,-h+r*(1-np.cos(np.linspace(0,np.pi/2,5))),0,h-r*(1-np.cos(np.linspace(0,np.pi/2,5)))]) for h in half]
    def vertex(q):
        c=np.clip(q,-core,core);v=q-c;v=c+v/np.linalg.norm(v)*r;key=tuple(np.round(v,8))
        if key not in lookup:lookup[key]=len(vertices);vertices.append(v)
        return lookup[key]
    for axis in range(3):
        u=(axis+1)%3;v=(axis+2)%3
        for sign in [-1,1]:
            grid=[]
            for x in coords[u]:
                row=[]
                for y in coords[v]:
                    q=np.zeros(3);q[axis]=sign*half[axis];q[u]=x;q[v]=y;row.append(vertex(q))
                grid.append(row)
            for i in range(len(grid)-1):
                for j in range(len(grid[0])-1):
                    q=[grid[i][j],grid[i+1][j],grid[i+1][j+1],grid[i][j+1]]
                    if sign<0:q=q[::-1]
                    faces.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
    uv=[[(v[0]/size[0]+.5)*2,(v[1]/size[1]+v[2]/size[2]+1)*2] for v in vertices]
    return mesh_geom(b,n,vertices,faces,pos,mat,mass,visual,uv)

def taper(b,n,z0,z1,r0,r1,mat):
    vertices=[[r*np.cos(t),r*np.sin(t),z] for z,r in [(z0,r0),(z1,r1)] for t in np.arange(32)*2*np.pi/32]
    faces=[]
    for i in range(32):
        j=(i+1)%32;faces.extend([[i,j,j+32],[i,j+32,i+32]])
    for i in range(1,31):faces.extend([[0,i+1,i],[32,32+i,33+i]])
    return mesh_geom(b,n,vertices,faces,[0,0,0],mat,.35)

def floor_region(n,polygon,mat):
    if sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon)))<0:polygon=polygon[::-1]
    vertices=[[x,y,z] for z in [-.003,.001] for x,y in polygon];count=len(polygon);faces=[]
    for i in range(1,count-1):faces.extend([[0,i+1,i],[count,count+i,count+i+1]])
    for i in range(count):
        j=(i+1)%count;faces.extend([[i,j,j+count],[i,j+count,i+count]])
    mesh_geom(world,n,vertices,faces,[0,0,0],mat,visual=True,uv=[[x*3,y*3] for x,y,z in vertices])

def folding_chair(n,x,y,yaw=90):
    b=body(n,x,y,yaw,size=[.43,.48,.82])
    rounded(b,n+'_seat',[0,0,.45],[.41,.40,.035],.012,'white',1.3)
    rounded(b,n+'_back',[0,.18,.73],[.39,.028,.17],.012,'white',.8)
    for side in [-1,1]:
        capsule(b,f'{n}_front{side}',[side*.18,-.20,.018],[side*.18,.18,.81],.011,'chrome')
        capsule(b,f'{n}_rear{side}',[side*.18,.22,.018],[side*.18,-.12,.46],.011,'chrome')
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
        shelves=[.44,.83,1.24] if n=='white_cabinet' else [h*.5]
        for i,z in enumerate(shelves):box(b,f'{n}_shelf{i+1}',[0,.01,z],[w-.08,d-.05,.025],mat,mass=2)
        if n=='white_cabinet':inventory[-1].update(shelf_heights_m=shelves,source='Capture exterior and additional_data/WhiteCabinet_view1.jpg: three internal shelves; heights scaled from the photographed cabinet frame.')
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
colors={'white':'.89 .89 .86 1','wall':'.86 .85 .79 1','oak':'.68 .52 .31 1','charcoal':'.055 .065 .075 1','metal':'.17 .19 .20 1','sofa':'.36 .37 .38 1','cushion':'.43 .44 .45 1','brass':'.53 .44 .24 1','screenframe':'.29 .055 .045 1','paper':'.9 .9 .84 1','display':'.12 .32 .43 1','orange':'.96 .17 .025 1','blue':'.035 .12 .6 1'}
colors.update(concrete='.72 .70 .66 1',chrome='.60 .62 .63 1',red='.72 .025 .025 1',cardboard='.56 .40 .22 1',tape='.66 .50 .28 1',linen='.83 .83 .79 1',blanket='.40 .41 .41 1',frosted='.65 .68 .66 .86',sink='.32 .33 .32 1',steel='.57 .59 .59 1')
for n,c in colors.items():el(a,'material',name=n,rgba=c,specular='.15',shininess='.1')
el(a,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.25 .29 .34',rgb2='.075 .095 .12',width='512',height='3072')
for name,base,fiber in [('carpet','.27 .27 .265','.43 .43 .42'),('carpet_dark','.18 .18 .18','.26 .26 .26'),('carpet_mid','.205 .205 .205','.28 .28 .28')]:
    el(a,'texture',name=name+'_tex',type='2d',builtin='flat',rgb1=base,mark='random',markrgb=fiber,random='.42',width='256',height='256')
    el(a,'material',name=name,texture=name+'_tex',texrepeat='3 3',texuniform='true',reflectance='0',specular='0')
for name in ['sofa','cushion','linen','blanket','red','concrete']:
    el(a,'texture',name=name+'_weave',type='2d',builtin='flat',rgb1='.98 .98 .98',mark='random',markrgb='.86 .86 .86',random='.28',width='128',height='128')
    a.find(f"material[@name='{name}']").set('texture',name+'_weave')
    a.find(f"material[@name='{name}']").set('specular','0.02')
for name in ['chrome','steel']:
    mat=a.find(f"material[@name='{name}']");mat.set('specular','.65');mat.set('shininess','.5')
world=el(model,'worldbody')
el(world,'light',pos='0 0 6',dir='0 0 -1',diffuse='.7 .7 .7',castshadow='false',directional='true')
walls=json.loads((OUT/'walls.json').read_text())
corners=np.array(walls['corners_xy_m']);height=walls['height_m'];thickness=walls['thickness_m']
lo=corners.min(0);hi=corners.max(0);mid=(lo+hi)/2
box(world,'floor',[*mid,-.081],[*(hi-lo+.3),.16],'carpet')
# Long straight carpet boundaries traced in the RGB-D floor projection.
# Extend into occluded margins; the physical support remains one continuous floor.
floor_region('floor_dark_diagonal',[[-4.2,0],[-1.6,0],[1.2,-2.8],[-1.4,-2.8]],'carpet_dark')
floor_region('floor_wall_band',[[-1.4,-2.8],[4.0,-2.8],[2.3,-4.5],[.3,-4.5]],'carpet_dark')
floor_region('floor_mid_diagonal',[[-1.6,0],[5.5,0],[5.5,4.5],[2.1,4.5]],'carpet_mid')
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
    inventory.append(dict(name=door['name'],movable=False,position_m=[*pos,0],yaw_deg=yaw,dimensions_m=[w,.045,h],source='Raw RGB-D doorway observations; hinged room door.'))
    el(b,'joint',name=f'room_door_hinge{i+1}',type='hinge',axis='0 0 1',range='0 100',damping='4',frictionloss='1')
    box(b,f'room_door_panel{i}',[w/2,0,h/2],[w-.035,.045,h],'oak',mass=20)
    box(b,f'room_door_handle{i}',[w-.14,.06,1.05],[.13,.08,.025],'metal',mass=.15)
    left=end
wall_segment('north_wall_2',north_point(left),corners[1])
# Large objects re-estimated from manual_photo_samples and direct raw depth plan.
b=body('sofa',-1.55,2.30,0,size=[2.2,.9,.9])
rounded(b,'sofa_base',[0,0,.24],[2.2,.9,.25],.055,'sofa',mass=28)
box(b,'sofa_back',[0,.34,.64],[2.2,.17,.44],'sofa',mass=12)
for s in [-1,1]:rounded(b,f'sofa_arm{s}',[s*1.015,0,.50],[.155,.9,.43],.045,'sofa',mass=4)
for i in range(3):
    rounded(b,f'sofa_seat_cushion{i}',[(i-1)*.635,-.075,.43],[.625,.69,.14],.045,'cushion',mass=2)
    rounded(b,f'sofa_back_cushion{i}',[(i-1)*.635,.255,.69],[.625,.16,.40],.05,'cushion',mass=1).set('euler','-10 0 0')
legs(b,'sofa',2.0,.70,.1,thick=.05)
b=body('bed',4.12,-1.10,90,size=[1.96,1.06,.485])
for i in [-1,1]:
    box(b,f'bed_box{i}',[i*.49,0,.1675],[.95,.98,.335],'cardboard',mass=16)
    box(b,f'bed_tape{i}',[i*.49,-.493,.1675],[.065,.004,.335],'tape',visual=True)
g=rounded(b,'mattress',[0,0,.410],[1.96,1.06,.15],.07,'linen',mass=8)
g.set('contype','0');g.set('conaffinity','0')
box(b,'mattress_contact',[0,0,.410],[1.82,.92,.15],'white',mass=0).set('rgba','0 0 0 0')
p=body('pillow',4.00,-1.65,75,size=[.62,.41,.16])
rounded(p,'pillow_piping',[0,0,.52],[.64,.43,.034],.015,'charcoal',mass=.1)
rounded(p,'pillow_geom',[0,0,.57],[.62,.41,.17],.08,'linen',mass=.6)
b=body('folded_blanket',4.10,-.52,80,size=[.52,.40,.19])
for j in range(2):rounded(b,f'blanket_fold{j}',[0,0,.53+j*.075],[.51-j*.025,.39,.075],.035,'blanket',mass=.5)
desk=table('desk',4.80,.95,1.30,.72,.76,90)
box(desk,'desktop_monitor_stand',[0,.10,.86],[.20,.16,.16],'metal',mass=1)
box(desk,'desktop_monitor',[0,.14,1.12],[.49,.045,.31],'charcoal',mass=2)
box(desk,'desktop_monitor_screen',[0,.113,1.12],[.45,.008,.27],'display',visual=True)
chair('office_chair',4.12,.30,90)
# Frame 32: round folding stool, separate red tool bag, striped cone.
b=body('round_stool',-4.20,-3.12,size=[.43,.43,.61])
g=cyl(b,'stool_seat',[0,0,.593],.215,.034,'white',1.8);g.set('contype','0');g.set('conaffinity','0')
box(b,'stool_seat_contact',[0,0,.593],[.30,.30,.034],'white',mass=0).set('rgba','0 0 0 0')
for side in [-1,1]:
    capsule(b,f'stool_leg_a{side}',[side*.145,-.17,.015],[side*.145,.14,.577],.009,'chrome')
    capsule(b,f'stool_leg_b{side}',[side*.145,.17,.015],[side*.145,-.14,.577],.009,'chrome')
    capsule(b,f'stool_foot{side}',[-.145,side*.17,.015],[.145,side*.17,.015],.009,'chrome')
b=body('red_tool_bag',-4.20,-3.12,90,size=[.32,.22,.22])
rounded(b,'bag_body',[0,0,.72],[.32,.22,.22],.05,'red',mass=1.0)
box(b,'bag_base',[0,0,.616],[.31,.21,.012],'charcoal',mass=.1)
for side in [-1,1]:
    capsule(b,f'bag_handle_left{side}',[-.09,side*.06,.79],[-.075,side*.06,.87],.012,'charcoal',visual=True)
    capsule(b,f'bag_handle_top{side}',[-.075,side*.06,.87],[.075,side*.06,.87],.012,'charcoal',visual=True)
    capsule(b,f'bag_handle_right{side}',[.075,side*.06,.87],[.09,side*.06,.79],.012,'charcoal',visual=True)
b=body('traffic_cone',-3.95,-3.78,size=[.34,.34,.78])
rounded(b,'cone_base',[0,0,.025],[.34,.34,.05],.02,'red',mass=2)
levels=[.05,.35,.45,.55,.65,.74]
for j,(z0,z1) in enumerate(zip(levels,levels[1:])):
    radius=lambda z:.15-(z-.05)/.69*.128
    taper(b,f'cone_band{j}',z0,z1,radius(z0),radius(z1),'white' if j in [1,3] else 'red')
cyl(b,'cone_tip',[0,0,.755],.018,.03,'red',.1)
folding_chair('visitor_chair',-4.85,-3.15)
chair('workstation_chair',-4.87,-2.13,90)
folding_chair('workstation_guest_chair',-4.83,-1.12)

cabinet('white_cabinet',1.55,2.03,.85,.42,1.70,0)
cabinet('black_dresser',.85,2.04,.48,.42,1.18,0,mat='charcoal',drawers=True)
cabinet('sink_cabinet',-3.72,.77,.95,.65,.85,90,mat='sink')
b=world.find("body[@name='sink_cabinet']")
# Raised stainless rim, recessed basin and ribbed draining board.
box(b,'sink_counter',[0,0,.86],[1.00,.68,.035],'steel',mass=2)
box(b,'sink_basin',[-.23,-.015,.882],[.39,.46,.008],'charcoal',visual=True)
for x in [-.455,-.015]:box(b,f'sink_rim{x}',[x,0,.9],[.025,.60,.035],'chrome',visual=True)
for y in [-.29,.29]:box(b,f'sink_edge{y}',[-.235,y,.9],[.46,.025,.035],'chrome',visual=True)
for j in range(10):box(b,f'drainer_rib{j}',[.045+j*.04,0,.886],[.012,.52,.007],'chrome',visual=True)
for side,label in [(-1,'left'),(1,'right')]:
    door=b.find(f"body[@name='sink_cabinet_{label}_door']")
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('size','.15 .02 .01')
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('pos',vec([-side*.19,-.032,.61]))
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('material','chrome')
    for j in range(5):box(door,f'sink_door_rib{label}{j}',[-side*(.055+j*.075),-.017,.36],[.005,.004,.60],'metal',visual=True)
box(b,'sink_splashback',[0,.30,1.11],[.94,.022,.43],'steel',mass=1)
capsule(b,'faucet_stem',[-.24,.22,.9],[-.24,.22,1.25],.015,'chrome')
capsule(b,'faucet_arch',[-.24,.22,1.25],[-.24,.08,1.30],.015,'chrome')
capsule(b,'faucet_spout',[-.24,.08,1.30],[-.24,-.02,1.25],.015,'chrome')
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
for side in [-1,1]:
    box(b,f'tv_foot{side}',[side*.45,0,.09],[.075,.68,.065],'metal',mass=6)
    for y in [-.28,.28]:el(b,'geom',name=f'tv_caster{side}_{y}',type='sphere',pos=vec([side*.45,y,.04]),size='.04',material='charcoal',mass='.2')
    cyl(b,f'tv_column{side}',[side*.11,.10,1.01],.025,1.82,'chrome',4)
box(b,'tv_crossbar',[0,0,.09],[.95,.08,.065],'metal',mass=2)
box(b,'tv_frame',[0,0,1.58],[1.23,.055,.73],'charcoal',mass=7)
box(b,'tv_screen',[0,-.032,1.58],[1.16,.008,.65],'display',visual=True)
# Large screen appearance only; no source screenshots or legible interface text.
box(b,'tv_window',[.035,-.038,1.58],[1.02,.003,.60],'paper',visual=True)
for j in range(9):box(b,f'tv_text_line{j}',[.03,-.041,1.78-j*.045],[.86 if j%3 else .65,.002,.006],'sofa',visual=True)
box(b,'tv_tray',[0,-.11,.92],[.53,.34,.025],'charcoal',mass=1)
b=body('display_table',-4.55,-2.12,90,size=[1.35,.72,1.15])
box(b,'display_table_top',[0,0,.735],[1.35,.72,.035],'oak',mass=9)
for side in [-1,1]:
    box(b,f'display_table_column{side}',[side*.55,0,.365],[.065,.065,.70],'metal',mass=2)
    box(b,f'display_table_foot{side}',[side*.55,0,.025],[.085,.67,.05],'chrome',mass=2)
rounded(b,'desk_privacy_panel',[0,-.30,.955],[1.20,.018,.40],.008,'frosted',mass=1)
for side in [-1,1]:box(b,f'privacy_clip{side}',[side*.52,-.30,.77],[.055,.03,.075],'chrome',visual=True)

chair('display_chair',-3.48,-.83,-90,False)
b=body('bedside_cart',4.05,-2.45,size=[.52,.48,.77])
legs(b,'laundry_frame',.49,.45,.68,mat='chrome',thick=.022)
box(b,'laundry_bottom',[0,0,.17],[.46,.42,.035],'chrome',mass=2)
for x in [-.23,.23]:box(b,f'laundry_side{x}',[x,0,.47],[.018,.44,.57],'linen',mass=.5)
for y in [-.21,.21]:box(b,f'laundry_face{y}',[0,y,.47],[.46,.018,.57],'linen',mass=.5)
for x in [-.25,.25]:box(b,f'laundry_rim{x}',[x,0,.755],[.035,.48,.055],'sofa',mass=.2)
for y in [-.225,.225]:box(b,f'laundry_end{y}',[0,y,.755],[.52,.035,.055],'sofa',mass=.2)

shelf('utility_cart',4.46,-3.60,.67,.58,.62,2,mat='white')
b=body('large_box',4.46,-3.60,size=[.44,.38,.30]);box(b,'large_box_geom',[0,0,.80],[.44,.38,.30],'oak',mass=1.5)
# Soft goods use flat contact proxies beneath their rounded visual surfaces.
for name in ['pillow','folded_blanket','red_tool_bag']:
    b=world.find(f"body[@name='{name}']")
    for g in b.findall('geom'):
        if g.get('type')=='mesh':g.set('contype','0');g.set('conaffinity','0')
    if name!='red_tool_bag':
        proxy=box(b,name+'_contact',[0,0,.57 if name=='pillow' else .57],[.50,.30,.17 if name=='pillow' else .15],'white',mass=0)
        proxy.set('rgba','0 0 0 0')
    joint=b.find('freejoint');joint.tag='joint';joint.set('type','free');joint.set('damping','.12')
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
# The desk was concealed in the capture; extra photos document its geometry,
# not its later photographed position or the tabletop clutter.
b=body('concealed_desk',3.05,3.0,0,size=[1.50,.65,.74],source='additional_data/Desk_view1.jpg and Desk_view2.jpg: wood top, two pedestal legs and casters. Approximate captured placement behind desk_screen.')
box(b,'concealed_desk_top',[0,0,.725],[1.50,.65,.03],'oak',mass=10)
for side in [-1,1]:
    cyl(b,f'concealed_desk_column{side}',[side*.56,0,.395],.047,.62,'charcoal',3)
    box(b,f'concealed_desk_foot{side}',[side*.56,0,.08],[.075,.64,.045],'charcoal',mass=1)
    for end in [-1,1]:el(b,'geom',name=f'concealed_desk_wheel{side}_{end}',type='sphere',pos=vec([side*.56,end*.275,.035]),size='.035',material='charcoal',mass='.15')
# Inclined concrete support behind the sofa, fitted to frames 42–46.
b=body('diagonal_pillar',0,0,free=False,size=[2.14,.65,height],source='Visible pillar edges and depth samples in recorded frames 42–46; unseen back face extrapolated.')
vertices=[[x+center,y,z] for z,center in [(0,-2.12),(height,-.78)] for x,y in [(-.4,2.80),(.4,2.80),(.4,3.45),(-.4,3.45)]]
faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
mesh_geom(b,'concrete_pillar',vertices,faces,[0,0,0],'concrete',mass=300,uv=[[x,z] for x,y,z in vertices])

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
