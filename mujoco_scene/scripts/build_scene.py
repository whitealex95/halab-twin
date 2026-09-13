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

def caster(b,n,x,y,r=.04,mat='charcoal'):
    contact=el(b,'geom',name=n+'_contact',type='sphere',pos=vec([x,y,r]),size=r,rgba='0 0 0 0',mass='.12',friction='.24 .005 .001')
    wheel=cyl(b,n+'_tire',[x,y,r],r,.028,mat,.10);wheel.set('euler','90 0 0');wheel.set('contype','0');wheel.set('conaffinity','0')
    for side in [-1,1]:
        hub=cyl(b,n+f'_hub{side}',[x,y+side*.015,r],r*.43,.003,'chrome',.015);hub.set('euler','90 0 0');hub.set('contype','0');hub.set('conaffinity','0')
        box(b,n+f'_fork{side}',[x,y+side*.022,r*1.35],[r*.5,.006,r*.9],'metal',visual=True)
    box(b,n+'_mount',[x,y,r*1.9],[r,.055,.014],'metal',visual=True)

def chair(n,x,y,yaw=0,swivel=True):
    b=body(n,x,y,yaw,size=[.60,.62,1.12 if swivel else .82])
    rounded(b,n+'_seat',[0,0,.46],[.48,.45,.07],.03,'charcoal',mass=3)
    if swivel:
        # A curved mesh back with a narrower lumbar region and a tubular rim.
        rows=np.linspace(0,1,11);vertices=[]
        def profile(t):return .178+.040*t-.023*math.sin(math.pi*t)
        def back_y(t):return .21+.075*t*t
        for face in [-1,1]:
            for t in rows:
                for u in np.linspace(-1,1,9):vertices.append([u*profile(t),back_y(t)+face*.006+.012*(1-u*u),.54+t*.56])
        faces=[];layer=len(rows)*9
        for face in range(2):
            for j in range(10):
                for i in range(8):
                    a0=face*layer+j*9+i;q=[a0,a0+1,a0+10,a0+9]
                    if face==0:q=q[::-1]
                    faces.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
        edge=list(range(9))+[j*9+8 for j in range(1,11)]+list(range(97,89,-1))+[j*9 for j in range(9,0,-1)]
        for j,a0 in enumerate(edge):
            a1=edge[(j+1)%len(edge)];faces.extend([[a0,a1,a1+layer],[a0,a1+layer,a0+layer]])
        mesh_geom(b,n+'_mesh_back',vertices,faces,[0,0,0],'chair_mesh',mass=.8)
        for side in [-1,1]:
            for j,(t0,t1) in enumerate(zip(rows,rows[1:])):capsule(b,n+f'_back_rim{side}_{j}',[side*profile(t0),back_y(t0),.54+t0*.56],[side*profile(t1),back_y(t1),.54+t1*.56],.014,'charcoal',mass=.035)
        for t in [0,1]:capsule(b,n+f'_back_end{t}',[-profile(t),back_y(t),.54+t*.56],[profile(t),back_y(t),.54+t*.56],.014,'charcoal',mass=.12)
        capsule(b,n+'_back_support',[0,.19,.46],[0,.22,.62],.025,'charcoal',mass=.4)

    else:
        box(b,n+'_back',[0,.20,.75],[.43,.025,.15],'charcoal',mass=1)
        for side in [-1,1]:capsule(b,f'{n}_backpost{side}',[side*.20,.20,.41],[side*.20,.20,.82],.012,'metal')
    if swivel:
        cyl(b,n+'_column',[0,0,.24],.036,.38,mass=2)
        for i,a in enumerate(np.arange(5)*2*math.pi/5):
            el(b,'geom',name=f'{n}_spoke{i}',type='capsule',fromto=vec([0,0,.09,.26*math.cos(a),.26*math.sin(a),.055]),size='.018',material='metal',mass='.3')
            el(b,'geom',name=f'{n}_caster{i}',type='sphere',pos=vec([.26*math.cos(a),.26*math.sin(a),.032]),size='.032',material='charcoal',mass='.1',friction='.22 .005 .001')
        for side in [-1,1]:
            rounded(b,f'{n}_arm{side}',[side*.26,0,.66],[.065,.28,.035],.016,'charcoal',mass=.25)
            capsule(b,f'{n}_arm_support{side}',[side*.245,.08,.46],[side*.245,.08,.64],.014,'metal',mass=.12)
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
        # Five open-top boxes fit inside the carcass, with clearance at the top,
        # back and side runners. Keep part of each box supported when extended.
        pitch=(h-t-.004-.104)/5
        drawer_width=w-2*t-.020
        front_y=-d/2+.0125
        back_y=d/2-t-.013
        depth=back_y-front_y
        for i in range(5):
            z=.104+pitch*(i+.5);dh=pitch-.006;bottom_z=-dh/2+.014;wall_h=dh-.035
            dr=el(b,'body',name=f'{n}_drawer{i+1}',pos=vec([0,0,z]))
            el(dr,'joint',name=f'{n}_slide{i+1}',type='slide',axis='0 -1 0',range=f'0 {d*.72}',damping='8',frictionloss='2')
            box(dr,f'{n}_drawer_front{i}',[0,-d/2,0],[w-2*t-.004,.025,dh],mat,mass=.8)
            box(dr,f'{n}_drawer_tray{i}',[0,(front_y+back_y)/2,bottom_z],[drawer_width,depth,.010],mat,mass=.3)
            for side in [-1,1]:
                box(dr,f'{n}_drawer_side{i}_{side}',[side*(drawer_width/2-.004),(front_y+back_y)/2,bottom_z+.005+wall_h/2],[.008,depth,wall_h],mat,mass=.2)
                box(dr,f'{n}_moving_runner{i}_{side}',[side*(drawer_width/2+.002),(front_y+back_y)/2,bottom_z+.035],[.004,depth-.025,.020],'metal',mass=.04)
                box(b,f'{n}_fixed_runner{i}_{side}',[side*(drawer_width/2+.007),(front_y+back_y)/2,z+bottom_z+.035],[.004,depth-.025,.025],'metal',mass=.05)
            box(dr,f'{n}_drawer_back{i}',[0,back_y-.004,bottom_z+.005+wall_h/2],[drawer_width-.016,.008,wall_h],mat,mass=.2)
            box(dr,f'{n}_drawer_handle{i}',[0,-d/2-.028,0],[w*.3,.03,.018],'metal',mass=.1)
        inventory[-1].update(drawer_count=5,drawer_travel_m=d*.72,drawer_construction='Open-top boxes with bottom, side and back walls; paired stationary/moving runners and carcass clearance.')
    else:
        shelves=[.44,.83,1.24] if n=='white_cabinet' else [h*.5]
        for i,z in enumerate(shelves):box(b,f'{n}_shelf{i+1}',[0,.01,z],[w-.08,d-.05,.025],mat,mass=2)
        if n=='white_cabinet':inventory[-1].update(shelf_heights_m=shelves,source='Capture exterior and additional_data/WhiteCabinet_view1.jpg: three internal shelves; heights scaled from the photographed cabinet frame.')
        for s,label in [(-1,'left'),(1,'right')]:
            door=el(b,'body',name=n+'_'+label+'_door',pos=vec([s*(w/2+.002),-d/2-.016,.1]))
            el(door,'joint',name=n+'_'+label+'_hinge',type='hinge',axis='0 0 1',range='-180 0' if s==-1 else '0 180',damping='2',frictionloss='.4')
            box(door,n+'_'+label+'_panel',[-s*w/4,0,(h-.12)/2],[(w-.008)/2,.025,h-.12],mat,mass=2)
            box(door,n+'_'+label+'_handle',[-s*(w/2-.10),-.032,h*.48],[.022,.04,.15],'brass',mass=.08)
            if n=='white_cabinet':
                for k,(zz,hh) in enumerate([(.38,.62),(1.16,.69)]):
                    box(door,f'{n}_{label}_inset{k}',[-s*w/4,-.0135,zz],[(w-.008)/2-.075,.002,hh],'panel_inset',visual=True)
                    for side in [-1,1]:box(door,f'{n}_{label}_trim{k}_{side}',[-s*w/4,-.019,zz+side*hh/2],[(w-.008)/2-.07,.008,.012],'white',visual=True)
    return b

model=ET.Element('mujoco',model='HaLab — image-based interactive reconstruction')
el(model,'compiler',angle='degree',autolimits='true')
el(model,'option',timestep='.002',integrator='implicitfast',gravity='0 0 -9.81',cone='elliptic',iterations='60')
el(model,'size',njmax='8000',nconmax='2000')
el(model,'statistic',center='0 0 .8',extent='6.8')
v=el(model,'visual');el(v,'global',offwidth='1600',offheight='1200',azimuth='125',elevation='-48');el(v,'headlight',ambient='.72 .72 .72',diffuse='.15 .15 .15',specular='.02 .02 .02');el(v,'rgba',haze='.12 .15 .19 1')
d=el(model,'default');el(d,'geom',friction='.7 .01 .001',solref='.015 1',solimp='.95 .99 .001',condim='4');el(d,'joint',damping='.5')
a=el(model,'asset')
colors={'white':'.89 .89 .86 1','wall':'.88 .87 .81 1','north_paint':'.90 .88 .85 1','west_paint':'.86 .82 .73 1','oak':'.68 .52 .31 1','charcoal':'.055 .065 .075 1','metal':'.17 .19 .20 1','chair_mesh':'.22 .235 .24 1','panel_inset':'.87 .87 .84 1','cream_canvas':'.73 .67 .53 1','piping':'.25 .28 .32 1','screen_light':'.81 .91 .97 1','sofa':'.41 .42 .43 1','cushion':'.49 .50 .51 1','brass':'.53 .44 .24 1','screenframe':'.30 .12 .10 1','paper':'.98 .97 .91 1','display':'.12 .32 .43 1','orange':'.96 .17 .025 1','blue':'.035 .12 .6 1'}
colors.update(shelf_wood='.32 .225 .15 1',concrete='.72 .70 .66 1',chrome='.60 .62 .63 1',red='.72 .025 .025 1',cardboard='.65 .52 .36 1',tape='.66 .50 .28 1',linen='.83 .83 .79 1',blanket='.52 .54 .54 1',frosted='.65 .68 .66 .86',sink='.32 .33 .32 1',steel='.57 .59 .59 1')
for n,c in colors.items():el(a,'material',name=n,rgba=c,specular='.15',shininess='.1')
el(a,'texture',name='shelf_grain',type='2d',builtin='flat',rgb1='.94 .89 .81',mark='random',markrgb='.65 .59 .52',random='.35',width='32',height='512')
a.find("material[@name='shelf_wood']").set('texture','shelf_grain')
a.find("material[@name='shelf_wood']").set('specular','.06')
el(a,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.25 .29 .34',rgb2='.075 .095 .12',width='512',height='3072')
for name,base,fiber in [('carpet','.26 .26 .255','.39 .39 .38'),('carpet_dark','.23 .23 .23','.32 .32 .32'),('carpet_mid','.245 .245 .245','.35 .35 .35')]:
    el(a,'texture',name=name+'_tex',type='2d',builtin='flat',rgb1=base,mark='random',markrgb=fiber,random='.42',width='256',height='256')
    el(a,'material',name=name,texture=name+'_tex',texrepeat='3 3',texuniform='true',reflectance='0',specular='0')
for name in ['sofa','cushion','linen','blanket','red','concrete','cream_canvas','chair_mesh']:
    el(a,'texture',name=name+'_weave',type='2d',builtin='flat',rgb1='.98 .98 .98',mark='random',markrgb='.86 .86 .86',random='.28',width='128',height='128')
    a.find(f"material[@name='{name}']").set('texture',name+'_weave')
    a.find(f"material[@name='{name}']").set('specular','0.02')
    a.find(f"material[@name='{name}']").set('texuniform','true')
    a.find(f"material[@name='{name}']").set('texrepeat','4 4')
for name in ['chrome','steel']:
    mat=a.find(f"material[@name='{name}']");mat.set('specular','.65');mat.set('shininess','.5')
el(a,'texture',name='oak_grain',type='2d',builtin='flat',rgb1='.99 .98 .96',mark='random',markrgb='.92 .90 .86',random='.22',width='128',height='1024')
a.find("material[@name='oak']").set('texture','oak_grain')
a.find("material[@name='screen_light']").set('emission','.12')
world=el(model,'worldbody')
# Shared native/offscreen lighting: mostly diffuse room fill with a gentle camera light.
# North-wall fill approximates the ceiling fixtures without view-dependent exposure.
el(world,'light',name='ceiling_fill',pos='0 0 6',dir='0 0 -1',diffuse='.16 .16 .16',specular='.02 .02 .02',castshadow='false',directional='true')
el(world,'light',name='north_fill',pos='0 0 3',dir='0 -1 -.1',diffuse='.15 .15 .15',specular='0 0 0',castshadow='false',directional='true')
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
    geom=box(world,name,[*center,(z0+z1)/2],[length,thickness,z1-z0],'north_paint' if name.startswith(('north_wall','door_lintel')) else 'west_paint' if name=='west_wall' else 'wall',group=group)
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
    for face in [-1,1]:
        plate=cyl(b,f'room_door_rose{i}_{face}',[w-.14,face*.027,1.05],.027,.005,'chrome',.03);plate.set('euler','90 0 0')
        capsule(b,f'room_door_handle_stem{i}_{face}',[w-.14,face*.03,1.05],[w-.14,face*.065,1.05],.009,'chrome',mass=.03)
        capsule(b,f'room_door_handle{i}_{face}',[w-.14,face*.065,1.05],[w-.25,face*.065,1.035],.009,'chrome',mass=.06)
    for j,zz in enumerate([.25,1.2,2.2]):box(b,f'room_door_hardware{i}_{j}',[.012,.027,zz],[.022,.008,.065],'chrome',visual=True)
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
for i,(center,width) in enumerate([(-.34,1.27),(.64,.63)]):
    box(b,f'bed_box{i}',[center,0,.1675],[width,.98,.335],'cardboard',mass=16 if i==0 else 9)
    box(b,f'bed_tape{i}',[center,-.493,.1675],[.055,.003,.335],'tape',visual=True)
    box(b,f'bed_box_seam{i}',[center,-.495,.275],[width-.03,.002,.004],'tape',visual=True)
g=rounded(b,'mattress',[0,0,.410],[1.96,1.06,.15],.07,'linen',mass=8)
g.set('contype','0');g.set('conaffinity','0')
box(b,'mattress_contact',[0,0,.410],[1.82,.92,.15],'white',mass=0).set('rgba','0 0 0 0')
p=body('pillow',4.00,-1.65,75,size=[.62,.41,.16])
rounded(p,'pillow_piping',[0,0,.523],[.64,.43,.016],.007,'piping',mass=.1)
rounded(p,'pillow_geom',[0,0,.57],[.62,.41,.17],.08,'linen',mass=.6)
b=body('folded_blanket',4.10,-.52,80,size=[.52,.40,.19])
rounded(b,'blanket_fold0',[0,0,.53],[.51,.39,.065],.032,'blanket',mass=.3)
for j,yy in enumerate([-.09,.065]):rounded(b,f'blanket_roll{j}',[0,yy,.60+j*.025],[.49-j*.035,.245,.15],.07,'blanket',mass=.35).set('euler',f'{-8+j*16} 0 0')
desk=body('desk',4.80,.95,-90,size=[1.30,.72,.76],source='Recorded frames 10, 85, 175: monitor faces into the room; wood top on two metal T legs.')
rounded(desk,'desk_top',[0,0,.7425],[1.30,.72,.035],.012,'oak',mass=9)
for side in [-1,1]:
    box(desk,f'desk_column{side}',[side*.51,0,.39],[.06,.075,.67],'metal',mass=2)
    box(desk,f'desk_foot{side}',[side*.51,0,.042],[.09,.65,.06],'metal',mass=2)
box(desk,'desktop_monitor_stand',[0,.10,.86],[.20,.16,.16],'metal',mass=1)
box(desk,'desktop_monitor',[0,.14,1.12],[.56,.04,.34],'charcoal',mass=2)
box(desk,'desktop_monitor_screen',[0,.113,1.12],[.52,.008,.30],'display',visual=True)
chair('office_chair',4.12,.30,100)
# Frame 32: round folding stool, separate red tool bag, striped cone.
b=body('round_stool',-4.20,-3.12,size=[.43,.43,.61])
g=cyl(b,'stool_seat',[0,0,.593],.215,.034,'white',1.8);g.set('contype','0');g.set('conaffinity','0')
box(b,'stool_seat_contact',[0,0,.593],[.30,.30,.034],'white',mass=0).set('rgba','0 0 0 0')
for side in [-1,1]:
    capsule(b,f'stool_leg_a{side}',[side*.145,-.17,.015],[side*.145,.14,.577],.009,'chrome')
    capsule(b,f'stool_leg_b{side}',[side*.145,.17,.015],[side*.145,-.14,.577],.009,'chrome')
    capsule(b,f'stool_foot{side}',[-.145,side*.17,.015],[.145,side*.17,.015],.009,'chrome')
b=body('red_tool_bag',-4.20,-3.12,90,size=[.32,.22,.22])
rounded(b,'bag_body',[0,0,.695],[.32,.22,.15],.035,'red',mass=1.0)
box(b,'bag_base',[0,0,.616],[.31,.21,.012],'charcoal',mass=.1)
for side in [-1,1]:
    capsule(b,f'bag_handle_left{side}',[-.09,side*.06,.75],[-.075,side*.06,.82],.012,'charcoal',visual=True)
    capsule(b,f'bag_handle_top{side}',[-.075,side*.06,.82],[.075,side*.06,.82],.012,'charcoal',visual=True)
    capsule(b,f'bag_handle_right{side}',[.075,side*.06,.82],[.09,side*.06,.75],.012,'charcoal',visual=True)
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
# Counter surrounds the recessed bowl; the right half is the draining board.
b.remove(b.find("geom[@name='sink_cabinet_top']"))
box(b,'sink_counter',[.25,0,.86],[.50,.68,.035],'steel',mass=1)
for yy in [-.285,.285]:box(b,f'sink_counter_edge{yy}',[-.25,yy,.86],[.50,.11,.035],'steel',mass=.3)
box(b,'sink_counter_left',[-.485,0,.86],[.03,.68,.035],'steel',mass=.2)
box(b,'sink_basin',[-.23,0,.755],[.40,.46,.014],'steel',mass=.4)
for xx in [-.435,-.025]:box(b,f'sink_bowl_side{xx}',[xx,0,.806],[.012,.46,.11],'steel',mass=.15)
for yy in [-.235,.235]:box(b,f'sink_bowl_end{yy}',[-.23,yy,.806],[.42,.012,.11],'steel',mass=.15)
cyl(b,'sink_drain',[-.23,0,.764],.026,.003,'charcoal',.02)
for x in [-.455,-.015]:box(b,f'sink_rim{x}',[x,0,.9],[.025,.60,.035],'chrome',visual=True)
for y in [-.29,.29]:box(b,f'sink_edge{y}',[-.235,y,.9],[.46,.025,.035],'chrome',visual=True)
for j in range(10):box(b,f'drainer_rib{j}',[.045+j*.04,0,.886],[.012,.52,.007],'chrome',visual=True)
for side,label in [(-1,'left'),(1,'right')]:
    door=b.find(f"body[@name='sink_cabinet_{label}_door']")
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('size','.011 .02 .08' if side==-1 else '.15 .02 .01')
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('pos',vec([.40 if side==-1 else -.235,-.032,.43 if side==-1 else .61]))
    door.find(f"geom[@name='sink_cabinet_{label}_handle']").set('material','chrome')
    for j in range(5):box(door,f'sink_door_rib{label}{j}',[-side*(.055+j*.075),-.017,.36],[.005,.004,.60],'metal',visual=True)
box(b,'sink_splashback',[0,.30,1.11],[.94,.022,.43],'steel',mass=1)
capsule(b,'faucet_stem',[-.24,.22,.9],[-.24,.22,1.25],.015,'chrome')
capsule(b,'faucet_arch',[-.24,.22,1.25],[-.24,.08,1.30],.015,'chrome')
capsule(b,'faucet_spout',[-.24,.08,1.30],[-.24,-.02,1.25],.015,'chrome')
b=body('equipment_rack',.81,-3.94,-5,size=[.61,.40,.91],source='Recorded frames 19–22: three shallow black trays, blue uprights, end handles and four casters.')
for i,(xx,yy) in enumerate([(x,y) for x in [-.265,.265] for y in [-.16,.16]]):
    caster(b,f'rack_wheel{i}',xx,yy,.04)
    cyl(b,f'rack_post{i}',[xx,yy,.48],.013,.78,'blue',.5)
for i,zz in enumerate([.17,.465,.765]):
    rounded(b,f'rack_tray{i}',[0,0,zz],[.61,.40,.026],.012,'charcoal',mass=1.3)
    for side in [-1,1]:box(b,f'rack_lip{i}_{side}',[0,side*.19,zz+.025],[.61,.012,.025],'charcoal',mass=.1)
for xx in [-.265,.265]:capsule(b,f'rack_handle{xx}',[xx,-.16,.875],[xx,.16,.875],.018,'charcoal')
b=body('kitchen_cart',.03,-3.91,size=[.72,.44,.85],source='Recorded frames 19–22: white wheeled stove stand, two lower shelves, front controls and four burners.')
for i,(xx,yy) in enumerate([(x,y) for x in [-.32,.32] for y in [-.18,.18]]):
    caster(b,f'kitchen_wheel{i}',xx,yy,.035)
    box(b,f'kitchen_post{i}',[xx,yy,.425],[.025,.025,.71],'white',mass=.6)
for i,zz in enumerate([.13,.44]):box(b,f'kitchen_shelf{i}',[0,0,zz],[.69,.41,.025],'white',mass=1.6)
box(b,'stovetop',[0,0,.775],[.72,.44,.08],'steel',mass=3)
box(b,'stove_controls',[0,.227,.774],[.70,.02,.085],'white',mass=.4)
for i,xx in enumerate(np.linspace(-.26,.26,5)):
    g=cyl(b,f'stove_knob{i}',[xx,.247,.774],.022,.019,'charcoal',.03);g.set('euler','90 0 0')
for i,(xx,yy) in enumerate([(x,y) for x in [-.17,.17] for y in [-.12,.12]]):
    cyl(b,f'burner{i}',[xx,yy,.822],.071,.009,'charcoal',.06)
    for side in [-1,1]:
        capsule(b,f'grate_x{i}_{side}',[xx-.09,yy+side*.085,.83],[xx+.09,yy+side*.085,.83],.004,'charcoal',.015)
        capsule(b,f'grate_y{i}_{side}',[xx+side*.085,yy-.09,.83],[xx+side*.085,yy+.09,.83],.004,'charcoal',.015)
b=table('low_bench',-.92,-3.79,1.0,.40,.45,mat='charcoal')
for i,yy in enumerate(np.linspace(-.17,.17,7)):capsule(b,f'bench_lower_slat{i}',[-.46,yy,.12],[.46,yy,.12],.007,'metal',mass=.1)
for xx in [-.46,.46]:capsule(b,f'bench_lower_end{xx}',[xx,-.17,.12],[xx,.17,.12],.009,'charcoal')
# Frames 123–126: split upper shelving above three full-width lower boards.
b=body('tall_shelf',-2.10,-3.80,0,size=[1.00,.43,1.94],source='Recorded frames 123–126: dark wood, slim black frame, narrow left upper bay, wide right upper bay; raw depth surface samples.')
inventory[-1].update(lower_shelf_depth_m=.43,upper_shelf_depth_m=.23,upper_shelves_back_aligned=True)
for j,x in enumerate([-.49,.16,.49]):
    box(b,f'tall_shelf_rear_upright{j}',[x,-.205,.975],[.018,.018,1.94],'charcoal',mass=.65)
    box(b,f'tall_shelf_lower_front_upright{j}',[x,.205,.49],[.018,.018,.97],'charcoal',mass=.35)
    box(b,f'tall_shelf_upper_front_upright{j}',[x,.005,1.455],[.018,.018,.98],'charcoal',mass=.35)
for j,z in enumerate([.06,.47,.97]):box(b,f'tall_shelf_full_board{j}',[0,0,z],[1.00,.43,.018],'shelf_wood',mass=2)
for j,z in enumerate([1.25,1.57]):box(b,f'tall_shelf_narrow_board{j}',[.325,-.10,z],[.31,.23,.018],'shelf_wood',mass=.35)
box(b,'tall_shelf_wide_upper_board',[-.165,-.10,1.57],[.64,.23,.018],'shelf_wood',mass=.7)
for j,x in enumerate([-.49,.16,.49]):
    for k,z in enumerate([.06,.47,.97,1.25,1.57,1.91]):
        shallow=z>1.;box(b,f'tall_shelf_side_rung{j}_{k}',[x,-.10 if shallow else 0,z],[.018,.23 if shallow else .43,.016],'charcoal',mass=.08)
for j,z in enumerate([.06,.47,.97,1.49,1.83]):box(b,f'tall_shelf_rear_rail{j}',[0,-.205,z],[1.0,.018,.016],'charcoal',mass=.15)
for j,ends in enumerate([[-.47,-.212,.08,.47,-.212,.92],[.47,-.212,.08,-.47,-.212,.92]]):capsule(b,f'tall_shelf_rear_brace{j}',ends[:3],ends[3:],.005,'charcoal',mass=.08)
# The integral socket/backing panel faces the user from the wall side of the bay.
box(b,'tall_shelf_socket_support',[.325,-.194,1.15],[.31,.018,.16],'shelf_wood',mass=.2)
box(b,'tall_shelf_socket_panel',[.325,-.180,1.15],[.23,.008,.065],'charcoal',visual=True)

# TV on its own mobile stand.
b=body('mobile_display',-4.0,-.95,103,size=[1.25,.68,1.9])
for side in [-1,1]:
    box(b,f'tv_foot{side}',[side*.45,0,.09],[.075,.68,.065],'steel',mass=6)
    for j,y in enumerate([-.28,.28]):caster(b,f'tv_caster{side}_{j}',side*.45,y,.04)
    cyl(b,f'tv_column{side}',[side*.11,.10,1.01],.025,1.82,'chrome',4)
rounded(b,'tv_crossbar',[0,-.27,.10],[.96,.16,.035],.016,'steel',mass=2)
rounded(b,'tv_center_foot',[0,.02,.10],[.24,.68,.035],.016,'steel',mass=1)
box(b,'tv_frame',[0,0,1.515],[1.19,.055,.73],'charcoal',mass=7)
box(b,'tv_screen',[0,-.032,1.515],[1.13,.008,.67],'screen_light',visual=True)
# Large screen appearance only; no source screenshots or legible interface text.
box(b,'tv_window',[.035,-.038,1.515],[1.04,.003,.62],'screen_light',visual=True)
box(b,'tv_sidebar',[-.52,-.041,1.515],[.045,.003,.64],'display',visual=True)
for j in range(12):box(b,f'tv_text_line{j}',[.12,-.041,1.76-j*.04],[.67 if j%3 else .43,.002,.002],'sofa',visual=True)
rounded(b,'tv_tray',[0,-.11,.83],[.53,.34,.025],.012,'steel',mass=1)
b=body('display_table',-4.55,-2.12,90,size=[1.35,.72,1.15])
box(b,'display_table_top',[0,0,.735],[1.35,.72,.035],'oak',mass=9)
for side in [-1,1]:
    box(b,f'display_table_column{side}',[side*.55,0,.365],[.065,.065,.70],'metal',mass=2)
    box(b,f'display_table_foot{side}',[side*.55,0,.025],[.085,.67,.05],'chrome',mass=2)
rounded(b,'desk_privacy_panel',[0,-.30,.955],[1.20,.018,.40],.008,'frosted',mass=1)
for side in [-1,1]:box(b,f'privacy_clip{side}',[side*.52,-.30,.77],[.055,.03,.075],'chrome',visual=True)

b=body('display_chair',-3.84,-.30,-65,size=[.46,.48,.82],source='Recorded frame 106: black side chair facing the display; seat/back edges guide yaw and placement.')
rounded(b,'display_chair_seat',[0,0,.455],[.43,.42,.028],.012,'sofa',mass=2)
box(b,'display_chair_back',[0,.19,.75],[.43,.022,.14],'charcoal',mass=.6)
for side in [-1,1]:
    capsule(b,f'display_chair_rear{side}',[side*.20,.22,.014],[side*.20,.19,.815],.011,'charcoal',mass=.5)
    capsule(b,f'display_chair_front{side}',[side*.20,-.20,.014],[side*.20,-.17,.46],.011,'charcoal',mass=.4)
b=body('bedside_cart',4.00,-2.38,8,size=[.47,.43,.77],source='Recorded frames 136, 169, 172: cream fabric laundry cart, gray cuff, tubular frame and four casters.')
for i,(xx,yy) in enumerate([(x,y) for x in [-.205,.205] for y in [-.185,.185]]):
    caster(b,f'laundry_wheel{i}',xx,yy,.027)
    capsule(b,f'laundry_post{i}',[xx,yy,.065],[xx,yy,.72],.011,'chrome',mass=.4)
box(b,'laundry_bottom',[0,0,.115],[.43,.39,.02],'cream_canvas',mass=1)
for xx in [-.215,.215]:box(b,f'laundry_side{xx}',[xx,0,.407],[.014,.40,.56],'cream_canvas',mass=.4)
for yy in [-.195,.195]:box(b,f'laundry_face{yy}',[0,yy,.407],[.43,.014,.56],'cream_canvas',mass=.4)
for xx in [-.22,.22]:rounded(b,f'laundry_rim{xx}',[xx,0,.708],[.025,.43,.065],.012,'sofa',mass=.15)
for yy in [-.20,.20]:rounded(b,f'laundry_end{yy}',[0,yy,.708],[.47,.025,.065],.012,'sofa',mass=.15)
for yy in [-.20,.20]:capsule(b,f'laundry_base{yy}',[-.21,yy,.075],[.21,yy,.075],.012,'chrome')
# Platform trolley: handle at the end toward the bed (left in frames 136/172).
b=body('utility_cart',4.32,-3.35,104,size=[.82,.64,.86],source='Recorded frames 136, 169, 172: low metal platform, four casters, and handle at the end nearest the laundry cart. RGB-D edge directions guide yaw.')
deck=rounded(b,'utility_deck',[0,0,.185],[.82,.64,.035],.016,'steel',mass=8)
deck.set('contype','0');deck.set('conaffinity','0')
box(b,'utility_deck_contact',[0,0,.185],[.78,.60,.035],'white',mass=0).set('rgba','0 0 0 0')
rounded(b,'utility_bumper',[0,0,.165],[.84,.66,.027],.012,'charcoal',mass=.6)
for i,(xx,yy) in enumerate([(x,y) for x in [-.32,.32] for y in [-.24,.24]]):caster(b,f'utility_wheel{i}',xx,yy,.075,mat='sofa')
for side in [-1,1]:
    capsule(b,f'utility_handle_post{side}',[.32,side*.26,.20],[.32,side*.26,.73],.016,'chrome',mass=.5)
    capsule(b,f'utility_handle_curve{side}',[.32,side*.26,.73],[.32,side*.205,.82],.017,'chrome',mass=.2)
for zz in [.36,.56]:capsule(b,f'utility_handle_rail{zz}',[.32,-.26,zz],[.32,.26,zz],.012,'chrome')
capsule(b,'utility_handle_grip',[.32,-.205,.82],[.32,.205,.82],.020,'charcoal',mass=.3)
# One existing large box, resting on the deck near the handle; no contents added.
cart_angle=math.radians(104);offset=np.array([[math.cos(cart_angle),-math.sin(cart_angle)],[math.sin(cart_angle),math.cos(cart_angle)]])@np.array([.06,0])
b=body('large_box',4.32+offset[0],-3.35+offset[1],104,size=[.47,.49,.48],source='Recorded frames 136, 169, 172: open cardboard box on the platform toward its handle, not above a second shelf.')
base=.2025
box(b,'large_box_bottom',[0,0,base+.005],[.47,.49,.01],'cardboard',mass=.4)
for xx in [-.23,.23]:box(b,f'large_box_side{xx}',[xx,0,base+.155],[.01,.49,.30],'cardboard',mass=.2)
for yy in [-.24,.24]:box(b,f'large_box_end{yy}',[0,yy,base+.155],[.45,.01,.30],'cardboard',mass=.2)
for side in [-1,1]:
    flap=box(b,f'large_box_flap{side}',[0,side*.275,base+.375],[.45,.007,.18],'cardboard',visual=True);flap.set('euler',f'{-side*35} 0 0')
box(b,'large_box_tape',[0,.246,base+.08],[.035,.002,.14],'tape',visual=True)
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
SCREEN_COLUMNS = 6
SCREEN_ROWS = 15
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
                         panel_width_m=width,panel_height_m=SCREEN_HEIGHT,grid_columns=SCREEN_COLUMNS,grid_rows=SCREEN_ROWS,
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
        for face,yy in [('front',-.017),('back',.017)]:
            for j,z in enumerate(np.linspace(.10,1.8,SCREEN_ROWS+1)):
                box(cur,f'{n}_rail{i}_{face}_{j}',[width/2,yy,z],[width,.008,.007],'screenframe',visual=True)
            for j,xx in enumerate(np.linspace(.0125,width-.0125,SCREEN_COLUMNS+1)[1:-1]):
                box(cur,f'{n}_grid{i}_{face}_{j}',[xx,yy,.95],[.006,.008,1.70],'screenframe',visual=True)
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

# Folded wall tables / mounts in frame 126. Positions are ray/plane measurements,
# and the folded support rails sit proud of each dark rectangular backplate.
for i,(x,w) in enumerate([(.3764,.79),(-1.1825,.77)],1):
    p=north_point(x)+np.array([0,.0215])
    b=body(f'wall_table_mount{i}',*p,yaw,free=False,size=[w,.10,.52],source='Recorded frame 126: four mount corners projected onto the fitted north wall plus 3.5 cm face offset; cross-checked in frames 123–125.')
    box(b,f'wall_mount{i}_backplate',[0,0,1.535],[w,.027,.52],'charcoal',mass=3)
    # A folded rectangular bracket with open vertical slots and side handles.
    for side in [-1,1]:
        box(b,f'wall_mount{i}_bracket_side{side}',[side*.235,.04,1.53],[.055,.035,.37],'metal',mass=.3)
        box(b,f'wall_mount{i}_bracket_inner{side}',[side*.08,.04,1.53],[.035,.035,.34],'metal',mass=.15)
        capsule(b,f'wall_mount{i}_handle_top{side}',[side*.26,.045,1.62],[side*.33,.07,1.60],.014,'charcoal')
        capsule(b,f'wall_mount{i}_handle_side{side}',[side*.33,.07,1.60],[side*.33,.07,1.46],.014,'charcoal')
        capsule(b,f'wall_mount{i}_handle_bottom{side}',[side*.33,.07,1.46],[side*.26,.045,1.44],.014,'charcoal')
    for side in [-1,1]:box(b,f'wall_mount{i}_crosspiece{side}',[0,.04,1.53+side*.175],[.50,.035,.045],'metal',mass=.3)
    for side in [-1,1]:box(b,f'wall_mount{i}_hinge{side}',[side*.17,.026,1.785],[.11,.032,.04],'chrome',visual=True)
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
