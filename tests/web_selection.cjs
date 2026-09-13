const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1600,height:1050},reducedMotion:'reduce'});
 await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8765/');await page.waitForFunction(()=>window.halab?.state().loaded===0);
 const project=point=>page.evaluate(async point=>{
  const THREE=await import('./vendor/three.module.js'),rect=document.querySelector('#stage canvas').getBoundingClientRect(),view=window.halab.viewState();
  const aspect=rect.width/rect.height,fov=2*Math.atan(Math.tan(24*Math.PI/180)*Math.max(1,1.35/aspect))*180/Math.PI;
  const camera=new THREE.PerspectiveCamera(fov,aspect,.025,60);camera.up.set(0,0,1);camera.position.fromArray(view.position);camera.lookAt(new THREE.Vector3(...view.target));camera.updateMatrixWorld();
  const p=new THREE.Vector3(...point).project(camera);return {x:rect.left+(p.x+1)*rect.width/2,y:rect.top+(1-p.y)*rect.height/2};
 },point);
 await page.locator('#ceiling').focus();await page.keyboard.press('Space');assert(await page.locator('#ceiling').isChecked());await page.keyboard.press('Space');assert(!(await page.locator('#ceiling').isChecked()));
 assert(await page.locator('aside input[type=checkbox]').evaluateAll(inputs=>inputs.every(input=>{const style=getComputedStyle(input);return style.appearance==='none'&&parseFloat(style.width)>parseFloat(style.height);})), 'Sidebar checkboxes should render as switches');
 await page.locator('#path').uncheck();
 for(const [name,point] of [['floor',[0,0,0]],['wall',[.0734523,-4.24053,3.01]],['background',null]]){
  await page.locator('.asset[data-name="white_cabinet"]').click();assert.equal(await page.locator('#articulations input').count(),2);
  await page.locator('#top').click();await page.locator('#wall-mode').selectOption('full');
  const view=await page.evaluate(()=>window.halab.viewState()),p=point?await project(point):{x:12,y:70};
  await page.mouse.click(p.x,p.y);
  assert.equal(await page.evaluate(()=>window.halab.state().selected),null,name+' should deselect');
  assert.equal(await page.locator('.asset.selected').count(),0);assert.equal(await page.locator('#articulations input').count(),0);
  const after=await page.evaluate(()=>window.halab.viewState());
  for(const key of ['position','target'])assert(Math.hypot(...after[key].map((v,i)=>v-view[key][i]))<1e-6,name+' should keep camera framing: '+JSON.stringify({before:view,after}));
  assert.equal(after.focusing,false);
 }
 await page.locator('#orbit').click();
 const cabinetPoint=await project([1.55,1.996,.854]);await page.mouse.click(cabinetPoint.x,cabinetPoint.y);
 assert.equal(await page.evaluate(()=>window.halab.state().selected),'white_cabinet','Cutaway walls should not intercept a visible cabinet click');
 const view=await page.evaluate(()=>window.halab.viewState());assert(Math.hypot(...view.position.map((v,i)=>v-view.target[i]))>4,'Cabinet focus should leave surrounding context');
 await page.locator('.asset[data-name="red_tool_bag"]').click();const small=await page.evaluate(()=>window.halab.viewState());assert(Math.hypot(...small.position.map((v,i)=>v-small.target[i]))>=3.49,'Small assets should not cause extreme close-ups');
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.locator('#open-all').check();assert((await page.evaluate(()=>Object.values(window.halab.articulationState()))).every(Boolean));await page.locator('#open-all').uncheck();
 console.log('PASS: sidebar switches and keyboard access; floor/wall/background deselection; cutaway picking; gentler asset focus; mobile switches');
 await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
