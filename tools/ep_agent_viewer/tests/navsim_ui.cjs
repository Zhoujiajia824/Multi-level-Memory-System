// Saved-artifact browser regression. No driver, scorer or renderer runs.
const {spawn}=require('node:child_process');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
  const root=process.env.EPVIZ_NAVSIM_REAL_ROOT;
  if(!root)throw new Error('Set EPVIZ_NAVSIM_REAL_ROOT to an existing saved NAVSIM run');
  const script="import sys; from epviz.server import ViewerServer; s=ViewerServer(('127.0.0.1',0),sys.argv[1]); print(s.server_port,flush=True); s.serve_forever()";
  const server=spawn(process.env.EPVIZ_PYTHON||'python',['-u','-c',script,root],{cwd:path.resolve(__dirname,'..'),windowsHide:true});
  let browser;
  try{
    const port=await new Promise((resolve,reject)=>{server.stdout.once('data',v=>resolve(Number(String(v).trim())));server.once('error',reject);server.once('exit',code=>reject(new Error('server exited '+code)));});
    const base='http://127.0.0.1:'+port;
    browser=await chromium.launch({executablePath:process.env.EPVIZ_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-swiftshader']});
    const page=await browser.newPage({viewport:{width:1720,height:1160}}),errors=[],heavyRequests=[],heavyResponses=[];
    const heavyKind=url=>{const p=new URL(url).pathname;return /^\/scene(?:-geometry)?\.js$/.test(p)?'scene_modules':p==='/assets/ego-tesla-model3.glb'?'tesla_model':p==='/vendor/plotly.min.js'?'plotly':/^\/vendor\/(three[^/]*\.js|OrbitControls\.js|utils\/|loaders\/|draco\/|environments\/)/.test(p)?'three':null;};
    page.on('request',request=>{const kind=heavyKind(request.url());if(kind)heavyRequests.push({kind,path:new URL(request.url()).pathname});});
    page.on('response',response=>{const kind=heavyKind(response.url());if(kind)heavyResponses.push({kind,path:new URL(response.url()).pathname,bytes:Number(response.headers()['content-length']||0)});});
    const cdp=await page.context().newCDPSession(page);await cdp.send('Network.enable');await cdp.send('Network.setCacheDisabled',{cacheDisabled:true});
    page.on('pageerror',e=>errors.push(e.message));
    assert.equal((await page.goto(base+'/',{waitUntil:'domcontentloaded'})).status(),200);
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 1 / 41'&&document.querySelector('#video').readyState>=1,null,{timeout:90000});
    assert.equal(await page.locator('#video-view').isVisible(),true);
    assert.equal(await page.locator('#scene-canvas').isVisible(),false);
    assert.equal(await page.locator('#navsim-panel').getAttribute('open'),null);
    assert.equal(await page.locator('#camera-grid .camera-card').count(),8);
    assert.match(await page.locator('[data-view="scene"]').textContent(),/非官方/);
    assert.deepEqual(heavyRequests,[],'default NAVSIM video must not request Three, GLTF, Draco or Plotly');
    assert.equal(await page.locator('#scene-canvas').getAttribute('data-renderer'),null);
    assert.equal(await page.locator('#scene-canvas').getAttribute('data-ego-model'),null);
    assert.equal(await page.evaluate(()=>typeof window.Plotly),'undefined');
    const defaultResources={requests:heavyRequests.length,bytes:heavyResponses.reduce((n,r)=>n+r.bytes,0),renderer:null,ego_model:null,plotly:false};
    const items=(await(await page.request.get(base+'/api/episodes')).json()).episodes,realId=items[0].id;
    const manifest=await(await page.request.get(base+'/api/episodes/'+realId)).json();
    assert.equal(await page.locator('#video').getAttribute('data-official-visualization'),String(manifest.video.official_visualization));
    if(!manifest.video.official_visualization)assert.match(await page.locator('#video-status').textContent(),/旧版自定义BEV视频（非官方绘图）/);
    // Adjacent manual samples must select the actual 0.1s frame, not retain
    // t=0 under the old generic 0.15s playback resynchronization threshold.
    await page.locator('#next-frame').click();
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 2 / 41'&&!document.querySelector('#video').seeking&&Math.abs(document.querySelector('#video').currentTime-.1)<.005);
    assert.equal(await page.locator('#video').evaluate(video=>video.paused),true);
    await page.locator('#previous-frame').click();
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 1 / 41'&&!document.querySelector('#video').seeking&&Math.abs(document.querySelector('#video').currentTime)<.005);
    await page.locator('#timeline').fill('1');await page.locator('#timeline').dispatchEvent('input');
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 2 / 41'&&!document.querySelector('#video').seeking&&Math.abs(document.querySelector('#video').currentTime-.1)<.005);
    await page.locator('#video').evaluate(async video=>{video._qaSeeks=0;video.addEventListener('seeking',()=>video._qaSeeks++);await video.play();});
    await page.waitForFunction(()=>document.querySelector('#video').currentTime>=.7);
    await page.locator('#video').evaluate(video=>video.pause());
    assert.equal(await page.locator('#video').evaluate(video=>video._qaSeeks),0,'native video playback must not be pulled back by delayed frame responses');
    await page.locator('#video').evaluate(video=>{video.currentTime=1.;});
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 11 / 41');
    assert.equal(await page.locator('#camera-grid .camera-card').count(),0);
    await page.locator('#timeline').fill('0');await page.locator('#timeline').dispatchEvent('input');
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 1 / 41');
    await page.locator('#navsim-panel summary').click();
    await page.waitForFunction(()=>document.querySelector('#navsim-plot')?.data?.length===4);
    assert.deepEqual(await page.locator('#navsim-plot').evaluate(plot=>plot.data.slice(0,3).map(trace=>trace.x.length)),[41,8,41]);
    assert.equal(heavyRequests.filter(r=>r.kind==='plotly').length,1);
    assert.equal(heavyRequests.filter(r=>r.kind!=='plotly').length,0,'opening curve diagnostics must not initialize 3D');
    // API fixtures reuse one saved video under distinct run URLs. This tests UI
    // state isolation; renderer-marker authenticity is checked in Python tests.
    const source=await(await page.request.get(base+'/api/episodes/'+realId+'/frames/0')).json();
    const ids=['qa-official-run','qa-legacy-run','qa-regular-platform'],clone=v=>JSON.parse(JSON.stringify(v));
    const fixtures=ids.map((id,index)=>{
      const m=clone(manifest);m.id=id;m.episode_id=id;
      if(index<2){m.video.url=manifest.video.url+'&viewer_qa_run='+id;m.video.official_visualization=index===0;m.video.renderer_version=index===0?'navsim-sdk-plots-1.0':null;m.video.visualization_label=index===0?'NAVSIM SDK官方绘图 · 评分执行视频':'旧版自定义BEV视频（非官方绘图）';m.video.visualization_note=index===0?null:'缺少已认证的官方renderer标记';}
      else{delete m.dataset;delete m.navsim;m.platform='highway';m.frames=[m.frames[0]];m.times=[0];m.frame_count=1;m.video={available:false};}
      return m;
    });
    let startWithRegular=false;
    await page.route('**/api/episodes**',async route=>{
      const url=new URL(route.request().url()),parts=url.pathname.split('/'),idx=ids.indexOf(parts[3]);
      if(url.pathname==='/api/episodes')return route.fulfill({json:{episodes:startWithRegular?[fixtures[2],fixtures[0],fixtures[1]]:fixtures,read_only:true}});
      if(idx<0)return route.continue();
      if(parts.length===4)return route.fulfill({json:fixtures[idx]});
      if(parts[4]==='frames'){
        const frame=idx===2?clone(source):await(await page.request.get(base+'/api/episodes/'+realId+'/frames/'+parts[5])).json();frame.episode_id=ids[idx];
        if(idx===2){delete frame.navsim;delete frame.dataset;frame.platform='highway';}
        return route.fulfill({json:frame});
      }
      return route.continue();
    });
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.querySelector('#video').dataset.episode==='qa-official-run'&&document.querySelector('#timeline-frame').textContent==='FRAME 1 / 41');
    assert.equal(await page.locator('#video-view').isVisible(),true);
    assert.match(await page.locator('#video-status').textContent(),/NAVSIM SDK官方绘图/);
    assert.equal(await page.locator('[data-view="video"]').textContent(),'官方执行视频');
    const first=await page.locator('#video').getAttribute('src');
    // If the user returns to video while scene modules are downloading, do not
    // construct a late renderer or start the large GLTF request.
    let releaseScene,markSceneRequested;
    const sceneGate=new Promise(resolve=>{releaseScene=resolve;}),sceneRequested=new Promise(resolve=>{markSceneRequested=resolve;});
    await page.route('**/scene.js',async route=>{markSceneRequested();await sceneGate;await route.continue();});
    await page.locator('[data-view="scene"]').click();await sceneRequested;
    await page.locator('[data-view="video"]').click();releaseScene();
    await page.waitForFunction(()=>document.querySelector('#scene-canvas').dataset.sceneLoad==='deferred');
    assert.equal(await page.locator('#scene-canvas').getAttribute('data-renderer'),null);
    assert.equal(heavyRequests.filter(r=>r.kind==='tesla_model').length,0);
    await page.unroute('**/scene.js');
    await page.locator('[data-view="scene"]').click();
    await page.waitForFunction(()=>document.querySelector('#scene-canvas').dataset.egoModel==='loaded'&&document.querySelector('#scene-canvas').dataset.frameGeometry==='available',null,{timeout:90000});
    assert.equal(await page.locator('#scene-canvas').isVisible(),true);
    assert.match(await page.locator('#scene-description').textContent(),/三维诊断（非官方绘图）/);
    assert.ok(heavyRequests.some(r=>r.kind==='scene_modules'));
    assert.ok(heavyRequests.some(r=>r.kind==='three'));
    assert.equal(heavyRequests.filter(r=>r.kind==='tesla_model').length,1);
    const clickedResources=Object.fromEntries(['scene_modules','three','tesla_model','plotly'].map(kind=>[kind,{requests:heavyRequests.filter(r=>r.kind===kind).length,bytes:heavyResponses.filter(r=>r.kind===kind).reduce((n,r)=>n+r.bytes,0)}]));
    await page.locator('#episode-select').selectOption('qa-legacy-run');
    await page.waitForFunction(()=>document.querySelector('#video').dataset.episode==='qa-legacy-run'&&document.querySelector('#timeline-frame').textContent==='FRAME 1 / 41');
    assert.equal(await page.locator('#video-view').isVisible(),true);
    assert.equal(await page.locator('#scene-canvas').isVisible(),false);
    assert.equal(await page.locator('#navsim-panel').getAttribute('open'),null);
    assert.notEqual(await page.locator('#video').getAttribute('src'),first);
    assert.match(await page.locator('#video-status').textContent(),/旧版自定义BEV视频（非官方绘图）/);
    await page.locator('#episode-select').selectOption('qa-regular-platform');
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 1 / 1');
    assert.equal(await page.locator('#scene-canvas').isVisible(),true);
    assert.equal(await page.locator('[data-view="scene"]').textContent(),'空间场景');
    assert.equal(await page.locator('#navsim-panel').isVisible(),false);
    startWithRegular=true;await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.querySelector('#timeline-frame').textContent==='FRAME 1 / 1'&&document.querySelector('#scene-canvas').dataset.egoModel==='loaded'&&!!window.Plotly,null,{timeout:90000});
    assert.equal(await page.locator('#scene-canvas').isVisible(),true,'ordinary platform must still initialize its scene without a diagnostic click');
    await page.unrouteAll({behavior:'wait'});
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({saved_samples:items.length,default_view:'video',renderer:manifest.video.renderer_version||'unversioned',default_diagnostic_resources:defaultResources,clicked_diagnostic_resources:clickedResources,late_scene_construction_prevented:true,manual_sample_seek_exact:true,native_playback_resync_seeks:0,native_video_seek:true,diagnostics_collapsed:true,same_token_runs_isolated:true,other_platform_scene_preserved:true,pageErrors:errors}));
  }finally{await browser?.close();server.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
