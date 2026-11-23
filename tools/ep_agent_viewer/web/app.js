import {CameraGallery} from './cameras.js';
import {PotentialView} from './potential.js';

const $=id=>document.getElementById(id);
const fmt=(value,digits=2)=>Number.isFinite(Number(value))&&value!==null?Number(value).toFixed(digits):'—';
const json=value=>typeof value==='string'?value:JSON.stringify(value,null,2);
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
const clock=t=>{if(!finite(t))return'—';return `${String(Math.floor(Math.max(0,t)/60)).padStart(2,'0')}:${(Math.max(0,t)%60).toFixed(2).padStart(5,'0')}`;};
const initialParams=new URLSearchParams(location.search);
const state={episodes:[],runs:[],run:initialParams.get('run'),stage:['stage_one','stage_two','unknown'].includes(initialParams.get('stage'))?initialParams.get('stage'):'all',runSummary:null,runController:null,runRequest:0,listingRequest:0,episode:initialParams.get('episode'),manifest:null,frame:null,index:0,requested:-1,playing:false,rate:1,follow:initialParams.get('follow')==='1',epoch:0,manifestEpoch:0,manifestRequest:0,frameController:null,potentialController:null,view:'scene',isNavsim:false,nonNavsimView:'scene',pendingVideoSeek:null,playTime:0,lastWall:0,fetching:false};
const sceneLayers={roads:true,sidewalk:true,junction:true,crosswalk:true,topology:false,regions:true,objects:true,grid_matches:true,sources:true,attention:true,trajectory:true,corridor:true,goal:true};
let scene=null,scenePromise=null,plotlyPromise=null,navsimPlotLoading=false;
const assetUrl=path=>!path?'':typeof path==='object'?(path.url||assetUrl(path.path)):path.startsWith('/api/')||path.startsWith('/assets/')?path:`/api/asset?episode=${encodeURIComponent(state.episode||'')}&path=${encodeURIComponent(path)}`;
const gallery=new CameraGallery($('camera-grid'),$('camera-modal'),{assetUrl,layers:sceneLayers,onOpen:()=>{pause();follow(false);}});
let toastTimer,potentialData=null,potentialSourceFrame=null,potentialGeneration=0,potentialBusy=false;
const potentialView=new PotentialView($('potential-plot'));
const potentialCache=new Map();

function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4200);}
function connection(message,status='good'){$('connection').className=`badge ${status}`;$('connection').replaceChildren(el('i'),document.createTextNode(message));}
async function request(url,signal,options={}){const response=await fetch(url,{signal,cache:'no-store',...options});if(!response.ok){let detail;try{detail=(await response.json()).error;}catch{}const error=new Error(detail||`HTTP ${response.status}`);error.status=response.status;throw error;}return response.json();}
function episodeName(item){return item.scenario||item.summary?.scenario||item.episode_id||item.id;}
function modeName(mode){return ({pure_base:'纯基础规划',base:'纯基础规划',base_model:'基础规划 + 模型',model:'基础规划 + 模型',retrieval:'模型 + 经验记忆',full_pipeline:'模型辅助规划',fast:'基础规划 + 模型',emergency_unverified:'紧急制动'})[mode]||mode||'执行模式未归档';}
function metricValue(value){if(value===true)return'是';if(value===false)return'否';if(value===undefined||value===null)return'未归档';if(typeof value==='number')return Number.isInteger(value)?String(value):fmt(value,3);return typeof value==='string'?value:json(value);}
function emptyPanel(container,title,message,icon='⌁'){container.replaceChildren();const block=el('div','empty-state small');block.append(el('span','empty-icon',icon),el('strong',null,title),el('p',null,message));container.append(block);}
function metricList(container,title,values){container.append(el('div','metric-title',title));const dl=el('dl','metric-list');for(const [label,value]of values){const row=el('div');row.append(el('dt',null,label),el('dd',typeof value==='boolean'?(value?'good':'bad'):null,metricValue(value)));dl.append(row);}container.append(dl);}
function rawDetails(container,title,value){if(value==null)return;const d=el('details');d.append(el('summary',null,title),el('pre',null,json(value)));container.append(d);}
const stageName=stage=>({stage_one:'第一阶段',stage_two:'第二阶段',unknown:'阶段未归档'})[stage]||'阶段未归档';
const scoreColumns=[['NC','无责任碰撞'],['DAC','可行驶区域遵守'],['DDC','行驶方向遵守'],['TLC','交通灯遵守'],['EP','自车行驶进度'],['TTC','碰撞时间安全指标'],['LK','车道保持'],['HC','历史衔接舒适性'],['EC','相邻规划扩展舒适性']];
const otherRun='__other_experiments__';
const selectedRun=()=>state.runs.find(run=>run.id===state.run);
function rememberSelection(){
  const url=new URL(location.href);
  for(const [key,value]of [['run',state.run],['stage',state.stage==='all'?null:state.stage],['episode',state.episode]]){if(value)url.searchParams.set(key,value);else url.searchParams.delete(key);}
  history.replaceState(null,'',url);
}
function visibleEpisodes(){return state.episodes.filter(item=>{const inRun=!state.runs.length||(state.run===otherRun?!item.run_id:item.run_id===state.run);return inRun&&(state.stage==='all'||item.stage===state.stage);});}
function renderRunResults(){
  const run=selectedRun(),panel=$('run-results');panel.hidden=!run;if(!run)return;
  const summary=state.runSummary,ready=summary?.scores_status==='complete',scores=ready?summary.scores||{}:{},failed=(summary||run).status==='failed';
  const status=$('run-results-status');status.className=`badge ${ready?'good':failed?'bad':'subtle'}`;status.textContent=ready?'已完成':failed?'测评失败':summary?.scores_status==='unavailable'?'汇总不可用':summary?.status==='running'?'测评进行中':summary?'汇总未生成':'读取中';
  $('run-results-name').textContent=`${run.condition_label||run.label||run.name} · ${run.name||run.id}`;
  $('run-results-name').title=run.path||'';
  const detail=summary||run,split=detail.split||run.split||'';
  $('run-results-progress').textContent=[split,`${detail.completed_samples??0}${detail.expected_samples!=null?' / '+detail.expected_samples:''} 个场景已评分`].filter(Boolean).join(' · ');
  $('run-score-total').textContent=finite(scores.combined)?fmt(scores.combined*100):'—';
  const table=$('run-score-table'),head=el('tr');head.append(el('th',null,'阶段'));head.firstChild.scope='col';
  for(const [key,label]of [...scoreColumns,['EPDMS','阶段综合分']]){const th=el('th',null,key);th.scope='col';th.title=label;head.append(th);}
  table.tHead.replaceChildren(head);const body=table.tBodies[0];body.replaceChildren();
  for(const stage of ['stage_one','stage_two']){
    const row=el('tr',state.stage===stage?'selected':'');row.dataset.stage=stage;
    const label=el('th',null,stageName(stage));label.scope='row';row.append(label);
    for(const [key]of scoreColumns){const value=ready?summary.metrics?.find(metric=>metric.key===key)?.[stage]:null;const cell=el('td',null,finite(value)?fmt(value*100):'—');cell.dataset.metric=key;row.append(cell);}
    const score=el('td','stage-score',finite(scores[stage])?fmt(scores[stage]*100):'—');score.dataset.metric='EPDMS';row.append(score);body.append(row);
  }
  const phases={data_inspection:'数据与导航预检',cache_verification:'评分缓存核验',official_scoring:'官方评分',official_result_validation:'官方结果核验'};
  let note=ready?'官方 CSV 汇总 · 0–100 分，越高越好':failed?`${phases[detail.run_phase]||detail.run_phase||'测评'}失败；已有场景仍可查看。`:
    summary?.scores_status==='unavailable'?'官方评分汇总不可用，请检查本轮评分文件。':'官方汇总尚未生成；可继续查看已完成场景。';
  if(failed&&summary?.error?.message)note+=` ${String(summary.error.message).slice(0,300)}`;
  const condition=summary?.condition_audit;
  if(ready&&condition?.model_requested>0)note+=` · 模型实际应用 ${condition.model_semantics_applied??condition.valid_model_answers??0}/${condition.model_requested}，回退 ${condition.base_fallback_samples??0}`;
  $('run-results-note').textContent=note;
  $('run-results-note').title=summary?.official_csv||summary?.warnings?.join('\n')||'';
}
async function loadRunResults(){
  const run=selectedRun(),requestId=++state.runRequest;state.runController?.abort();state.runController=new AbortController();state.runSummary=null;renderRunResults();if(!run)return;
  try{
    const summary=await request(`/api/runs/${encodeURIComponent(run.id)}`,state.runController.signal);
    if(requestId!==state.runRequest||state.run!==run.id)return;state.runSummary=summary;renderRunResults();
  }catch(error){if(error.name==='AbortError'||requestId!==state.runRequest)return;$('run-results-status').textContent='汇总读取失败';$('run-results-note').textContent=error.message;}
}

async function ensureScene(){
  if(!state.manifest||state.isNavsim&&state.view!=='scene')return null;
  if(scene){scene.setFrame(state.frame);return scene;}
  if(!scenePromise){
    $('scene-canvas').dataset.sceneLoad='loading';
    scenePromise=import('./scene.js').then(({SceneRenderer})=>{
      // The user may have changed samples or returned to video while modules
      // were downloading. Do not construct a late hidden renderer/GLTF load.
      if(!state.manifest||state.isNavsim&&state.view!=='scene'){$('scene-canvas').dataset.sceneLoad='deferred';return null;}
      scene=new SceneRenderer($('scene-canvas'),$('object-tooltip'));
      scene.layers=sceneLayers;scene.setFrame(state.frame);
      $('scene-canvas').dataset.sceneLoad='initialized';return scene;
    }).catch(error=>{$('scene-canvas').dataset.sceneLoad='failed';toast('三维诊断加载失败：'+error.message);return null;})
      .finally(()=>{scenePromise=null;});
  }
  return scenePromise;
}
function ensurePlotly(){
  if(window.Plotly)return Promise.resolve(window.Plotly);
  if(!plotlyPromise)plotlyPromise=new Promise((resolve,reject)=>{
    const script=document.createElement('script');script.src='/vendor/plotly.min.js';script.async=true;
    script.onload=()=>window.Plotly?resolve(window.Plotly):reject(new Error('图表资源未能初始化'));
    script.onerror=()=>{script.remove();reject(new Error('无法读取本地图表资源'));};
    document.head.append(script);
  }).catch(error=>{plotlyPromise=null;throw error;});
  return plotlyPromise;
}

async function loadEpisodes(){
  const requestId=++state.listingRequest;
  try{
    const data=await request('/api/episodes');if(requestId!==state.listingRequest)return;
    state.episodes=Array.isArray(data)?data:data.episodes||[];state.runs=Array.isArray(data.runs)?data.runs:[];
    const runSelect=$('run-select');runSelect.replaceChildren();$('run-filters').hidden=!state.runs.length;
    for(const run of state.runs){const option=new Option(`${run.condition_label||run.label||run.name} · ${run.name||run.id}`,run.id);option.title=run.path||'';runSelect.append(option);}
    const hasOther=state.episodes.some(item=>!item.run_id);if(state.runs.length&&hasOther)runSelect.append(new Option('其他实验',otherRun));
    if(!state.runs.length){state.run=null;state.stage='all';}
    else if(!state.runs.some(run=>run.id===state.run)&&!(state.run===otherRun&&hasOther)){state.run=state.episodes.find(item=>(item.id||item.episode_id)===state.episode)?.run_id||state.runs[0].id;}
    runSelect.value=state.run||'';configureStageFilter();
    const summaryRequest=loadRunResults();await applyEpisodeFilters();await summaryRequest;
  }catch(error){if(requestId!==state.listingRequest)return;connection('连接失败','bad');toast(error.message);$('episode-meta').textContent=error.message;}
}
function configureStageFilter(){
  const select=$('stage-select'),run=selectedRun();if(!run)state.stage='all';
  select.disabled=!run;select.value=state.stage;
  select.querySelector('[value="unknown"]').hidden=!(run?.stages?.unknown>0)&&state.stage!=='unknown';
}
async function applyEpisodeFilters(){
  const items=visibleEpisodes(),select=$('episode-select');select.replaceChildren();select.disabled=!items.length;
  if(!items.length){
    select.append(new Option('当前选择暂无场景',''));await selectEpisode(null);connection('目录已连接');
    $('episode-meta').textContent='当前轮次和阶段暂无已归档场景';return;
  }
  for(const item of items){const id=item.id||item.episode_id,label=item.run_id?`${stageName(item.stage)} · ${episodeName(item)}`:`${episodeName(item)} · ${String(item.episode_id||id).slice(-29)}`;const option=new Option(label,id);option.title=item.path||'';select.append(option);}
  const desired=items.some(item=>(item.id||item.episode_id)===state.episode)?state.episode:items[0].id||items[0].episode_id;
  select.value=desired;connection('产物已连接');rememberSelection();
  if(desired!==state.episode||!state.manifest)await selectEpisode(desired);else await loadManifest(false);
}
async function selectEpisode(id){
  pause();state.epoch++;state.manifestEpoch++;potentialGeneration++;potentialBusy=false;potentialSourceFrame=null;state.frameController?.abort();state.potentialController?.abort();state.episode=id;state.frame=null;state.manifest=null;state.index=0;state.requested=-1;state.pendingVideoSeek=null;state.playTime=0;state.fetching=false;potentialData=null;scene?.setFrame(null);
  $('stage-empty').hidden=false;gallery.setFrame(null);$('video').pause();$('video').removeAttribute('src');$('video').removeAttribute('poster');$('video').load();$('video').dataset.episode='';$('video-preview').hidden=true;$('video-preview').removeAttribute('src');$('repair-video').hidden=true;$('repair-video').disabled=false;$('repair-video').textContent='生成浏览器兼容副本';state.compatVideo=null;$('bev-image').removeAttribute('src');
  $('navsim-panel').hidden=true;$('navsim-panel').open=false;navsimPlotToken=null;
  $('sim-time').textContent='—';$('ego-speed').textContent='—';$('actor-count').textContent='—';$('frame-id').textContent='等待新源帧';$('plan-safety').textContent='等待读取';$('plan-safety').className='badge subtle';$('warnings').hidden=true;$('camera-count').textContent='0 VIEWS';drawPotential();
  emptyPanel($('dialogue-panel'),'读取当前实验','正在关联源帧与归档内容…');
  rememberSelection();
  if(!id){
    $('video').hidden=true;$('video-status').textContent='当前选择暂无场景';$('stage-empty').querySelector('strong').textContent='当前选择暂无场景';
    $('stage-empty').querySelector('p').textContent='请选择其他测评轮次或阶段。';$('episode-meta').textContent='暂无场景';
    $('run-mode').textContent=selectedRun()?.condition_label||'—';$('run-status').className='status-dot';
    $('timeline').max=0;$('timeline').value=0;$('timeline-current').textContent='—';$('timeline-end').textContent='—';$('timeline-frame').textContent='FRAME — / —';$('timeline-events').replaceChildren();
    emptyPanel($('dialogue-panel'),'暂无场景','请选择其他测评轮次或阶段。');emptyPanel($('metrics-panel'),'暂无场景','整轮官方汇总仍显示在上方。');emptyPanel($('memory-panel'),'暂无场景','');return;
  }
  await loadManifest(true);
}
async function clearUnavailableEpisode(){
  await selectEpisode(null);$('episode-select').value='';
  $('episode-meta').textContent='当前样本未完成或失败，旧回放已清除；刷新后重新选择';
  connection('当前样本不可用','warn');await loadRunResults();
}
async function loadManifest(initial=false){
  if(!state.episode)return;
  const episode=state.episode,epoch=state.manifestEpoch,requestId=++state.manifestRequest;
  try{
    const manifest=await request(`/api/episodes/${encodeURIComponent(episode)}`);if(episode!==state.episode||epoch!==state.manifestEpoch||requestId!==state.manifestRequest)return;
    state.manifest=manifest;const count=Number(manifest.frame_count??manifest.frames?.length??manifest.times?.length??0);
    if(!manifest.frames)manifest.frames=Array.from({length:count},(_,index)=>({index,time:manifest.times?.[index]}));
    if(!manifest.times)manifest.times=manifest.frames.map(f=>f.time);
    const summary=manifest.summary||{};
    const item=state.episodes.find(item=>(item.id||item.episode_id)===episode),run=selectedRun();
    $('episode-meta').textContent=manifest.dataset==='navsim'?`${stageName(item?.stage)} · ${count} 个采样`:`${count} 个规划帧 · ${manifest.platform||'实验回放'}`;
    $('episode-meta').title=manifest.dataset==='navsim'?`锚点 ${manifest.navsim.anchor_token}，独立4秒，不与其他样本串接`:manifest.episode_id||episode;
    const scenario=manifest.scenario||summary.scenario||'驾驶现场';
    $('scene-title').textContent=manifest.dataset==='navsim'?'NAVSIM 评测回放':({highway:'高速公路',roundabout:'环岛',intersection:'十字路口',merge:'汇入'})[scenario]||scenario;
    $('scene-title').title=scenario;
    $('run-mode').textContent=run?.condition_label||item?.condition_label||modeName(summary.execution_mode||manifest.execution_mode||manifest.configuration?.execution_mode);
    $('run-status').className=`status-dot ${run?.status==='complete'||summary.arrival?'good':''}`;
    $('data-root').textContent=manifest.path||manifest.source||manifest.relative_path||'由服务端指定的只读实验目录';
    $('timeline').max=Math.max(0,count-1);$('timeline-end').textContent=clock(manifest.times.at(-1));
    $('next-event').disabled=!(manifest.model_event_indices?.length);
    $('timeline-events').replaceChildren();for(const index of manifest.model_event_indices||[]){const mark=el('i');mark.style.left=`${100*index/Math.max(1,count-1)}%`;$('timeline-events').append(mark);}
    setupVideo(manifest.video);
    configureDatasetView(manifest,initial);
    connection(state.follow?'跟随产物 · 1Hz':'产物已连接');
    if(count){const next=state.follow?count-1:Math.min(state.index,count-1);if(initial||state.follow||next!==state.index||!state.frame||state.frame.pending)await loadFrame(next);}
    else{$('stage-empty').hidden=false;$('stage-empty').querySelector('strong').textContent='实验尚未写入规划帧';}
  }catch(error){if(episode!==state.episode||epoch!==state.manifestEpoch||requestId!==state.manifestRequest)return;if(error.status===404&&state.isNavsim){await clearUnavailableEpisode();return;}connection('读取失败','bad');if(initial)toast(error.message);}
}
async function loadFrame(index,{fromPlayback=false,fromVideoClock=false}={}){
  const count=state.manifest?.frames?.length||0;if(!count)return;
  if(fromPlayback&&state.fetching)return;
  index=Math.max(0,Math.min(count-1,index));if(state.requested===index&&state.fetching)return;
  const epoch=++state.epoch,episode=state.episode;state.frameController?.abort();state.frameController=new AbortController();state.requested=index;state.fetching=true;
  try{
    const frame=await request(`/api/episodes/${encodeURIComponent(episode)}/frames/${index}`,state.frameController.signal);
    if(epoch!==state.epoch||episode!==state.episode)return;
    state.frame=frame;state.index=index;if(!fromPlayback)state.playTime=finite(frame.time)?frame.time:0;
    renderFrame(frame,{exactVideoSeek:!fromPlayback,fromVideoClock});potentialLabel();if(!$('potential-panel').hidden)loadPotential(frame,{playback:fromPlayback});
  }catch(error){if(error.name!=='AbortError'&&epoch===state.epoch){if(error.status===404&&state.isNavsim){await clearUnavailableEpisode();return;}toast(`帧读取失败：${error.message}`);pause();}}
  finally{if(epoch===state.epoch){state.fetching=false;if(fromVideoClock)syncFromVideo();}}
}
function renderFrame(frame,{exactVideoSeek=false,fromVideoClock=false}={}){
  $('stage-empty').hidden=state.view!=='scene'||!!frame.ego;if(!state.isNavsim||state.view==='scene')scene?.setFrame(frame);
  const vis=frame.visualization||{},diagnostics=frame.plan?.diagnostics||{},selection=diagnostics.planning_object_selection||{};
  for(const name of['sidewalk','junction','crosswalk']){const input=document.querySelector(`[data-layer="${name}"]`),layer=vis.render_geometry?.semantic_surfaces?.[name];input.disabled=!layer?.polygons?.length;input.closest('label').classList.toggle('unavailable',input.disabled);input.closest('label').title=input.disabled?(layer?.reason||'当前归档未提供此地图图层'):layer.source||'归档地图图层';sceneLayers[name]=Boolean(input.checked&&!input.disabled);scene?.setLayer(name,sceneLayers[name]);}
  $('sim-time').replaceChildren(document.createTextNode(fmt(frame.time)),el('small',null,'s'));
  const velocity=frame.ego?.velocity,speed=Array.isArray(velocity)?Math.hypot(...velocity.slice(0,2)):Number(frame.ego?.speed);
  $('ego-speed').replaceChildren(document.createTextNode(finite(speed)?fmt(speed*3.6,1):'—'),el('small',null,'km/h'));
  const count=selection.selected_count??vis.objects?.filter(o=>o.selected||o.potential_source).length;
  $('actor-count').replaceChildren(document.createTextNode(count??'—'),el('small',null,'个'));
  $('frame-id').textContent=frame.frame_id||'未归档';$('alignment').textContent=frame.pending?'等待源帧状态写入':frame.alignment?.policy==='exact_frame_id_and_source_time'?'严格源帧关联':'关联信息未归档';
  const cameras=frame.cameras||[],calibrated=cameras.filter(c=>frame.camera_overlays?.[c.view]?.valid).length;
  $('calibration').textContent=cameras.length?`${calibrated} / ${cameras.length} 路可投影`:'当前源帧无图像';
  const approximate=(vis.objects||[]).filter(o=>o.geometry_quality==='illustrative_2_5d').length;
  $('geometry-quality').textContent=approximate?`${approximate} 对象高度为2.5D示意`:(vis.objects?.length?'归档对象与道路几何':'无对象几何 / 尚未纳入');
  const safe=frame.plan?.safety_feasible??diagnostics.safety_feasible??frame.plan?.safe;
  const emergency=diagnostics.emergency===true||String(diagnostics.execution_mode||frame.plan?.execution_mode||'').includes('emergency'),invalid=frame.plan?.valid===false;
  const badge=$('plan-safety');badge.className=`badge ${emergency||invalid?'bad':safe===true?'good':safe===false?'warn':'subtle'}`;badge.textContent=emergency?'紧急状态':invalid?'轨迹不可执行':safe===true?'安全验收通过':safe===false?'安全验收未通过':frame.plan?'已归档规划':'规划未写入';
  $('scene-description').textContent='规划轨迹与行车走廊';$('scene-description').title=vis.corridor_note||'几何行车走廊，非独立安全证明';
  const warnings=[...(frame.warnings||[]),...(frame.navsim&&state.view!=='scene'?[]:vis.warnings||[])];
  const notable=warnings.filter(w=>!String(w).startsWith('No archived RGB'));$('warnings').hidden=!notable.length;$('warnings').textContent=notable.slice(0,3).join(' · ');$('warnings').title=warnings.join('\n');
  $('camera-count').textContent=`${cameras.length} 路`;$('next-camera-frame').disabled=!state.manifest.frames.some(f=>f.index>state.index&&(f.camera_available||state.manifest.platform==='highway'&&f.captured));gallery.setFrame(frame);
  $('timeline').value=frame.index??state.index;$('timeline-current').textContent=clock(frame.time);$('timeline-frame').textContent=`FRAME ${state.index+1} / ${state.manifest.frames.length}`;
  renderDialogue(frame);renderMemory(frame);renderMetrics(frame);renderBEV(frame);syncVideo(frame.time,{exact:exactVideoSeek,fromVideoClock});
  renderNavsim(frame);
}

let navsimPlotToken=null;
function renderNavsim(frame){
  const n=frame.navsim;$('navsim-panel').hidden=!n;
  $('potential-toggle').disabled=!!n;
  if(!n){navsimPlotToken=null;return;}
  $('alignment').textContent=n.source_frame?'独立源锚点 · t=0':'官方评分回放 · 无重新规划';
  $('scene-description').textContent=state.view==='video'?`${state.manifest?.video?.visualization_label||'归档执行视频'} · 每样本独立4秒`:state.view==='scene'?'自定义三维诊断（非官方绘图） · 源帧局部坐标':'源规划BEV诊断图（非官方绘图） · 仅t=0';
  $('navsim-note').textContent=`锚点 ${n.anchor_token} · ${fmt(frame.time,1)}s · ${n.trace_available?'官方仿真轨迹已归档':'未归档官方仿真，仅展示源帧'} · 所有分数均为最终或源锚点评分，不随时间轴变化。`;
  if(!n.source_frame){$('plan-safety').className='badge subtle';$('plan-safety').textContent='官方评分仿真回放';}
  $('potential-panel').hidden=true;
  if(!$('navsim-panel').open)return;
  if(!window.Plotly){
    if(!navsimPlotLoading){navsimPlotLoading=true;ensurePlotly().then(()=>{if(state.frame?.navsim&&$('navsim-panel').open)renderNavsim(state.frame);}).catch(error=>toast(error.message)).finally(()=>{navsimPlotLoading=false;});}
    return;
  }
  const paths=n.trajectories||{},point=(paths.official_simulated_rear_axle||[])[frame.index];
  const plotKey=`${frame.episode_id}:${n.anchor_token}`;
  if(navsimPlotToken===plotKey){Plotly.restyle('navsim-plot',{x:[point?[point[0]]:[]],y:[point?[point[1]]:[]]},[3]);return;}
  navsimPlotToken=plotKey;
  const specs=[['源规划 · 中心','source_plan_center','#6bdda2'],['提交 · 后轴','submitted_rear_axle','#f3b75e'],['官方仿真 · 后轴','official_simulated_rear_axle','#70b9fa']];
  const traces=specs.map(([name,key,color])=>{const rows=paths[key]||[];return {name,x:rows.map(p=>p[0]),y:rows.map(p=>p[1]),mode:key==='submitted_rear_axle'?'lines+markers':'lines',line:{color,width:2},marker:{size:5},hovertemplate:'x=%{x:.2f} m<br>y=%{y:.2f} m<extra>'+name+'</extra>'};});
  traces.push({name:'当前仿真采样',x:point?[point[0]]:[],y:point?[point[1]]:[],mode:'markers',marker:{size:9,color:'#263b38'},showlegend:false});
  Plotly.react('navsim-plot',traces,{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#52665c',size:12},margin:{l:45,r:12,t:10,b:45},legend:{orientation:'h',y:1.12},xaxis:{title:'源帧 x / m',gridcolor:'#e1e9e2',zerolinecolor:'#c2d1c4'},yaxis:{title:'y / m',scaleanchor:'x',scaleratio:1,gridcolor:'#e1e9e2',zerolinecolor:'#c2d1c4'},uirevision:plotKey},{responsive:true,displayModeBar:false});
}
$('navsim-panel').addEventListener('toggle',()=>{if($('navsim-panel').open&&state.frame?.navsim)renderNavsim(state.frame);});

function setView(view){
  state.view=view;for(const b of $('view-tabs').children)b.classList.toggle('active',b.dataset.view===view);
  $('video-view').hidden=view!=='video';$('bev-view').hidden=view!=='bev';$('scene-canvas').hidden=view!=='scene';$('stage-badges').hidden=view!=='scene';$('stage-help').hidden=view!=='scene';$('stage-empty').hidden=view!=='scene'||!!state.frame?.ego;
  document.querySelector('.scene-actions').hidden=state.isNavsim&&view!=='scene';
  if(view!=='video')$('video').pause();if(view==='scene')ensureScene().then(renderer=>renderer?.render());
  if(state.frame?.navsim)renderNavsim(state.frame);
}
function configureDatasetView(manifest,initial){
  const navsim=manifest.dataset==='navsim',wasNavsim=state.isNavsim;
  if(navsim&&!wasNavsim)state.nonNavsimView=state.view;
  state.isNavsim=navsim;
  $('view-tabs').querySelector('[data-view="scene"]').textContent=navsim?'三维诊断（非官方）':'空间场景';
  $('view-tabs').querySelector('[data-view="video"]').textContent=navsim?(manifest.video?.official_visualization?'官方执行视频':'归档执行视频'):'原始录像';
  $('view-tabs').querySelector('[data-view="bev"]').textContent=navsim?'源规划BEV（诊断）':'BEV图';
  if(initial)setView(navsim?'video':wasNavsim?state.nonNavsimView:state.view);
  if(!navsim){ensureScene();ensurePlotly().then(drawPotential).catch(error=>toast(error.message));}
}

function modelMode(frame){const summary=state.manifest?.summary||{};return summary.execution_mode||frame.configuration?.execution_mode||state.manifest?.execution_mode;}
function chat(container,role,label,time,content,{collapsed=false}={}){
  const event=el('article',`chat-event ${role==='agent'?'agent':''}`);event.append(el('div','chat-icon',role==='agent'?'✧':'↗'));
  const bubble=el('div','chat-bubble'),meta=el('div','chat-meta');meta.append(el('strong',null,label),el('time',null,finite(time)?`${fmt(time)}s`:'源时间未归档'));bubble.append(meta);
  const body=el('div','chat-content');
  const text=json(content);
  if(collapsed)rawDetails(body,'查看完整输入',content);
  else if(content&&typeof content==='object'&&content.explanation){
    body.classList.add('answer-content');
    if(typeof content.explanation==='string')body.append(el('p','answer-explanation',content.explanation));
    else for(const [label,value]of Object.entries(content.explanation)){const part=el('div','answer-section');part.append(el('strong',null,label),el('p',null,typeof value==='string'?value:json(value)));body.append(part);}
    if(content.local_goal){const goal=el('div','answer-goal');goal.append(el('span',null,'局部引导点'));for(const [label,value]of Object.entries(content.local_goal))goal.append(el('span',null,`${label} = ${metricValue(value)}`));body.append(goal);}
    rawDetails(body,'查看完整回答',content);
  }else if(text.length>1800){body.append(document.createTextNode(text.slice(0,900)+'…'));rawDetails(body,'展开完整归档内容',content);}else body.textContent=text;
  bubble.append(body);event.append(bubble);container.append(event);return body;
}
function renderDialogue(frame){
  const panel=$('dialogue-panel');panel.replaceChildren();
  const current=!!(frame.model_input||frame.model_output),active=!current&&!!(frame.active_model_input||frame.active_model_output);
  const input=current?frame.model_input:frame.active_model_input,output=current?frame.model_output:frame.active_model_output;
  const mode=modelMode(frame),pure=mode==='pure_base';
  $('agent-mode-badge').textContent=pure?'基础':input||output?'模型':'空档';
  $('agent-subtitle').textContent=pure?'基础规划':active?'上次决策生效':'决策记录';
  if(!input&&!output){emptyPanel(panel,pure?'本轮未调用模型':'此帧没有模型记录',pure?'本轮使用基础规划。':'点击“模型事件”查看其他调用。','✧');return;}
  const source=input?.frame_id||output?.frame_id||frame.model_source_frame_id;
  const info=el('details','call-details');info.append(el('summary',null,'调用信息'),el('p',null,`${active?'绑定决策来源':'实际模型调用源帧'} · ${source||'未归档'}${active?' · 此帧未重新调用':''}`));
  if(frame.navsim)info.append(el('p',null,'t=0独立锚点的归档记录；评分回放期间无新调用。缺少实际输入图片清单时，不推断发送过哪些图。'));panel.append(info);
  if(input){const body=chat(panel,'user','场景输入',input.time,input.prompt||input.messages||input.input||input,{collapsed:true});
    const images=input.images||input.image_paths||[],urls=input.image_urls||[];
    if(urls.length){
      const attachments=el('div','input-images');
      urls.forEach((asset,i)=>{const url=typeof asset==='string'?asset:asset.url;if(!url)return;const button=el('button','input-image');button.type='button';button.title=`放大实际模型输入图 ${i+1} · ${input.frame_id||source}`;const image=el('img');image.alt=`模型输入图片 ${i+1}`;image.src=url;image.loading='lazy';button.append(image,el('span',null,`输入图 ${i+1} · ${fmt(input.time)}s`));attachments.append(button);
        button.addEventListener('click',()=>{pause();follow(false);$('camera-modal-title').textContent=`实际模型输入图 ${i+1} · ${input.frame_id||source}`;const view=el('div','model-input-enlarged'),large=el('img');large.src=url;large.alt=image.alt;view.append(large);$('camera-modal-content').replaceChildren(view);$('camera-modal-note').textContent=`调用源帧 ${input.frame_id||source} 的原始输入图。此图不叠加当前帧状态。`;$('camera-modal').showModal();});
        image.addEventListener('error',()=>{image.hidden=true;button.append(el('span',null,'输入图资产不可读取'));});
      });body.parentElement.append(attachments);
    }
    if(images.length)body.parentElement.append(el('p','context-caption',`${images.length} 张输入图${urls.length<images.length?` · ${images.length-urls.length} 张资产未找到`:''}`));
  }else panel.append(el('div','chat-empty','模型输入没有归档，无法还原输入内容。'));
  if(output){const content=output.answer||output.raw||output.response||output.output||output;const body=chat(panel,'agent','规划建议',output.time,content);
    if(frame.navsim&&output.raw!==null&&output.raw!==undefined)rawDetails(panel,'模型原始返回（未改写）',output.raw);
    const attention=output.answer?.attention||output.attention||[];if(attention.length){const chips=el('div','attention-chips');for(const obj of attention)chips.append(el('span','attention-chip',String(obj.id||obj.object_id||obj)));body.append(chips);}
    if(output.errors?.length)panel.append(el('div','chat-empty','归档模型解析问题：'+json(output.errors)));
  }else panel.append(el('div','chat-empty','本次调用尚无归档返回。'));
  if(current&&(frame.active_model_input||frame.active_model_output)){
    const previous=el('div','chat-empty');previous.append(el('p',null,`当前控制另绑定旧决策源帧 ${frame.model_source_frame_id||frame.active_model_input?.frame_id||'未归档'}。不与本次新请求配对。`));
    rawDetails(previous,'查看仍在生效的已绑定输入',frame.active_model_input);rawDetails(previous,'查看仍在生效的已绑定输出',frame.active_model_output);panel.append(previous);
  }
}
function renderMemory(frame){
  const panel=$('memory-panel');panel.replaceChildren();const retrieval=frame.retrieval||frame.active_retrieval;
  let records=Array.isArray(retrieval)?retrieval:retrieval?.records||retrieval?.results||retrieval?.matches||[];
  if(!Array.isArray(records))records=[];$('memory-count').textContent=records.length;
  if(!records.length){emptyPanel(panel,'没有检索记忆',modelMode(frame)==='pure_base'?'纯基础规划模式未使用检索记忆。':'当前模型决策没有归档检索结果。','◈');if(frame.navsim&&retrieval)rawDetails(panel,'源锚点归档检索摘要',retrieval);return;}
  panel.append(el('p','context-caption',`真实检索来源 · ${retrieval.frame_id||frame.model_source_frame_id||frame.frame_id}`));
  const scoreList=retrieval.scores||[];
  records.forEach((record,i)=>{
    const card=el('article','memory-card'),head=el('div','memory-header');head.append(el('span','memory-index',`经验 ${String(i+1).padStart(2,'0')}`));card.append(head);
    card.append(el('div','memory-id',record.memory_id||record.id||record.episode_id||`已归档经验 ${i+1}`));
    const score=scoreList[i]||record.scores||record.score||{};
    const entries=typeof score==='number'?[['总分',score]]:Object.entries(score).filter(([k,v])=>typeof v==='number'&&/bev|occ|grid|score|similarity|fusion|total/i.test(k)).slice(0,6);
    for(const [name,value]of entries){const row=el('div','score-row');row.title=`${name}：${value}（归档原值）`;row.append(el('span',null,name.toUpperCase().slice(0,8)));const track=el('span','score-track'),bar=el('i');bar.style.width=`${Math.min(100,Math.max(0,value*100))}%`;track.append(bar);row.append(track,el('span',null,fmt(value,3)));card.append(row);}
    if(!entries.length)card.append(el('p','context-caption','相似度分数未归档'));
    const content=record.experience||record.text||record.description||record.content||record.reasoning||record.answer?.explanation||record.answer;
    if(content)card.append(el('p',null,typeof content==='string'?content:json(content)));else card.append(el('p','context-caption','经验正文未归档 · 可查看实际索引信息'));
    rawDetails(card,'归档检索详情',record);panel.append(card);
  });
  if(frame.navsim&&retrieval?.diagnostics)rawDetails(panel,'源锚点检索诊断摘要',retrieval.diagnostics);
}
function renderMetrics(frame){
  const panel=$('metrics-panel');panel.replaceChildren();const plan=frame.plan||{},d=plan.diagnostics||{},timing=frame.timing||{},audit=frame.execution_audit||{};
  if(frame.navsim){
    const n=frame.navsim,s=n.scores||{},q=s.internal_quality||{},a=s.official_anchor||{},run=s.official_two_stage_aggregate||{},meta=n.source_plan_metadata||{};
    metricList(panel,'ANCHOR · 独立样本',[['源token',n.anchor_token],['回放偏移秒',frame.time],['输入条件',n.input_condition],['源规划可执行',meta.valid],['源规划安全条件满足',meta.diagnostics?.safety_feasible]]);
    metricList(panel,'INTERNAL · 源规划内部质量（非官方分数）',[['内部quality',q.score],['内部有效',q.valid]]);
    metricList(panel,'OFFICIAL · 单锚点（两阶段聚合前）',[['归档PDM score',a.pdm_score],['CSV单锚点score',s.official_csv_anchor?.score],['CSV有效',s.official_csv_anchor?.valid]]);
    metricList(panel,'OFFICIAL · 整轮两阶段聚合',[['官方两阶段EPDMS',run.score],['已评估token',run.evaluated_tokens],['失败token',run.failed_tokens],['split',run.split],['可与纯传感器标准条件直接比较',run.standard_sensor_only_comparable]]);
    panel.append(el('p','context-caption','上述值为源规划或最终评分，回放中不会重算。CSV可包含两帧舒适度等后处理；它与归档单锚点PDM值及整轮两阶段聚合是不同字段。'));
    rawDetails(panel,'内部quality原始字段（已省略内部数组）',s.internal_quality);rawDetails(panel,'官方单锚点原始指标',s.official_anchor);rawDetails(panel,'官方CSV本token行',s.official_csv_anchor);rawDetails(panel,'源规划诊断摘要',meta);rawDetails(panel,'归档时域约定（主轨迹与诊断尾段分开）',n.horizon_contract);rawDetails(panel,'坐标与来源',n.coordinates);rawDetails(panel,'评分回放来源',n.provenance);rawDetails(panel,'源模型/规划耗时',timing);rawDetails(panel,'数据来源证据',frame.evidence);return;
  }
  metricList(panel,'PLANNING · 当前源帧',[
    ['源帧',frame.frame_id],['规划后端',d.planner_backend||state.manifest?.summary?.planner_backend],['轨迹可执行',plan.valid??plan.executable??plan.success],['安全条件满足',plan.safety_feasible??d.safety_feasible],['执行状态',d.execution_mode||plan.execution_mode],['候选来源',d.candidate_source||d.selected_candidate||d.optimizer_termination],['规划耗时',finite(d.planning_seconds)?`${fmt(d.planning_seconds*1000,1)} ms`:finite(plan.solve_seconds)?`${fmt(plan.solve_seconds*1000,1)} ms`:timing.mpc_s?`${fmt(timing.mpc_s*1000,1)} ms`:undefined],['质量评分',frame.quality?.score],
  ]);
  const selection=d.planning_object_selection||{};metricList(panel,'SCOPE · 计算对象',[
    ['原始对象',frame.actors?.length],['纳入规划',selection.selected_count],['常规八向对象',selection.base_selected_count],['安全例外',selection.exception_added_count],['模型额外关注',selection.attention_added_count],
  ]);
  if(Object.keys(audit).length){metricList(panel,'EXECUTION · 归档实际执行',[
    ['审核通过',audit.valid??audit.passed],['源帧',audit.frame_id],['执行备注',audit.reason||audit.status],
  ]);rawDetails(panel,'查看实际执行记录',audit);}
  rawDetails(panel,'时序与耗时原始字段',timing);rawDetails(panel,'质量评分原始字段',frame.quality);rawDetails(panel,'数据来源证据',frame.evidence);
}

function renderBEV(frame){const path=frame.assets?.bev_url||frame.assets?.bev;const img=$('bev-image');img.hidden=!path;if(path){const url=assetUrl(path);if(img.getAttribute('src')!==url)img.src=url;$('bev-status').textContent=`当前源帧 BEV · ${frame.frame_id}`;}else{$('bev-status').textContent='当前规划帧没有归档BEV · 不借用邻帧图像';img.removeAttribute('src');}}
function setupVideo(video){
  const player=$('video'),preview=$('video-preview'),navsim=state.manifest?.dataset==='navsim';
  const poster=video?.preview_url||assetUrl(video?.preview_path);
  if(poster)player.poster=poster;else player.removeAttribute('poster');
  player.dataset.episode=state.episode||'';player.dataset.rendererVersion=video?.renderer_version||'unversioned';player.dataset.officialVisualization=String(video?.official_visualization===true);
  preview.hidden=true;
  if(!video?.available||!video.path&&!video.url){player.pause();player.removeAttribute('src');player.load();player.hidden=true;if(poster){preview.src=poster;preview.hidden=false;preview.alt=video.visualization_label+'首帧';}$('video-status').textContent=navsim?`${poster?video.visualization_label+'首帧 · ':''}此锚点尚无完整执行视频；不会切换成自定义三维主展示。`:'此轮尚无可读取录像。运行中的 MP4 可能需等待编码结束。';return;}
  player.hidden=false;const url=state.compatVideo||video.url||assetUrl(video.path);if(player.getAttribute('src')!==url){player.src=url;player.load();}
  $('video-status').textContent=navsim?`${video.visualization_label} · 独立4秒`:video.sync_available?'录像与时间轴同步':'独立录像 · 无时间映射';
  $('video-status').title=navsim?`${video.renderer_version||'绘图版本未归档'} · ${video.visualization_note||'每个视频覆盖本锚点4秒评分执行。'}`:video.sync_available?'使用归档时间映射':'无逐帧仿真时间映射，不强行同步';
}
function videoTime(simTime){const v=state.manifest?.video;if(!v?.sync_available)return null;const mapping=v.time_map||v.timestamps;if(Array.isArray(mapping)&&mapping.length){const point=mapping.reduce((a,b)=>Math.abs((b.sim_time??b.time)-simTime)<Math.abs((a.sim_time??a.time)-simTime)?b:a);return point.video_time??point.media_time??null;}if(finite(v.simulation_start_time)&&finite(v.media_start_time))return simTime-v.simulation_start_time+v.media_start_time;return null;}
function syncVideo(simTime,{exact=false,fromVideoClock=false}={}){
  // Native playback is the NAVSIM clock. A delayed HTTP frame response must
  // never seek its video backwards. Explicit user selection pauses and seeks
  // precisely, including adjacent endpoint-inclusive 0.1s samples.
  if(state.isNavsim&&fromVideoClock)return;
  const target=videoTime(simTime),video=$('video'),precise=state.isNavsim&&exact;
  if(!finite(target))return;
  if(!finite(video.duration)||video.readyState<1){
    if(precise)state.pendingVideoSeek={episode:state.episode,source:video.getAttribute('src'),time:simTime};
    return;
  }
  if(precise)state.pendingVideoSeek=null;
  if(Math.abs(video.currentTime-target)>(precise ? 0.0001 : 0.15))video.currentTime=Math.max(0,Math.min(video.duration,target));
}
$('video').addEventListener('loadedmetadata',()=>{
  const pending=state.pendingVideoSeek;
  if(pending?.episode===state.episode&&pending.source===$('video').getAttribute('src'))syncVideo(pending.time,{exact:true});
});
$('video').addEventListener('error',()=>{if(!state.manifest)return;$('video-status').textContent=state.isNavsim?`${state.manifest.video?.visualization_label||'归档执行视频'} · 此视频无法直接解码，可生成浏览器兼容副本。`:state.manifest.live?'录像仍在生成，需等待编码完成。场景帧可独立回放。':'此录像编码无法直接播放，可生成独立的浏览器兼容副本。';$('repair-video').hidden=!!state.manifest.live||!state.manifest.video?.available;});
$('video').addEventListener('play',()=>{if(state.isNavsim){pause();follow(false);}});
function syncFromVideo(){
  const video=$('video');if(!state.isNavsim||state.view!=='video'||state.playing||video.dataset.episode!==state.episode)return;
  const times=state.manifest?.times;if(!times?.length||!finite(video.currentTime))return;
  const nearest=times.reduce((best,t,index)=>Math.abs(t-video.currentTime)<Math.abs(times[best]-video.currentTime)?index:best,0);
  if(nearest!==state.requested)loadFrame(nearest,{fromPlayback:true,fromVideoClock:true});
}
$('video').addEventListener('timeupdate',syncFromVideo);
$('repair-video').addEventListener('click',async()=>{
  const episode=state.episode,endpoint=`/api/episodes/${encodeURIComponent(episode)}/video`,button=$('repair-video');button.disabled=true;button.textContent='正在准备兼容副本…';
  const apply=result=>{if(episode!==state.episode)return true;if(result.status==='ready'&&result.url){state.compatVideo=result.url;$('video').src=result.url;$('video').load();button.hidden=true;button.disabled=false;button.textContent='生成浏览器兼容副本';$('video-status').textContent='浏览器兼容副本 · 保留原始录像 · 独立视频回放';return true;}if(['unavailable','failed'].includes(result.status)){button.disabled=false;button.textContent='重新生成兼容副本';$('video-status').textContent=result.error||'当前录像无法生成兼容副本';return true;}return false;};
  try{const result=await request(endpoint,undefined,{method:'POST'});if(apply(result))return;const poll=async()=>{if(episode!==state.episode)return;try{const value=await request(endpoint);if(!apply(value))setTimeout(poll,1500);}catch(error){if(episode===state.episode){button.disabled=false;button.textContent='重新生成兼容副本';$('video-status').textContent=error.message;}}};setTimeout(poll,1200);}
  catch(error){if(episode===state.episode){button.disabled=false;button.textContent='重新生成兼容副本';$('video-status').textContent=error.message;}}
});

function potentialLabel(){
  if(!potentialSourceFrame)return;
  const lag=finite(state.frame?.time)&&finite(potentialSourceFrame.time)?state.frame.time-potentialSourceFrame.time:0;
  $('field-frame').textContent=`势场源帧 ${potentialSourceFrame.frame_id} · ${clock(potentialSourceFrame.time)}${Math.abs(lag)>.01?` · 与主视图相差 ${lag.toFixed(1)}s`:''}`;
}
function drawPotential(){potentialLabel();potentialView.setData(potentialData,potentialSourceFrame);}
async function loadPotential(frame,{playback=false}={}){
  if(playback&&potentialBusy)return;
  const generation=++potentialGeneration,episode=state.episode;potentialBusy=true;
  state.potentialController?.abort();state.potentialController=new AbortController();
  const key=`${state.episode}:${frame.frame_id}:${frame.index??state.index}`;
  if(!playback||!potentialData?.available){potentialData=null;potentialSourceFrame=frame;drawPotential();}
  $('potential-status').className='badge subtle';$('potential-status').textContent='计算归档势场…';
  potentialLabel();
  try{
    const data=potentialCache.get(key)||await request(`/api/episodes/${encodeURIComponent(state.episode)}/potential/${frame.index??state.index}`,state.potentialController.signal);
    if(episode!==state.episode||generation!==potentialGeneration)return;
    if(data.available&&((data.frame_id&&data.frame_id!==frame.frame_id)||(finite(data.time)&&Math.abs(data.time-frame.time)>1e-6)))throw new Error('势场源帧与请求不一致，拒绝叠加');
    potentialSourceFrame=frame;
    if(data.available&&!frame.pending){potentialCache.set(key,data);if(potentialCache.size>12)potentialCache.delete(potentialCache.keys().next().value);}
    potentialData=data;$('potential-status').className=`badge ${data.available?'good':'warn'}`;
    $('potential-status').textContent=data.available?'源帧势场重建':'资料不足';
    $('potential-note').textContent=data.available?
      `${data.goal?(data.sources?.some(s=>s.effective)?'吸引谷 + 排斥峰':'仅吸引谷'):data.sources?.some(s=>s.effective)?'无活动模型目标 · 仅排斥场':'当前无势场源或活动目标 · 零场'} · ${data.width} × ${data.height} 个实际采样 · 原规划域外不定义势能。`:
      data.reason||'缺少车道域、参数或对象证据，无法重建';
    $('potential-note').title=data.note||data.reason||'';drawPotential();
  }catch(error){if(error.name!=='AbortError'&&episode===state.episode&&generation===potentialGeneration){$('potential-status').textContent='重建不可用';potentialData={available:false,reason:error.message};drawPotential();}}
  finally{if(generation===potentialGeneration)potentialBusy=false;}
}
$('field-view').addEventListener('click',e=>{const b=e.target.closest('[data-field-view]');if(!b)return;for(const item of $('field-view').children)item.classList.toggle('active',item===b);potentialView.setMode(b.dataset.fieldView);});
$('field-quantity').addEventListener('change',e=>potentialView.setQuantity(e.target.value));
$('expand-potential').addEventListener('click',()=>{
  const expanded=$('potential-panel').classList.toggle('expanded');$('expand-potential').textContent=expanded?'⊡':'⛶';
  $('expand-potential').setAttribute('aria-label',expanded?'收起势场':'放大势场');potentialView.resize();
});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&$('potential-panel').classList.contains('expanded')){$('potential-panel').classList.remove('expanded');$('expand-potential').textContent='⛶';potentialView.resize();}});

function pause(){state.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','播放');}
function follow(enabled){state.follow=enabled;$('follow-live').classList.toggle('active',enabled);$('follow-live').setAttribute('aria-pressed',String(enabled));if(enabled){pause();loadManifest(false);}}
function play(){if(!state.manifest?.frames.length)return;if(state.playing){pause();return;}follow(false);if(state.isNavsim)$('video').pause();if(state.index>=state.manifest.frames.length-1){state.playTime=state.manifest.times[0]||0;loadFrame(0,{fromPlayback:true});}else state.playTime=state.frame?.time||0;state.playing=true;state.lastWall=performance.now();$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停');}
function manualFrame(index){pause();follow(false);if(state.isNavsim)$('video').pause();loadFrame(index);}
function tick(wall){
  if(state.playing){const delta=Math.min(.15,(wall-state.lastWall)/1000);state.playTime+=delta*state.rate;const times=state.manifest.times;let low=0,high=times.length-1;while(low<high){const mid=Math.ceil((low+high)/2);if((times[mid]??Infinity)<=state.playTime)low=mid;else high=mid-1;}if(low!==state.requested)loadFrame(low,{fromPlayback:true});if(state.index===times.length-1&&!state.fetching&&state.playTime>times.at(-1)+.05)pause();}
  state.lastWall=wall;requestAnimationFrame(tick);
}
function nextSourceFrame(){const m=state.manifest;if(!m)return;let next=m.frames.find(f=>f.index>state.index&&f.camera_available);if(!next&&m.platform?.includes('highway'))next=m.frames.find(f=>f.index>state.index&&f.captured);if(next)manualFrame(next.index);else toast('后续没有归档相机帧');}
function nextEvent(){const next=state.manifest?.model_event_indices?.find(i=>i>state.index);if(next!==undefined)manualFrame(next);else toast('后续没有真实模型调用事件');}

$('run-select').addEventListener('change',async e=>{state.run=e.target.value;configureStageFilter();const summaryRequest=loadRunResults();await applyEpisodeFilters();await summaryRequest;});
$('stage-select').addEventListener('change',async e=>{state.stage=e.target.value;renderRunResults();await applyEpisodeFilters();});
$('episode-select').addEventListener('change',e=>selectEpisode(e.target.value));$('refresh').addEventListener('click',loadEpisodes);
$('play').addEventListener('click',play);$('previous-frame').addEventListener('click',()=>manualFrame(state.index-1));$('next-frame').addEventListener('click',()=>manualFrame(state.index+1));
$('timeline').addEventListener('input',e=>manualFrame(Number(e.target.value)));$('playback-rate').addEventListener('change',e=>state.rate=Number(e.target.value));
$('follow-live').addEventListener('click',()=>follow(!state.follow));$('next-event').addEventListener('click',nextEvent);$('next-camera-frame').addEventListener('click',nextSourceFrame);
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{sceneLayers[input.dataset.layer]=input.checked;scene?.setLayer(input.dataset.layer,input.checked);gallery.updateLayers();}));
$('topdown').addEventListener('click',async()=>{const renderer=await ensureScene();if(!renderer)return;const enabled=!renderer.topdown;renderer.setTopdown(enabled);$('topdown').classList.toggle('active',enabled);$('topdown').setAttribute('aria-pressed',String(enabled));$('stage-badges').firstElementChild.textContent=enabled?'2D TOP VIEW':'3D SCENE';});
$('reset-view').addEventListener('click',async()=>{(await ensureScene())?.reset();});$('capture').addEventListener('click',async()=>{const renderer=await ensureScene();if(!renderer)return;renderer.download(`EP-Driver_${state.frame?.frame_id?.replaceAll(':','_')||'scene'}.png`);toast('已保存当前空间场景截图');});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{toast('当前浏览器不支持全屏');}});
$('view-tabs').addEventListener('click',e=>{const button=e.target.closest('[data-view]');if(button)setView(button.dataset.view);});
$('inspector-tabs').addEventListener('click',e=>{const button=e.target.closest('[data-tab]');if(!button)return;for(const b of $('inspector-tabs').children)b.classList.toggle('active',b===button);for(const name of['dialogue','memory','metrics'])$(`${name}-panel`).hidden=name!==button.dataset.tab;});
function showPotential(show){$('potential-panel').hidden=!show;if(!show)$('potential-panel').classList.remove('expanded');$('potential-toggle').classList.toggle('active',show);if(show){ensurePlotly().then(()=>{if(!$('potential-panel').hidden)drawPotential();}).catch(error=>toast(error.message));if(state.frame)loadPotential(state.frame);}else{potentialGeneration++;potentialBusy=false;state.potentialController?.abort();}}
$('potential-toggle').addEventListener('click',()=>showPotential($('potential-panel').hidden));$('close-potential').addEventListener('click',()=>showPotential(false));
$('close-camera').addEventListener('click',()=>$('camera-modal').close());$('camera-modal').addEventListener('click',e=>{if(e.target===$('camera-modal'))$('camera-modal').close();});
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA|BUTTON|A/.test(e.target.tagName)||$('camera-modal').open)return;if(e.code==='Space'){e.preventDefault();play();}if(e.key==='ArrowRight')manualFrame(state.index+1);if(e.key==='ArrowLeft')manualFrame(state.index-1);});
setInterval(()=>{if(state.follow&&!document.hidden){if(!state.episode)loadEpisodes();else loadManifest(false);}},1000);
follow(state.follow);loadEpisodes();requestAnimationFrame(tick);
