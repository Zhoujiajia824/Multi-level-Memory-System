import {CameraGallery,humanView,projectionDescription} from './cameras.js';
const $=id=>document.getElementById(id);
const node=(tag,cls,text)=>{const element=document.createElement(tag);if(cls)element.className=cls;if(text!==undefined)element.textContent=text;return element;};
const stringify=value=>typeof value==='string'?value:JSON.stringify(value,null,2);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const format=v=>v===null||v===undefined?'未保存':String(v);
const compact=v=>finite(v)?Number(v.toPrecision(6)).toString():format(v);
const timestamp=v=>finite(v)?v.toFixed(6):'未保存';
const roleNames={left_front:'左前',front:'前方',right_front:'右前',left_side:'左侧',ego:'自车',center:'自车',right_side:'右侧',left_rear:'左后',rear:'后方',right_rear:'右后'};
const channelNames={s_m:'纵向位置 s / m',s:'纵向位置 s / m',delta_s:'相对纵向距离 Δs / m',delta_s_m:'相对纵向距离 Δs / m',d:'横向位置 d / m',d_m:'横向位置 d / m',delta_d:'相对横向距离 Δd / m',delta_d_m:'相对横向距离 Δd / m',v_s:'纵向速度 / m·s⁻¹',v_d:'横向速度 / m·s⁻¹',vs_mps:'纵向速度 / m·s⁻¹',vd_mps:'横向速度 / m·s⁻¹',a_s:'纵向加速度 / m·s⁻²',a_d:'横向加速度 / m·s⁻²',as_mps2:'纵向加速度 / m·s⁻²',ad_mps2:'横向加速度 / m·s⁻²',heading_rad:'航向 / rad',heading:'航向 / rad',theta:'航向 / rad'};
const state={summary:null,records:[],total:0,offset:0,limit:40,selected:null,evidence:0,detail:null,listEpoch:0,detailEpoch:0,listController:null,detailController:null,tab:'experience',requestedId:new URLSearchParams(location.search).get('record'),sceneEpoch:0,sceneController:null,sceneKey:null,scenePayload:null,sceneTopdown:false,sceneView:'scene',cameraView:null};
let debounceTimer,toastTimer,sceneRenderer,sceneModulePromise;
const memoryLayers={roads:true,sidewalk:true,junction:true,crosswalk:true,topology:false,objects:true,grid_matches:true,regions:true,trajectory:true,corridor:true,attention:false,sources:false,goal:false};
const memoryGallery=new CameraGallery($('memory-front-gallery'),$('memory-camera-modal'),{assetUrl:path=>`/api/library/asset?id=${encodeURIComponent(state.selected||'')}&path=${encodeURIComponent(path||'')}`,layers:memoryLayers,fitToContainer:true,elements:{title:$('memory-camera-modal-title'),content:$('memory-camera-modal-content'),note:$('memory-camera-modal-note')}});

function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4800);}
async function request(url,signal){const response=await fetch(url,{signal,cache:'no-store'});if(!response.ok){let message;try{message=(await response.json()).error;}catch{}throw new Error(message||`HTTP ${response.status}`);}return response.json();}
function badge(text,style='muted'){return node('span',`badge ${style}`,text);}
function showEmpty(title,message){$('record-detail').hidden=true;$('detail-empty').hidden=false;$('detail-empty').querySelector('h2').textContent=title;$('detail-empty').querySelector('p').textContent=message;}
function sourceName(source){return ({openscene:'OpenScene',nuscenes:'nuScenes',highway:'highway-env'})[source]||source||'来源未登记';}
function statusName(status){return ({active:'库内活动',inactive:'库内停用',retired:'已退役',rejected:'已拒绝'})[status]||status||'状态未登记';}
function evidenceLabel(evidence){const offset=evidence?.offset_s;return finite(offset)&&Math.abs(offset)<1e-5?'情境当下':finite(offset)?`离线后续 +${format(offset)} 秒`:`证据 ${evidence?.index??'—'}`;}
function selectedEvidence(detail=state.detail){const evidence=detail?.evidence||[];return evidence.find(item=>Number(item.index)===Number(detail.selected_evidence??state.evidence))||evidence[0];}

async function initialize(){
  $('library-state').textContent='读取中';$('result-count').textContent='读取经验…';
  try{const summary=await request('/api/library');state.summary=summary;
    if(summary.experiments_configured===false){const link=document.querySelector('nav a[href="/"]');link.textContent='未加载实验';link.removeAttribute('href');link.setAttribute('aria-disabled','true');link.title='如需实验回放，启动时增加 --root 实验结果目录';}
    if(summary.available===false||summary.configured===false){$('library-browser').hidden=true;$('unconfigured').hidden=false;$('unconfigured-reason').textContent=summary.reason||'';$('library-state').textContent='未配置';return;}
    $('library-count').textContent=Number(summary.count??0).toLocaleString('zh-CN');
    const freezeFailed=summary.freeze_valid===false,freezeVerified=summary.frozen&&summary.freeze_valid===true;
    $('library-state').textContent=freezeFailed?'完整性校验未通过':freezeVerified?'清单已校验':summary.frozen?'冻结清单待校验':'已发布清单';$('library-state').className=`badge ${freezeFailed?'warning':freezeVerified?'good':'muted'}`;
    $('library-path').textContent=summary.root||'指定经验库';$('library-path').title=summary.root||'';const hash=summary.manifest_sha256;$('library-fingerprint').textContent=hash?`manifest SHA-256 ${hash.slice(0,16)}…`:'';$('library-fingerprint').title=hash||'';
    const compatibility=summary.compatibility||{},verified=compatibility.verification_status==='verified'&&compatibility.main_reader_compatible===true;
    $('compatibility').textContent=verified?'检索兼容已验证':'检索兼容未验证';$('compatibility').className=`badge ${verified?'good':'warning'}`;
    const sources=Array.isArray(summary.sources)?summary.sources:Object.keys(summary.source_counts||{});for(const source of sources){const key=typeof source==='string'?source:source.source||source.id;const count=summary.source_counts?.[key];$('source-filter').append(new Option(`${sourceName(key)}${count!==undefined?' · '+count:''}`,key));}
    const statuses=Array.isArray(summary.statuses)?summary.statuses:Object.keys(summary.status_counts||{});for(const status of statuses){const key=typeof status==='string'?status:status.status||status.id;$('status-filter').append(new Option(statusName(key),key));}
    const warnings=[...(summary.warnings||[])];if(freezeFailed)warnings.unshift('冻结完整性校验未通过。此状态不代表记录可安全用于正式检索。');
    const warningPanel=$('library-warnings');warningPanel.replaceChildren();warningPanel.hidden=!warnings.length;if(warnings.length){warningPanel.append(node('strong',null,'库级提示'));for(const warning of warnings)warningPanel.append(node('p',null,typeof warning==='string'?warning:stringify(warning)));}
    if(summary.manifest_valid===false){$('library-count').textContent='—';$('library-state').textContent='库清单读取失败';$('library-state').className='badge warning';$('result-count').textContent='清单无效，无法列出经验';showEmpty('库清单暂时不可读取','请检查指定目录中的 manifest.json；当前状态不能解释为库中没有经验。');return;}
    await loadList();if(state.requestedId){const requested=state.requestedId;state.requestedId=null;await loadDetail(requested,0,{direct:true});}
  }catch(error){$('library-state').textContent='连接失败';$('library-state').className='badge warning';$('record-list').append(node('div','empty-list error',error.message));showEmpty('经验库暂时不可读取','请确认独立展示服务已配置 --memory-root，且目录中存在发布清单。');}
}

async function loadList(){
  const epoch=++state.listEpoch;state.listController?.abort();state.listController=new AbortController();const query=new URLSearchParams({q:$('search').value,source:$('source-filter').value,status:$('status-filter').value,offset:state.offset,limit:state.limit});
  $('result-count').textContent='查询已发布清单…';
  try{const result=await request(`/api/library/records?${query}`,state.listController.signal);if(epoch!==state.listEpoch)return;state.records=result.records||[];state.total=result.total||0;state.offset=Number(result.offset??state.offset);renderList();
    if(!state.records.length){if(!state.requestedId)showEmpty('没有匹配的已发布经验','请调整搜索词或来源筛选；页面不会读取清单之外的待审核记录。');}
    else if(!state.selected&&!state.requestedId)await loadDetail(state.records[0].id,0);
  }catch(error){if(error.name==='AbortError'||epoch!==state.listEpoch)return;$('result-count').textContent='查询失败';$('record-list').replaceChildren(node('div','empty-list error',error.message));}
}
function renderList(){
  const list=$('record-list');list.replaceChildren();$('result-count').textContent=`${state.total.toLocaleString('zh-CN')} 条经验`;
  for(const record of state.records){const button=node('button',`record-item ${record.id===state.selected?'active':''}`);button.type='button';button.dataset.record=record.id;button.title=record.id;const top=node('div','item-top');top.append(node('span','source-label',sourceName(record.source)),badge(statusName(record.status)));button.append(top,node('strong',null,String(record.id).length>22?String(record.id).slice(0,18)+'…':record.id));
    const preview=record.text_preview||record.preview;button.append(node('p','item-preview',preview||'没有保存经验原文预览'));button.append(node('div','item-clip',record.clip_id?`clip · ${record.clip_id}`:record.frame_id?`frame · ${record.frame_id}`:'帧来源未登记'));button.addEventListener('click',()=>loadDetail(record.id,0));list.append(button);}
  if(!state.records.length)list.append(node('div','empty-list','没有匹配的已发布记录'));
  const page=Math.floor(state.offset/state.limit)+1,pages=Math.max(1,Math.ceil(state.total/state.limit));$('page-number').textContent=`${page} / ${pages}`;$('previous-page').disabled=state.offset===0;$('next-page').disabled=state.offset+state.limit>=state.total;
}

async function loadDetail(id,evidence=0,{direct=false}={}){
  const epoch=++state.detailEpoch;state.detailController?.abort();state.detailController=new AbortController();
  clearScene();$('record-detail').setAttribute('aria-busy','true');
  if(id!==state.selected){state.selected=id;state.detail=null;state.evidence=evidence;showEmpty('正在读取经验','关联原文、归档证据和实际保存的特征。');renderList();}
  state.detail=null;state.evidence=Number(evidence);
  try{const detail=await request(`/api/library/records/${encodeURIComponent(id)}?evidence=${encodeURIComponent(evidence)}`,state.detailController.signal);if(epoch!==state.detailEpoch)return;state.selected=id;state.evidence=Number(detail.selected_evidence??evidence);state.detail=detail;renderDetail(detail);renderList();const url=new URL(location.href);url.searchParams.set('record',id);history.replaceState(null,'',url);}
  catch(error){if(error.name==='AbortError'||epoch!==state.detailEpoch)return;showEmpty(direct?'该ID不在可读取的发布清单中':'经验记录暂时不可读取',`${error.message}。未按相似ID替换，也不会越过发布清单查找。`);toast(error.message);}
  finally{if(epoch===state.detailEpoch)$('record-detail').setAttribute('aria-busy','false');}
}
function renderDetail(detail){
  const record=detail.record||{},identity=detail.id||record.id||state.selected;$('detail-empty').hidden=true;$('record-detail').hidden=false;$('record-title').textContent='驾驶经验';$('record-title').title=identity;
  const badges=$('record-badges');badges.replaceChildren(badge(sourceName(record.source)),badge(statusName(record.status)));
  $('record-subtitle').textContent=`记录 · ${identity}\n片段 · ${record.clip_id||'未登记'}\n源帧 · ${record.frame_id||'未登记'}\n时间 · ${timestamp(record.time)} s\nMPC验证 · ${record.mpc_tested===true?'已测试，详见记录元数据':'未验证'}${record.pdm_tested===true?'\nPDM验证 · 已测试，详见记录元数据':''}`;
  const integrity=detail.integrity;
  const integrityWarnings=[...(detail.warnings||[])];if(integrity&&['freeze_valid','checksums_verified','record_verified','text_verified'].some(key=>integrity[key]===false)&&!integrityWarnings.length)integrityWarnings.push('记录完整性未通过校验，请检查记录详情。');
  $('record-integrity').textContent=integrityWarnings.join(' · ');$('record-integrity').hidden=!integrityWarnings.length;$('record-integrity').title=integrity?stringify(integrity):'';
  $('experience-text').textContent=detail.experience||'此记录没有保存 experience.txt 原文。';$('record-metadata').textContent=stringify(record);
  renderEvidence(detail);renderFeatures(detail);setTab(state.tab);
}

function renderEvidence(detail){
  renderEvidenceTimes(detail,$('evidence-times'));renderEvidenceTimes(detail,$('scene-evidence-times'));
  const current=selectedEvidence(detail),future=finite(current?.offset_s)&&current.offset_s>1e-5,context=$('evidence-context');context.className=`notice evidence-context ${future?'future':''}`;context.textContent=future?`${evidenceLabel(current)} · 历史回顾，不是在线预测。`:'情境当下 · 原始归档图像';
  const images=$('evidence-images');images.replaceChildren();const assets=detail.assets||{},entries=Object.entries(assets).filter(([,asset])=>asset&&(typeof asset==='string'||asset.url));
  for(const [key,value]of entries){const asset=typeof value==='string'?{url:value}:value;const label=asset.label||asset.view||({bev:'BEV俯视图',native:'原生 / 源数据图',grid:'八向图归档图',feature:'DINO特征输入图'})[key]||(/^source_/.test(key)?`源相机 ${key.replace('source_','')}`:key);const card=node('button','image-card');card.type='button';const image=node('img');image.loading='lazy';image.decoding='async';image.src=asset.url;image.alt=label;
    const caption=node('div','image-caption');caption.append(node('strong',null,label),node('span',null,`${evidenceLabel(current)}${finite(asset.time_offset_s)?' · 曝光偏移 '+format(asset.time_offset_s*1000)+' ms':''}`));card.append(image,caption);card.addEventListener('click',()=>openImage(asset,label,current));image.addEventListener('error',()=>{image.hidden=true;caption.append(node('span','error','归档图片不可读取'));});images.append(card);}
  if(!entries.length)images.append(node('p','missing-note','该证据时刻没有可通过当前浏览接口读取的归档图片。'));
  const sourceCount=entries.filter(([key])=>/^source_|camera/.test(key)).length;
  $('source-camera-note').textContent=sourceCount?'源相机只按真实登记浏览，未绘制未经适配验证的3D投影。源帧关联不等于严格同曝光；不同数据集相机可能异步。':'当前浏览接口未加载源相机；这不代表原始数据集没有相机。库外原始相机引用不会被此页面自动读取。';
}
function renderEvidenceTimes(detail,container){
  container.replaceChildren();const evidence=detail.evidence||[];
  for(const item of evidence){const index=Number(item.index??evidence.indexOf(item)),button=node('button',index===state.evidence?'active':'',evidenceLabel({...item,index}));button.type='button';button.setAttribute('aria-pressed',String(index===state.evidence));button.append(node('small',null,finite(item.time_s)?`${timestamp(item.time_s)} s`:`证据索引 ${index}`));button.addEventListener('click',()=>{if(index!==state.evidence)loadDetail(state.selected,index);});container.append(button);}
}
function openImage(asset,label,evidence){
  $('image-modal-title').textContent=label;$('image-modal-subtitle').textContent=`${state.selected} · ${evidenceLabel(evidence)}`;$('image-modal-img').src=asset.url;$('image-modal-img').alt=label;$('image-modal-note').textContent=`${asset.path||'已登记归档资产'} · ${evidence?.offset_s>0?'离线后续观察，非在线预测。':'情境当下归档证据。'} ${asset.note||'不追加未经验证的几何投影。'}`;$('image-modal').showModal();
}

function renderFeatures(detail){
  const current=selectedEvidence(detail);$('feature-context').textContent=`${evidenceLabel(current)} · ${current?.offset_s>0?'后续帧特征，不是当前检索输入':'当前帧特征，兼容状态见页头'}`;
  const grid=detail.grid,summary=$('grid-summary'),matrix=$('grid-tensor');summary.replaceChildren();matrix.replaceChildren();$('grid-cell-detail').hidden=true;
  const hasSlots=Array.isArray(grid?.slots)&&grid.slots.length===3&&grid.slots.every(row=>Array.isArray(row)&&row.length===3),hasValues=Array.isArray(grid?.values)&&grid.values.length===3&&grid.values.every(row=>Array.isArray(row)&&row.length===3);
  if(grid&&grid.available!==false&&(hasSlots||hasValues)){
    summary.append(badge(`八向张量 ${(grid.shape||[3,3]).join(' × ')}`),badge(grid.storage_version||'存储版本未登记'));
    const slots=grid.slots||[];
    for(let row=0;row<3;row++)for(let col=0;col<3;col++){
      const slot=slots[row]?.[col]||{},values=slot.values||grid.values?.[row]?.[col]||[],role=slot.role||`行${row+1} · 列${col+1}`,button=node('button',`grid-slot ${role==='ego'||role==='center'?'ego':''}`);button.type='button';
      const header=node('div','slot-role');header.append(node('span',null,roleNames[role]||role),node('span','slot-state',slot.state||'状态未登记'));button.append(header,node('div','slot-id',slot.id?`${slot.kind||'对象'} · ${slot.id}`:slot.applicable===false?'不适用':'未保存对象ID'));
      const channels=grid.channels||[];const preview=Array.isArray(values)?values.slice(0,4).map((value,i)=>`${channels[i]||'c'+i}: ${compact(value)}`).join('\n'):'槽位数值未保存';button.append(node('div','slot-values',preview));button.addEventListener('click',()=>{matrix.querySelectorAll('.grid-slot').forEach(item=>item.classList.toggle('selected',item===button));renderGridCell(slot,values,channels,row,col);});matrix.append(button);
    }
  }else{summary.append(node('p','missing-note',grid?.reason||'未保存可读取的3 × 3八向张量。不会从图片或文字重算特征。'));}
  const arrayPanel=$('array-summary');arrayPanel.replaceChildren();const arrays=detail.arrays||detail.feature_summary||[];
  if(arrays.length){const table=node('table'),thead=node('thead'),head=node('tr');for(const label of['资产 / 数组','Shape','Dtype','数值范围'])head.append(node('th',null,label));thead.append(head);table.append(thead);const body=node('tbody');
    for(const array of arrays){const row=node('tr'),name=node('td'),minimum=array.minimum??array.min,maximum=array.maximum??array.max;name.append(node('span',null,array.name||'数组'),node('small',null,array.path||''));row.append(name,node('td',null,Array.isArray(array.shape)?array.shape.join(' × '):'未保存'),node('td',null,array.dtype||'未保存'),node('td',null,finite(minimum)&&finite(maximum)?`${compact(minimum)} ~ ${compact(maximum)}`:'未保存'));body.append(row);}table.append(body);arrayPanel.append(table);
  }else arrayPanel.append(node('div','array-missing','该证据时刻没有可读取的数组摘要。未从图片推断或重建DINO、OCC与八向特征。'));
  $('feature-metadata').textContent=stringify(detail.features||{note:'未保存 features.json 元数据'});
}
function renderGridCell(slot,values,channels,row,col){
  const panel=$('grid-cell-detail');panel.hidden=false;panel.replaceChildren();panel.append(node('h4',null,`${roleNames[slot.role]||slot.role||'槽位'} · [${row}, ${col}] 原始数值`));const list=node('div','value-list');
  const rows=[['对象ID',slot.id],['对象类型',slot.kind],['状态',slot.state],['区域适用',slot.applicable===undefined?'未登记':String(slot.applicable)]];if(Array.isArray(values))values.forEach((value,i)=>rows.push([channelNames[channels[i]]||channels[i]||`channel ${i}`,value]));
  for(const [label,value]of rows){const line=node('div','value-row');line.append(node('span',null,label),node('strong',null,format(value)));list.append(line);}panel.append(list);
}

function sceneEmpty(title,message,status='尚未加载'){
  const panel=$('memory-scene-empty');panel.hidden=false;panel.querySelector('strong').textContent=title;panel.querySelector('p').textContent=message;$('scene-status').textContent=status;$('scene-status').className=`badge ${status==='读取失败'?'warning':'muted'}`;
  $('memory-scene-canvas').hidden=true;for(const id of['memory-topdown','memory-reset-view','memory-scene-capture'])$(id).disabled=true;if(state.sceneView==='front')panel.hidden=true;
}
function clearScene(){
  state.sceneEpoch++;state.sceneController?.abort();state.sceneController=null;state.sceneKey=null;state.scenePayload=null;sceneRenderer?.setFrame(null);
  if(sceneRenderer?.note)sceneRenderer.note.hidden=true;
  sceneEmpty('正在等待所选证据','读取完成后显示该记录、该证据帧的归档几何。');$('memory-scene-metadata').replaceChildren();$('memory-scene-warnings').replaceChildren();$('memory-scene-warnings').hidden=true;$('memory-scene-grid-matches').replaceChildren();$('memory-scene-grid-matches').hidden=true;$('memory-scene-tooltip').hidden=true;
  $('scene-context').className='notice evidence-context';$('scene-context').textContent='正在关联所选证据；切换期间已清除上一帧场景。';for(const key of['record','evidence','frame'])delete $('memory-scene-canvas').dataset[key];
  const regionControl=$('memory-region-control'),input=regionControl.querySelector('input');input.disabled=true;regionControl.classList.add('unavailable');regionControl.title='等待归档几何';
  memoryGallery.setFrame(null);$('memory-camera-select').replaceChildren(new Option('等待归档图像',''));$('memory-camera-select').disabled=true;$('memory-enlarge-front').disabled=true;$('memory-camera-note').textContent='';$('memory-camera-note').hidden=true;$('memory-trajectory-note').textContent='正在核验所选证据的归档后续实驶；切换期间已清除旧轨迹与走廊。';
  for(const id of['memory-trajectory-control','memory-corridor-control']){const control=$(id);control.classList.add('unavailable');control.querySelector('input').disabled=true;}syncSceneView();
}
function sceneRequestIsCurrent(epoch,id,evidence){return epoch===state.sceneEpoch&&state.tab==='scene'&&state.selected===id&&state.evidence===evidence&&state.detail?.id===id;}
async function loadScene(){
  if(state.tab!=='scene'||!state.detail)return;
  const id=state.selected,evidence=Number(state.evidence),key=`${id}:${evidence}`;
  if(state.sceneKey===key&&state.scenePayload){syncSceneView();requestAnimationFrame(()=>sceneRenderer?.resize());return;}
  state.sceneController?.abort();state.sceneController=new AbortController();const signal=state.sceneController.signal,epoch=++state.sceneEpoch;
  sceneEmpty('正在读取归档场景','仅关联所选证据的世界状态和真实道路几何。','读取中');
  try{
    sceneModulePromise??=import('./scene.js').catch(error=>{sceneModulePromise=null;throw error;});
    const [frame,module]=await Promise.all([request(`/api/library/records/${encodeURIComponent(id)}/scene?evidence=${evidence}`,signal),sceneModulePromise]);
    if(!sceneRequestIsCurrent(epoch,id,evidence))return;
    if((frame.record_id??frame.id)!==id||Number(frame.evidence_index)!==evidence)throw new Error('场景返回的记录ID或证据索引不一致；已拒绝混帧显示');
    if(frame.available===false){sceneEmpty('该证据没有可重建的场景',frame.reason||'未保存可读取的场景快照；不从图片或特征反推场景。','未归档');renderSceneMetadata(frame);return;}
    if(!frame.ego||!frame.visualization)throw new Error('场景缺少自车状态或渲染几何，无法核验重建来源');
    const expectedFrame=state.detail.features?.frame_id,currentEvidence=selectedEvidence(),expectedTime=currentEvidence?.time_s;
    if(expectedFrame!==undefined&&String(frame.frame_id)!==String(expectedFrame))throw new Error('场景帧号与所选证据的特征帧号不一致；已拒绝显示');
    if(finite(expectedTime)&&(!finite(frame.time)||Math.abs(frame.time-expectedTime)>1e-5))throw new Error('场景时间与所选证据登记时间不一致；已拒绝显示');
    const position=frame.ego.position||frame.ego.center;if(!Array.isArray(position)||!position.slice(0,2).every(Number.isFinite)||position.length<2)throw new Error('自车坐标无效；已清除上一帧场景');
    const canvas=$('memory-scene-canvas');canvas.hidden=false;
    sceneRenderer??=new module.SceneRenderer(canvas,$('memory-scene-tooltip'));
    if(!sceneRequestIsCurrent(epoch,id,evidence)){canvas.hidden=true;return;}
    sceneRenderer.setFrame(frame);sceneRenderer.setTopdown(state.sceneTopdown);
    if(sceneRenderer.note)sceneRenderer.note.hidden=false;
    for(const input of document.querySelectorAll('[data-memory-layer]')){memoryLayers[input.dataset.memoryLayer]=input.checked&&!input.disabled;sceneRenderer.setLayer(input.dataset.memoryLayer,memoryLayers[input.dataset.memoryLayer]);}
    state.sceneKey=key;state.scenePayload=frame;canvas.dataset.record=id;canvas.dataset.evidence=String(evidence);canvas.dataset.frame=String(frame.frame_id??'');
    $('memory-scene-empty').hidden=true;$('scene-status').textContent=sceneRenderer.failed?'WebGL不可用':'归档静帧';$('scene-status').className=`badge ${sceneRenderer.failed?'warning':'good'}`;
    for(const control of['memory-topdown','memory-reset-view','memory-scene-capture'])$(control).disabled=Boolean(sceneRenderer.failed);
    renderSceneMetadata(frame);renderCameras(frame);syncSceneView();requestAnimationFrame(()=>sceneRenderer?.resize());
  }catch(error){if(error.name==='AbortError'||!sceneRequestIsCurrent(epoch,id,evidence))return;sceneRenderer?.setFrame(null);sceneEmpty('三维场景暂时不可读取',error.message,'读取失败');}
}
function syncSceneView(){
  const front=state.sceneView==='front',ready=Boolean(state.scenePayload?.available&&sceneRenderer),active=state.tab==='scene';
  for(const button of document.querySelectorAll('[data-memory-view]')){const selected=button.dataset.memoryView===state.sceneView;button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));}
  $('memory-front-view').hidden=!front;$('memory-scene-canvas').hidden=!(active&&!front&&ready);$('memory-scene-empty').hidden=front||ready;if(sceneRenderer?.note)sceneRenderer.note.hidden=front||!ready;
  for(const id of['memory-topdown','memory-reset-view'])$(id).disabled=!ready||front||Boolean(sceneRenderer?.failed);
  $('memory-scene-capture').disabled=front?!memoryGallery.frame?.cameras?.length:!ready||Boolean(sceneRenderer?.failed);$('memory-camera-note').hidden=!front||!$('memory-camera-note').textContent;
  if(active&&!front&&ready)requestAnimationFrame(()=>sceneRenderer?.resize());
  if(active&&front)requestAnimationFrame(()=>memoryGallery.resize());
}
function renderCameras(frame){
  const selector=$('memory-camera-select'),warnings=[],cameras=(Array.isArray(frame.cameras)?frame.cameras:[]).filter(camera=>{
    const same=String(camera.frame_id)===String(frame.frame_id)&&(!finite(camera.time)||Math.abs(camera.time-frame.time)<=1e-5)&&['record_id','evidence_index'].every(key=>camera[key]===undefined||String(camera[key])===String(frame[key]));if(!same)warnings.push(`${humanView(camera.view)}：状态绑定不一致，未显示该图。`);return same;
  });
  selector.replaceChildren();for(const camera of cameras)selector.append(new Option(humanView(camera.view),camera.view));
  const preferred=cameras.find(camera=>camera.view===state.cameraView)||cameras.find(camera=>['CAM_F0','CAM_FRONT','front_rgb'].includes(camera.view))||cameras[0];state.cameraView=preferred?.view||null;selector.value=state.cameraView||'';selector.disabled=!cameras.length;$('memory-enlarge-front').disabled=!cameras.length;
  if(!cameras.length){selector.append(new Option('没有可读取的归档前视图',''));memoryGallery.setFrame({...frame,cameras:[]});$('memory-camera-note').textContent=[...warnings,frame.camera_note||frame.camera_data?.reason||'所选证据没有已加载的原始前视相机。未用 native/BEV 冒充 RGB，也未借用其他证据图片。'].join('\n');}
  else{memoryGallery.setFrame({...frame,cameras:[preferred]});const overlay=memoryGallery.items[0]?.overlay||frame.camera_overlays?.[preferred.view];$('memory-camera-note').textContent=[`${humanView(preferred.view)} · 状态帧 ${frame.frame_id}`,projectionDescription(preferred,overlay),...warnings].join('\n');}
  syncSceneView();
}
function captureSceneView(){
  if(state.sceneView!=='front'){sceneRenderer?.download(`${String(state.selected).replace(/[^a-zA-Z0-9_-]/g,'_')}_evidence${state.evidence}.png`);return;}
  const stage=$('memory-front-gallery').querySelector('.camera-stage'),image=stage?.querySelector('img'),overlay=stage?.querySelector('canvas');if(!image?.complete||!image.naturalWidth){toast('归档图像尚未读取完成');return;}
  try{const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const context=canvas.getContext('2d');context.drawImage(image,0,0);if(overlay)context.drawImage(overlay,0,0,canvas.width,canvas.height);const link=document.createElement('a');link.download=`${state.selected}_evidence${state.evidence}_${state.cameraView}.png`;link.href=canvas.toDataURL('image/png');link.click();}catch(error){toast(`当前图片无法导出：${error.message}`);}
}
function renderSceneMetadata(frame){
  const current=selectedEvidence(),future=Number(current?.offset_s)>0;
  $('scene-context').className=`notice evidence-context ${future?'future':''}`;$('scene-context').textContent=`${evidenceLabel(current)} · ${future?'归档回顾，非在线预测':'归档静帧'}`;
  const vis=frame.visualization||{},library=frame.library_scene||{},objects=vis.objects||[],regions=vis.regions||[],metadata=$('memory-scene-metadata');metadata.replaceChildren();
  for(const name of['sidewalk','junction','crosswalk']){const input=document.querySelector(`[data-memory-layer="${name}"]`),layer=vis.render_geometry?.semantic_surfaces?.[name];input.disabled=!layer?.polygons?.length;input.parentElement.classList.toggle('unavailable',input.disabled);input.parentElement.title=input.disabled?(layer?.reason||'当前归档未提供此地图图层'):layer.source||'归档地图图层';memoryLayers[name]=Boolean(input.checked&&!input.disabled);sceneRenderer?.setLayer(name,memoryLayers[name]);}
  const add=(label,value)=>{const cell=node('div');cell.append(node('span',null,label),node('strong',null,value));metadata.append(cell);};
  add('记录ID',frame.record_id??frame.id??'未登记');add('证据 / 帧号',`${frame.evidence_index??'—'} / ${frame.frame_id??'未登记'}`);add('源时间',`${timestamp(frame.time)} s`);add('几何来源',typeof frame.scene_source==='string'?frame.scene_source:stringify(frame.scene_source||'未登记'));
  const provenance=vis.trajectory_provenance||{},capabilities=library.capabilities||{},recorded=provenance.kind==='recorded_future_execution',hasTrajectory=recorded&&(vis.trajectory_segments?.some(segment=>segment.length>1)||vis.trajectory?.length>1),hasCorridor=hasTrajectory&&Boolean(vis.corridor_ribbon?.polygons?.length),topology=vis.topology||vis.render_geometry?.topology||{};
  if(frame.available!==false){add('归档交通对象 / 八向代表',`${objects.length} / ${objects.filter(object=>object.grid_matched).length} 个`);add('对象空间精度',objects.some(object=>object.geometry_quality==='illustrative_2_5d')?'世界XY + 示意高度（2.5D）':'以归档字段为准');add('未来实驶 / 几何走廊',hasTrajectory?`实驶记录${hasCorridor?' + 几何扫掠':' · 无走廊'}`:'有效后续点不足，不绘制');add('实际规划 / 模型语义',frame.plan?'仅按已归档字段显示':'未归档，不补造');add('中心拓扑',`${topology.edges?.length||0} 条归档边 · 非物理标线`);add('路面显示',vis.render_geometry?.road_surface?.polygons?.length?'仅已有局部地图几何 · 孔洞保留':'没有可读取的路面几何');}
  for(const[id,layer,available]of[['memory-trajectory-control','trajectory',hasTrajectory],['memory-corridor-control','corridor',hasCorridor]]){const control=$(id),input=control.querySelector('input');input.disabled=!available;control.classList.toggle('unavailable',!available);control.title=available?(layer==='trajectory'?'离线归档的后续真实行驶，不是规划结果':'沿归档实驶线的车辆宽度扫掠，未作安全认证'):capabilities.recorded_future_trajectory?.reason||'所选证据之后的有效归档点不足';memoryLayers[layer]=Boolean(available&&input.checked);sceneRenderer?.setLayer(layer,memoryLayers[layer]);}
  const topologyInput=document.querySelector('[data-memory-layer="topology"]');topologyInput.disabled=!topology.edges?.length;topologyInput.parentElement.classList.toggle('unavailable',topologyInput.disabled);topologyInput.parentElement.title=topology.note||'仅显示归档道路中心拓扑，不是道路漆线';memoryLayers.topology=topologyInput.checked&&!topologyInput.disabled;sceneRenderer?.setLayer('topology',memoryLayers.topology);
  $('memory-trajectory-note').textContent=hasTrajectory?`${provenance.title||'归档未来实际行驶轨迹'} · 仅离线审计，不是在线规划或预测。${provenance.sampling_note?'\n'+provenance.sampling_note:''}${provenance.anchor_frame_id?'\n未来来源锚点：'+provenance.anchor_frame_id:''}\n几何走廊仅按车辆宽度构造；不代表可通行或安全认证。`:capabilities.recorded_future_trajectory?.reason||'该证据之后没有足够的有效归档实驶点。未绘制替代轨迹或几何走廊。';
  const regionControl=$('memory-region-control'),regionInput=regionControl.querySelector('input'),hasRegions=Array.isArray(regions)&&regions.some(region=>(region.surfaces||region.polygons)?.length),regionReason=vis.region_status?.reason||library.region_status?.reason||'没有归档八向区域多边形；不从对象坐标重算填色区域';
  regionInput.disabled=!hasRegions;regionControl.classList.toggle('unavailable',!hasRegions);regionControl.title=hasRegions?'显示归档八向区域':regionReason;regionControl.querySelector('span').textContent=hasRegions?'八向区域':'八向区域未归档';memoryLayers.regions=Boolean(hasRegions&&regionInput.checked);sceneRenderer?.setLayer('regions',memoryLayers.regions);
  const messages=[];if(frame.available===false)messages.push(frame.reason||'场景快照不可读取');else{
    if(!hasRegions)messages.push(regionReason);
    if(objects.some(object=>object.geometry_quality==='illustrative_2_5d'))messages.push('对象的高度和地面高程未归档；三维框使用示意高度。位置、朝向与平面尺寸来自该证据快照。');
  }
  for(const warning of[...(frame.warnings||[]),...(vis.warnings||[]),...(library.notes||[])]){const message=typeof warning==='string'?warning:stringify(warning);if(message&&!messages.includes(message))messages.push(message);}
  const warnings=$('memory-scene-warnings');warnings.replaceChildren();warnings.hidden=!messages.length;for(const message of messages)warnings.append(node('p',null,message));
  const matches=Array.isArray(vis.grid_matches)?vis.grid_matches:[],panel=$('memory-scene-grid-matches');panel.replaceChildren();panel.hidden=!matches.length;
  if(matches.length){panel.append(node('h4',null,'八向代表对象 · 归档角色'),node('p',null,'角色来自该证据帧的八向图登记；未保存区域面时，仅展示匹配关系。'));const list=node('div','scene-match-list');for(const match of matches){const item=node('div','scene-match');item.append(node('strong',null,roleNames[match.role]||match.role||'区域未登记'),node('span',null,match.id?`${match.kind||'对象'} · ${match.id}`:match.state||'没有登记对象'));if(match.id&&match.actor_available===false)item.append(node('small',null,'快照未包含该ID的对象几何'));list.append(item);}panel.append(list);}
}
function setTab(tab){
  const leaving=state.tab==='scene'&&tab!=='scene';state.tab=tab;document.querySelectorAll('[data-tab]').forEach(button=>{const active=button.dataset.tab===tab;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));});for(const name of['experience','scene','evidence','features'])$(`${name}-tab`).hidden=name!==tab;
  if(leaving){state.sceneEpoch++;state.sceneController?.abort();$('memory-scene-canvas').hidden=true;$('memory-scene-tooltip').hidden=true;}
  if(tab==='scene')loadScene();
}
document.querySelector('.detail-tabs').addEventListener('click',event=>{const button=event.target.closest('[data-tab]');if(button)setTab(button.dataset.tab);});
function filterChanged(){state.offset=0;state.selected=null;state.detail=null;state.detailEpoch++;state.detailController?.abort();clearScene();showEmpty('正在查询已发布经验','只浏览发布清单内匹配的记录。');loadList();}
$('search').addEventListener('input',()=>{clearTimeout(debounceTimer);debounceTimer=setTimeout(filterChanged,300);});
for(const id of['source-filter','status-filter'])$(id).addEventListener('change',filterChanged);
$('previous-page').addEventListener('click',()=>{state.offset=Math.max(0,state.offset-state.limit);loadList();$('record-list').scrollTop=0;});$('next-page').addEventListener('click',()=>{state.offset+=state.limit;loadList();$('record-list').scrollTop=0;});
$('close-image').addEventListener('click',()=>$('image-modal').close());$('image-modal').addEventListener('click',event=>{if(event.target===$('image-modal'))$('image-modal').close();});
$('memory-topdown').addEventListener('click',()=>{state.sceneTopdown=!state.sceneTopdown;$('memory-topdown').setAttribute('aria-pressed',String(state.sceneTopdown));$('memory-topdown').textContent=state.sceneTopdown?'三维':'俯视';sceneRenderer?.setTopdown(state.sceneTopdown);});
$('memory-reset-view').addEventListener('click',()=>sceneRenderer?.reset());
$('memory-scene-capture').addEventListener('click',captureSceneView);
for(const input of document.querySelectorAll('[data-memory-layer]'))input.addEventListener('change',()=>{memoryLayers[input.dataset.memoryLayer]=input.checked&&!input.disabled;sceneRenderer?.setLayer(input.dataset.memoryLayer,memoryLayers[input.dataset.memoryLayer]);memoryGallery.updateLayers();});
for(const button of document.querySelectorAll('[data-memory-view]'))button.addEventListener('click',()=>{state.sceneView=button.dataset.memoryView;syncSceneView();});
$('memory-camera-select').addEventListener('change',()=>{state.cameraView=$('memory-camera-select').value;if(state.scenePayload)renderCameras(state.scenePayload);});
$('memory-enlarge-front').addEventListener('click',()=>$('memory-front-gallery').querySelector('.camera-card')?.click());
$('memory-close-camera').addEventListener('click',()=>$('memory-camera-modal').close());$('memory-camera-modal').addEventListener('click',event=>{if(event.target===$('memory-camera-modal'))$('memory-camera-modal').close();});
window.addEventListener('pagehide',event=>{state.sceneController?.abort();if(!event.persisted)sceneRenderer?.dispose();});
initialize();
