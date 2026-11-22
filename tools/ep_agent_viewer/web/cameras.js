/* Exact-source image overlays. Never pair a neighboring image with this state. */
const viewNames={front_rgb:'前视 FRONT',CAM_F0:'前视 CAM F0',CAM_FRONT:'前视 CAM FRONT',front_left_rgb:'左前 FRONT LEFT',front_right_rgb:'右前 FRONT RIGHT',back_rgb:'后视 REAR',back_left_rgb:'左后 REAR LEFT',back_right_rgb:'右后 REAR RIGHT',highway_topdown:'HIGHWAY · 环境俯视'};
export const humanView = view => viewNames[view] || view || '相机';
const roles={front:'前方',rear:'后方',left_front:'左前',right_front:'右前',left_side:'左侧',right_side:'右侧',left_rear:'左后',right_rear:'右后',ego:'自车',center:'自车'};
const shortId=id=>{const value=String(id??'未登记').replace(/^carla_/,'');return value.length>16?`${value.slice(0,6)}…${value.slice(-4)}`:value;};
export const plannedPathColor=progress=>`rgb(${Math.round(Math.max(0,Math.min(1,progress))*255)},255,0)`;
function hasTrajectoryProgress(overlay){const segments=overlay.trajectory_segments||[],progress=overlay.trajectory_segments_progress;return Array.isArray(progress)&&segments.every((points,index)=>Array.isArray(progress[index])&&progress[index].length===points.length&&progress[index].every(value=>Number.isFinite(value)&&value>=0&&value<=1));}
export function boundOverlay(frame,camera,overlay){
  if(!overlay?.valid)return overlay;
  for(const item of[camera,overlay]){
    if(item.frame_id===undefined||String(item.frame_id)!==String(frame?.frame_id))return {...overlay,valid:false,reason:'图像或投影的状态帧号与当前证据不一致；已拒绝叠加'};
    for(const key of['record_id','evidence_index'])if(item[key]!==undefined&&frame?.[key]!==undefined&&String(item[key])!==String(frame[key]))return {...overlay,valid:false,reason:'图像或投影的记录/证据索引不一致；已拒绝叠加'};
    if(Number.isFinite(item.time)&&Number.isFinite(frame?.time)&&Math.abs(item.time-frame.time)>1e-5)return {...overlay,valid:false,reason:'图像或投影绑定的状态时间不一致；已拒绝叠加'};
  }
  return overlay;
}
export function projectionDescription(camera,overlay){
  if(!overlay?.valid)return `${overlay?.reason||'缺少可验证的投影标定'}；仅浏览归档原图。`;
  const notes=[overlay.projection_note||'按归档标定投影；未推断遮挡。'];
  if(Number.isFinite(camera.time_offset_s))notes.push(`相机曝光相对状态时间 ${camera.time_offset_s>=0?'+':''}${(camera.time_offset_s*1000).toFixed(2)} ms。`);
  if(overlay.recorded_future_execution)notes.push('绿到黄线为离线归档未来实驶，黄色带为几何走廊；不是当前规划输出或安全认证。');
  if(overlay.trajectory_segments?.length&&!hasTrajectoryProgress(overlay))notes.push('旧投影缺少原始规划时序索引，使用固定绿线；不会把可见片段重新编号为完整绿黄渐变。');
  return notes.join(' ');
}

function line(context,segments,color,width=1.3,dash=[]){
  context.strokeStyle=color;context.lineWidth=width;context.setLineDash(dash);
  for(const segment of segments||[]){if(!Array.isArray(segment)||segment.length<2)continue;context.beginPath();context.moveTo(...segment[0]);for(const p of segment.slice(1))context.lineTo(...p);context.stroke();}context.setLineDash([]);
}
function polygons(context,items,color,stroke){
  for(const points of items||[]){if(!Array.isArray(points)||points.length<3)continue;context.beginPath();context.moveTo(...points[0]);for(const p of points.slice(1))context.lineTo(...p);context.closePath();context.fillStyle=color;context.fill();if(stroke){context.strokeStyle=stroke;context.lineWidth=.7;context.stroke();}}
}
function compoundPolygons(context,items,color){
  context.beginPath();let count=0;for(const points of items||[]){if(!Array.isArray(points)||points.length<3)continue;context.moveTo(...points[0]);for(const point of points.slice(1))context.lineTo(...point);context.closePath();count++;}if(count){context.fillStyle=color;context.fill();}
}
function plannedTrajectory(context,overlay,width){
  const segments=overlay.trajectory_segments||[],ordered=hasTrajectoryProgress(overlay);context.lineWidth=width;context.lineCap='round';context.lineJoin='round';context.setLineDash([]);
  if(ordered){for(let index=0;index<segments.length;index++){const points=segments[index],progress=overlay.trajectory_segments_progress[index];for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i];if(!a.every(Number.isFinite)||!b.every(Number.isFinite))continue;const gradient=context.createLinearGradient(a[0],a[1],b[0],b[1]);gradient.addColorStop(0,plannedPathColor(progress[i-1]));gradient.addColorStop(1,plannedPathColor(progress[i]));context.strokeStyle=gradient;context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke();}}}
  else line(context,segments,'#00ff00',width);
  if(!segments.length)for(const sample of overlay.trajectory_samples||[]){const point=sample.point||sample.pixel;if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite))continue;const progress=overlay.trajectory_source_point_count===1?0:Number.isFinite(sample.progress)?sample.progress:0;context.fillStyle=plannedPathColor(progress);context.beginPath();context.arc(point[0],point[1],width*1.2,0,Math.PI*2);context.fill();}
  return ordered?'original_sample_index_green_to_yellow':'fallback_fixed_green';
}
export function drawOverlay(canvas,overlay,layers){
  const previous=canvas.getContext('2d');previous.clearRect(0,0,canvas.width,canvas.height);
  if(canvas.dataset){canvas.dataset.trajectoryGradient='unavailable';canvas.dataset.corridorStyle='unavailable';}
  if(!overlay?.valid)return;
  const width=Number(overlay.width),height=Number(overlay.height);if(!width||!height)return;
  canvas.width=width;canvas.height=height;const c=canvas.getContext('2d');c.clearRect(0,0,width,height);
  const scale=Math.max(1,width/960);c.save();
  if(layers.regions)for(const region of overlay.region_polygons||[])polygons(c,region.polygons,'#7db5e214','#95c7ef60');
  // Projection polygons may be a triangulated surface. Fill one compound
  // path once; triangle edges are not corridor boundaries or road markings.
  if(layers.corridor)compoundPolygons(c,overlay.corridor_polygons,'#e4e52b4d');
  if(layers.trajectory){const order=plannedTrajectory(c,overlay,2.5*scale);if(canvas.dataset)canvas.dataset.trajectoryGradient=order;}
  if(canvas.dataset)canvas.dataset.corridorStyle='yellow_fill_without_edges';
  const labels=[];
  for(const box of overlay.boxes||[]){
    const attention=box.attention&&layers.attention,source=box.potential_source&&layers.sources,matched=box.grid_matched&&layers.grid_matches!==false;
    if(!layers.objects&&!attention&&!source&&!matched)continue;
    line(c,box.segments,matched?'#ffba68':attention?'#cf9cff':source?'#8dff9a':'#8bc7ef',(matched?2.7:attention?2:1.25)*scale);
    const points=(box.segments||[]).flat().filter(p=>Array.isArray(p)&&p.every(Number.isFinite));if(points.length&&(source||attention||matched)){const role=(box.grid_roles||[]).map(value=>roles[value]||value).join('/'),label=`${matched?role||'八向匹配':attention?'模型关注':'势场来源'} · ${shortId(box.id)}`;c.font=`600 ${13*scale}px "Microsoft YaHei",sans-serif`;const w=Math.min(width-12*scale,c.measureText(label).width+14*scale),h=24*scale,x=Math.max(4*scale,Math.min(width-w-4*scale,Math.min(...points.map(p=>p[0])))),anchorY=Math.min(...points.map(p=>p[1]));let y=Math.max(3*scale,anchorY-h-3*scale);
      while(labels.some(r=>x<r.x+r.w&&x+w>r.x&&y<r.y+r.h&&y+h>r.y)&&y>h+4*scale)y-=h+3*scale;
      labels.push({x,y,w,h});c.fillStyle='#0b1a2af0';c.fillRect(x,y,w,h);c.strokeStyle=matched?'#ffba68':attention?'#cf9cff':'#8dff9a';c.lineWidth=scale;c.strokeRect(x,y,w,h);c.fillStyle=matched?'#a5f4ff':attention?'#dcc1ff':'#9df5bc';c.fillText(label,x+7*scale,y+16.5*scale,w-14*scale);}
  }
  if(layers.goal&&overlay.goal){const [x,y]=overlay.goal;c.strokeStyle='#ffe38b';c.lineWidth=2*scale;c.beginPath();c.moveTo(x,y-9*scale);c.lineTo(x+7*scale,y);c.lineTo(x,y+9*scale);c.lineTo(x-7*scale,y);c.closePath();c.stroke();}
  c.restore();
}

export class CameraGallery{
  constructor(container,modal,{assetUrl,layers,onOpen,elements={},fitToContainer=false}){this.container=container;this.modal=modal;this.assetUrl=assetUrl;this.layers=layers;this.onOpen=onOpen;this.items=[];this.generation=0;this.fitToContainer=fitToContainer;this.elements={title:elements.title||document.getElementById('camera-modal-title'),content:elements.content||document.getElementById('camera-modal-content'),note:elements.note||document.getElementById('camera-modal-note')};modal.addEventListener('close',()=>{this.ownsModal=false;this.items=this.items.filter(item=>!item.modal);});if(fitToContainer){this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(container);}}
  setFrame(frame){
    if(this.ownsModal&&this.modal.open)this.modal.close();
    this.generation++;this.frame=frame;this.items=[];this.container.replaceChildren();
    const cameras=Array.isArray(frame?.cameras)?frame.cameras:[];
    if(!cameras.length){const el=document.createElement('div');el.className='camera-empty';el.textContent=frame?.record_id?'所选离线证据没有可读取的归档前视图；未借用其他证据的图片。':'当前源帧无归档图像。未使用邻帧旧图，可跳至下一归档相机帧。';this.container.append(el);return;}
    for(const camera of cameras){
      const overlay=boundOverlay(frame,camera,frame.camera_overlays?.[camera.view]);const card=document.createElement('button');card.type='button';card.className='camera-card';card.title='放大查看当前证据的归档图像与叠加';
      const stage=this.createStage(camera,overlay,frame);card.append(stage);
      const bottom=document.createElement('div');bottom.className='camera-projection-note';
      bottom.textContent=overlay?.valid?(overlay.display_title||'归档标定投影 · 精度见说明'):camera.view==='highway_topdown'?'原始环境画面 · 无透视标定':'归档图像 · '+(overlay?.reason||'缺少投影标定');bottom.title=projectionDescription(camera,overlay);
      card.append(bottom);card.addEventListener('click',()=>this.open(camera,overlay,frame));this.container.append(card);
    }
  }
  createStage(camera,overlay,frame=this.frame,{modal=false}={}){
    const generation=this.generation;const stage=document.createElement('div');stage.className='camera-stage';stage.dataset.frame=String(frame?.frame_id??'');stage.dataset.evidence=String(frame?.evidence_index??'');stage.dataset.record=String(frame?.record_id??'');
    const img=document.createElement('img');img.alt=humanView(camera.view);img.loading='lazy';img.decoding='async';
    if(camera.width&&camera.height)stage.style.aspectRatio=`${camera.width} / ${camera.height}`;
    const loading=document.createElement('span');loading.className='camera-loading';loading.textContent='读取归档图像…';stage.append(loading);
    img.src=camera.url||this.assetUrl(camera.path||camera.image||camera.rgb_path);stage.append(img);
    const canvas=document.createElement('canvas');canvas.setAttribute('aria-hidden','true');stage.append(canvas);
    const label=document.createElement('span');label.className='camera-label';label.textContent=humanView(camera.view);stage.append(label);
    const stamp=document.createElement('span');stamp.className='camera-frame';stamp.textContent=Number.isFinite(camera.time)?`${camera.time.toFixed(frame?.record_id?6:2)}s`:'归档源帧';stage.append(stamp);
    const item={stage,img,canvas,overlay,modal,loaded:false};this.items.push(item);
    img.addEventListener('load',()=>{if(generation!==this.generation)return;loading.remove();stage.style.aspectRatio=`${img.naturalWidth} / ${img.naturalHeight}`;item.loaded=true;if(overlay?.valid&&(img.naturalWidth!==Number(overlay.width)||img.naturalHeight!==Number(overlay.height))){item.overlay={...overlay,valid:false,reason:'图像尺寸与投影画幅不一致；已拒绝叠加'};const warning=document.createElement('span');warning.className='camera-overlay-warning';warning.textContent=item.overlay.reason;stage.append(warning);}drawOverlay(canvas,item.overlay,this.layers);this.resize();});
    img.addEventListener('error',()=>{if(generation!==this.generation)return;loading.remove();item.loaded=false;drawOverlay(canvas,null,this.layers);img.hidden=true;const note=document.createElement('span');note.className='camera-load-error';note.textContent='归档图像无法读取';stage.append(note);});
    return stage;
  }
  updateLayers(){this.items=this.items.filter(item=>item.canvas.isConnected);for(const {canvas,overlay,loaded} of this.items)drawOverlay(canvas,loaded?overlay:null,this.layers);}
  resize(){if(!this.fitToContainer)return;for(const item of this.items){if(!item.loaded||!item.stage.isConnected)continue;const availableWidth=item.modal?this.elements.content.clientWidth:this.container.clientWidth,availableHeight=item.modal?innerHeight*.72:Math.max(1,this.container.clientHeight-32),ratio=item.img.naturalWidth/item.img.naturalHeight;if(!availableWidth||!ratio)continue;const width=Math.min(availableWidth,availableHeight*ratio);item.stage.style.width=`${width}px`;item.stage.style.height=`${width/ratio}px`;}}
  open(camera,overlay,frame=this.frame){
    if(frame!==this.frame)return;
    this.onOpen?.();
    this.items=this.items.filter(item=>!item.modal);this.elements.title.textContent=`${humanView(camera.view)} · ${frame.frame_id}`;
    this.elements.content.replaceChildren(this.createStage(camera,overlay,frame,{modal:true}));
    this.elements.note.textContent=projectionDescription(camera,overlay);this.ownsModal=true;
    if(!this.modal.open)this.modal.showModal();requestAnimationFrame(()=>this.resize());
  }
}
