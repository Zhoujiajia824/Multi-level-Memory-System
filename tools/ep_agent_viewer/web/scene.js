/** Real WebGL scene; archived x,y-left,z-up maps to Three.js x,z,-y.
 * Rendering never alters planner geometry or controls the simulation. */
import * as THREE from '/vendor/three.module.js';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {GLTFLoader} from '/vendor/loaders/GLTFLoader.js';
import {DRACOLoader} from '/vendor/loaders/DRACOLoader.js';
import {RoomEnvironment} from '/vendor/environments/RoomEnvironment.js';
import {COLORS,REGION_COLORS,REGION_NAMES,BOX_EDGES,BOX_FACES,clamp,validPoint,number,disposeGroup,WorldGeometry} from './scene-geometry.js';

export class SceneRenderer{
  constructor(canvas,tooltip){
    this.canvas=canvas;this.tooltip=tooltip;this.frame=null;this.topdown=false;this.layers={roads:true,sidewalk:true,junction:true,crosswalk:true,topology:false,regions:true,objects:true,grid_matches:true,sources:true,attention:true,trajectory:true,corridor:true,goal:true};
    this.origin=null;this.ground=0;this.userOrbit=false;this.disposed=false;this.modelState='loading';this.roadKey=null;this.hitTargets=[];
    canvas.dataset.egoModel='loading';canvas.dataset.renderer='three-webgl';
    this.note=document.createElement('div');this.note.className='renderer-note';this.note.textContent='正在加载车辆模型…';canvas.parentElement?.append(this.note);
    try{this.renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});}
    catch(error){this.failed=true;canvas.dataset.renderer='webgl-unavailable';this.note.classList.add('renderer-error');this.note.textContent='WebGL不可用，请启用浏览器硬件加速。其他归档面板仍可使用。';return;}
    this.renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=.94;this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0x0f1a27);this.scene.fog=new THREE.FogExp2(0x0f1a27,.0045);canvas.dataset.sceneTheme='dark';
    this.camera=new THREE.PerspectiveCamera(43,1,.12,900);this.camera.position.set(-16,13,-8);this.orthoCamera=new THREE.OrthographicCamera(-50,50,40,-40,.1,900);this.activeCamera=this.camera;
    this.controls=new OrbitControls(this.camera,canvas);this.controls.enableDamping=true;this.controls.dampingFactor=.09;this.controls.minDistance=7;this.controls.maxDistance=190;this.controls.maxPolarAngle=Math.PI/2-.035;this.controls.minPolarAngle=.08;this.controls.screenSpacePanning=true;this.controls.target.set(8,0,0);
    this.controls.mouseButtons={LEFT:THREE.MOUSE.ROTATE,MIDDLE:THREE.MOUSE.DOLLY,RIGHT:THREE.MOUSE.PAN};this.controls.addEventListener('start',()=>{this.userOrbit=true;if(tooltip)tooltip.hidden=true;});this.controls.addEventListener('change',()=>this.render());
    this._setupLights();this._setupGround();this.groups={};for(const name of['roads','sidewalk','junction','crosswalk','topology','regions','objects','grid_matches','corridor','trajectory','goal']){const group=new THREE.Group();group.name=name;this.groups[name]=group;this.scene.add(group);}this.geometry=new WorldGeometry(this);
    this.egoPivot=new THREE.Group();this.egoPivot.name='recorded-ego-model';this.scene.add(this.egoPivot);
    const pmrem=new THREE.PMREMGenerator(this.renderer),room=new RoomEnvironment();this.environment=pmrem.fromScene(room,.04);this.scene.environment=this.environment.texture;this.scene.environmentIntensity=.65;room.dispose();pmrem.dispose();
    this.raycaster=new THREE.Raycaster();this.raycaster.params.Line.threshold=.3;this.pointer=new THREE.Vector2();this._move=e=>this._hover(e);this._leave=()=>{if(tooltip)tooltip.hidden=true;};canvas.addEventListener('pointermove',this._move);canvas.addEventListener('pointerleave',this._leave);
    this._wheel=e=>{if(this.topdown){e.preventDefault();this.orthoSpan=clamp((this.orthoSpan||75)*Math.exp(e.deltaY*.001),25,260);this._updateTopdown();this.render();}};canvas.addEventListener('wheel',this._wheel,{passive:false});
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(canvas);this._loadModel();this.resize();
    this.renderer.setAnimationLoop(()=>{if(this.disposed||this.canvas.hidden)return;if(!this.topdown)this.controls.update();this.render();});
  }

  _setupLights(){
    this.scene.add(new THREE.HemisphereLight(0xc6def6,0x22303c,1.1));this.keyLight=new THREE.DirectionalLight(0xfff4df,2.2);this.keyLight.position.set(-23,38,-22);this.keyLight.castShadow=true;this.keyLight.shadow.mapSize.set(2048,2048);Object.assign(this.keyLight.shadow.camera,{left:-45,right:45,top:45,bottom:-45,near:1,far:140});this.keyLight.shadow.bias=-.00012;this.keyLight.shadow.normalBias=.035;this.keyLight.shadow.radius=3;this.scene.add(this.keyLight,this.keyLight.target);
    const fill=new THREE.DirectionalLight(0x97bdda,.7);fill.position.set(30,18,30);this.scene.add(fill);
  }
  _setupGround(){
    this.floor=new THREE.Mesh(new THREE.PlaneGeometry(900,900),new THREE.MeshStandardMaterial({color:COLORS.ground,roughness:1,metalness:0}));this.floor.rotation.x=-Math.PI/2;this.floor.position.y=-.05;this.floor.receiveShadow=true;this.scene.add(this.floor);
    this.grid=new THREE.GridHelper(700,140,0x263f4f,0x263744);this.grid.material.transparent=true;this.grid.material.opacity=.12;this.grid.position.y=-.035;this.scene.add(this.grid);
  }

  _loadModel(){
    const draco=new DRACOLoader();draco.setDecoderPath('/vendor/draco/');draco.setDecoderConfig({type:'wasm'});this.draco=draco;const loader=new GLTFLoader();loader.setDRACOLoader(draco);
    loader.load('/assets/ego-tesla-model3.glb',gltf=>{
      if(this.disposed)return;const model=gltf.scene;model.updateMatrixWorld(true);
      // This asset contains the CARLA 0.9.15 Tesla LOD0 body, wheels and lamp
      // meshes, converted to +X forward / +Y up. Its black-paint mask, original
      // interior/tyre/plate textures and PBR glass are already in the GLB.
      // Keep those materials instead of applying the old Ferrari recolouring.
      model.traverse(node=>{if(node.isMesh){node.castShadow=true;node.receiveShadow=true;const originals=Array.isArray(node.material)?node.material:[node.material];const materials=originals.map(original=>{const material=original.clone();if(material.transparent){material.depthWrite=false;node.castShadow=false;}return material;});node.material=materials.length===1?materials[0]:materials;}});
      const aligned=new THREE.Group();aligned.add(model);aligned.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(aligned),size=box.getSize(new THREE.Vector3()),center=box.getCenter(new THREE.Vector3());aligned.position.set(-center.x,-box.min.y,-center.z);
      const template=new THREE.Group();template.add(aligned);this.egoTemplate=template;this.egoTemplateSize=size;this.egoPivot.add(template);this.modelState='loaded';this.canvas.dataset.egoModel='loaded';this.canvas.dataset.egoModelSource='carla-0.9.15-tesla-model3';this.canvas.dataset.egoModelForward='asset-positive-x';this.canvas.dataset.egoModelTitle='CARLA Tesla Model 3 · 黑色';this._updateEgo();this._updateNote();this.render();draco.dispose();
    },undefined,error=>{if(this.disposed)return;this.modelState='failed';this.canvas.dataset.egoModel='failed';this.canvas.dataset.egoModelError=String(error?.message||error);this._updateNote();console.error('独立展示自车模型加载失败',error);});
  }

  _roads(){
    const vis=this.frame.visualization||{},render=vis.render_geometry,polygons=render?.road_surface?.polygons||[],semantic=render?.semantic_surfaces||{},key=JSON.stringify([polygons,render?.markings||[],semantic]);if(key===this.roadKey)return;this.roadKey=key;for(const name of['roads','sidewalk','junction','crosswalk'])disposeGroup(this.groups[name]);const group=this.groups.roads;
    // Only the backend union preserves islands and holes. Never merge old
    // per-lane exteriors on the frontend or draw each section's transverse edge.
    for(const polygon of polygons){const material=new THREE.MeshStandardMaterial({color:COLORS.road,roughness:.91,metalness:.03,side:THREE.DoubleSide}),mesh=this.geometry.polygon(polygon,material,.006);if(mesh)group.add(mesh);for(const ring of[polygon.outer,...(polygon.holes||[])]){const line=this.geometry.line(ring,0x78909f,.34,.019);if(line)group.add(line);}}
    for(const marking of render?.markings||[]){if(marking.kind==='none')continue;const color=String(marking.color||'').toLowerCase().includes('yellow')?0xe1cc8c:COLORS.marking,stripe=this.geometry.stripe(marking.points,number(marking.width_m,.085),color,marking.kind==='broken');if(stripe)group.add(stripe);}
    // Semantic surfaces come from the archive, independently of road base and
    // lane paint. Missing map layers stay empty rather than being fabricated.
    for(const[name,color,offset]of[['sidewalk',0x46515c,.018],['junction',0x344352,.012],['crosswalk',0xd5dde5,.034]]){const layer=semantic[name],surfaces=layer?.polygons||[];for(const polygon of surfaces){const material=new THREE.MeshStandardMaterial({color,roughness:.96,metalness:0,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}),mesh=this.geometry.polygon(polygon,material,offset);if(mesh)this.groups[name].add(mesh);}this.canvas.dataset[name+'Polygons']=String(surfaces.length);this.canvas.dataset[name+'Source']=layer?.source||'unavailable';}
    this.canvas.dataset.roadPolygons=String(polygons.length);this.canvas.dataset.roadHoles=String(polygons.reduce((n,p)=>n+(p.holes?.length||0),0));this.canvas.dataset.roadMarkings=String(render?.markings?.length||0);
  }
  _topology(){
    const group=this.groups.topology;disposeGroup(group);const vis=this.frame.visualization||{},topology=vis.topology||vis.render_geometry?.topology||{};
    for(const edge of topology.edges||[]){const path=edge.path||edge.points;if(!Array.isArray(path))continue;const line=this.geometry.line(path,COLORS.topology,.58,.065,true);if(line){line.userData.topology=edge;group.add(line);this.hitTargets.push(line);}}
    this.canvas.dataset.topologyEdges=String(topology.edges?.length||0);
  }

  _regions(){
    const group=this.groups.regions;disposeGroup(group);let count=0;const regions=this.frame.visualization?.regions||[];for(const region of regions){const color=REGION_COLORS[region.name]||0x6aa8bd;for(const polygon of region.surfaces||region.polygons||[]){const mesh=this.geometry.polygon(polygon,new THREE.MeshBasicMaterial({color,transparent:true,opacity:.22,depthWrite:false,side:THREE.DoubleSide,toneMapped:false}),.045);if(mesh){count++;mesh.userData.region=region;mesh.renderOrder=2;group.add(mesh);this.hitTargets.push(mesh);}for(const ring of[polygon.outer||polygon,...(polygon.holes||[])]){const outline=this.geometry.line([...ring,ring[0]],color,.78,.062);if(outline){outline.renderOrder=3;group.add(outline);}}}}
    this.canvas.dataset.regionCount=String(regions.filter(r=>(r.surfaces||r.polygons)?.length).length);this.canvas.dataset.regionPolygons=String(count);this.canvas.dataset.regionStyle='persistent_fill_solid_outline';
  }

  _objects(){
    const group=this.groups.objects,matches=this.groups.grid_matches;disposeGroup(group);disposeGroup(matches);let matchedCount=0;
    for(const object of this.frame.visualization?.objects||[]){const source=object.potential_source&&this.layers.sources,attention=object.attention&&this.layers.attention,matched=Boolean(object.grid_matched),ordinary=this.layers.objects||source||attention;if(object.corners?.length!==8||!object.corners.every(validPoint))continue;
      let corners=object.corners;if(object.geometry_quality==='illustrative_2_5d'&&Math.abs(this.ground)>1e-5)corners=corners.map(p=>[p[0],p[1],p[2]+this.ground]);const vertices=corners.map(p=>this.geometry.point(p)),color=source?COLORS.source:attention?COLORS.attention:COLORS.object;
      const faceGeometry=new THREE.BufferGeometry();faceGeometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices.flatMap(v=>v.toArray()),3));faceGeometry.setIndex(BOX_FACES);faceGeometry.computeVertexNormals();const face=new THREE.Mesh(faceGeometry,new THREE.MeshBasicMaterial({color,transparent:true,opacity:source?.13:.035,visible:Boolean(ordinary),depthWrite:false,side:THREE.DoubleSide,toneMapped:false}));face.userData.object=object;face.userData.displayed=Boolean(ordinary||(matched&&this.layers.grid_matches));group.add(face);this.hitTargets.push(face);
      if(ordinary){const edges=[];for(const[a,b]of BOX_EDGES)edges.push(...vertices[a].toArray(),...vertices[b].toArray());const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(edges,3));group.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:(source||attention)?.95:.37,toneMapped:false})));}
      if(matched){matchedCount++;const fill=new THREE.Mesh(faceGeometry.clone(),new THREE.MeshBasicMaterial({color:COLORS.match,transparent:true,opacity:.075,depthWrite:false,side:THREE.DoubleSide,toneMapped:false}));matches.add(fill);for(const[a,b]of BOX_EDGES){const curve=new THREE.LineCurve3(vertices[a],vertices[b]),edge=new THREE.Mesh(new THREE.TubeGeometry(curve,1,.027,4,false),new THREE.MeshBasicMaterial({color:COLORS.match,toneMapped:false}));matches.add(edge);}}
      if(attention){const accent=[];for(const[a,b]of BOX_EDGES){const start=vertices[a],end=vertices[b];accent.push(...start.toArray(),...start.clone().lerp(end,.25).toArray(),...end.toArray(),...end.clone().lerp(start,.25).toArray());}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(accent,3));group.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:COLORS.attention,transparent:true,opacity:1,depthTest:false,toneMapped:false})));}
    }
    this.canvas.dataset.gridMatchedObjects=String(matchedCount);
  }

  _trajectory(){
    const vis=this.frame.visualization||{};disposeGroup(this.groups.corridor);disposeGroup(this.groups.trajectory);disposeGroup(this.groups.goal);const ribbon=vis.corridor_ribbon,provenance=vis.trajectory_provenance||{};
    for(const polygon of ribbon?.polygons||[]){const mesh=this.geometry.polygon(polygon,new THREE.MeshBasicMaterial({color:0xe4e52b,transparent:true,opacity:.30,depthWrite:false,side:THREE.DoubleSide,toneMapped:false,fog:false}),.07);if(mesh)this.groups.corridor.add(mesh);}
    const segments=vis.trajectory_segments||(vis.trajectory?.length?[vis.trajectory]:[]);
    const main=this.geometry.plannedTrajectory(vis.trajectory||[],.055,.11);if(main)this.groups.trajectory.add(main);
    this.canvas.dataset.trajectoryColorOrder='original_sample_index_green_to_yellow';this.canvas.dataset.corridorStyle='yellow_fill_without_edges';this.canvas.dataset.trajectoryKind=provenance.kind||'archived_plan_or_unspecified';this.canvas.dataset.trajectorySegments=String(segments.length);this.canvas.dataset.corridorPolygons=String(ribbon?.polygons?.length||0);this.canvas.dataset.corridorSource=ribbon?.source||'unavailable';
    if(vis.goal&&validPoint(vis.goal)){const marker=new THREE.Group(),material=new THREE.MeshBasicMaterial({color:COLORS.goal,side:THREE.DoubleSide,toneMapped:false,fog:false}),ring=new THREE.Mesh(new THREE.RingGeometry(.43,.51,48),material),dot=new THREE.Mesh(new THREE.CircleGeometry(.13,24),new THREE.MeshBasicMaterial({color:0xffffff,side:THREE.DoubleSide,toneMapped:false,fog:false}));ring.rotation.x=dot.rotation.x=-Math.PI/2;marker.add(ring,dot);const ticks=[];for(let angle=0;angle<Math.PI*2;angle+=Math.PI/2)ticks.push(Math.cos(angle)*.63,0,Math.sin(angle)*.63,Math.cos(angle)*.9,0,Math.sin(angle)*.9);const tickGeometry=new THREE.BufferGeometry();tickGeometry.setAttribute('position',new THREE.Float32BufferAttribute(ticks,3));marker.add(new THREE.LineSegments(tickGeometry,new THREE.LineBasicMaterial({color:COLORS.goal,toneMapped:false,fog:false})));marker.position.copy(this.geometry.point(vis.goal,.15));this.groups.goal.add(marker);}
    this.canvas.dataset.goalMarkerStyle='flat_target';
  }

  _updateEgo(){
    if(!this.frame?.ego||!this.origin){this.egoPivot.visible=false;return;}const ego=this.frame.ego,p=ego.position||ego.center;if(!validPoint(p)){this.egoPivot.visible=false;return;}this.egoPivot.visible=this.modelState==='loaded';this.egoPivot.position.copy(this.geometry.point([p[0],p[1],this.ground],.012));this.egoPivot.rotation.y=number(ego.heading??ego.yaw);
    if(this.egoTemplate){const length=number(ego.length,4.5),width=number(ego.width,1.8),raw=this.frame.camera_state?.ego||this.frame.raw?.ego||{},height=number(raw.height_m??raw.height??ego.height,0),scaleL=length/this.egoTemplateSize.x;this.egoTemplate.scale.set(scaleL,height>0?height/this.egoTemplateSize.y:scaleL,width/this.egoTemplateSize.z);this.canvas.dataset.egoLength=String(length);this.canvas.dataset.egoWidth=String(width);this.canvas.dataset.egoHeightEvidence=height>0?'recorded':'model-proportion';}
  }
  _updateNote(){if(this.failed)return;const approximate=(this.frame?.visualization?.objects||[]).some(o=>o.geometry_quality==='illustrative_2_5d'),provenance=this.frame?.visualization?.trajectory_provenance;this.note.textContent=this.modelState==='loading'?'正在加载真实GLTF车辆模型…':this.modelState==='failed'?'车辆模型未能加载 · 未用方块替代':`${approximate?'GLTF自车 · 其他对象高度为2.5D示意':'GLTF自车 · 归档世界几何'}${!this.frame?.visualization?.render_geometry?' · 道路渲染几何未归档':''}${provenance?.kind==='recorded_future_execution'?' · '+(provenance.title||'归档未来实驶，非在线规划'):''}`;}

  _clearGeometry(){
    for(const group of Object.values(this.groups))disposeGroup(group);this.egoPivot.visible=false;this.origin=null;this.roadKey=null;this.hitTargets=[];this.lastEgoPoint=null;if(this.tooltip)this.tooltip.hidden=true;
    for(const key of['roadPolygons','roadHoles','roadMarkings','sidewalkPolygons','junctionPolygons','crosswalkPolygons','regionCount','regionPolygons','topologyEdges','gridMatchedObjects','corridorPolygons','trajectorySegments','labelVisible'])this.canvas.dataset[key]='0';this.canvas.dataset.frameGeometry='unavailable';this.canvas.dataset.trajectoryKind='unavailable';this.canvas.dataset.corridorSource='unavailable';this.canvas.dataset.trajectoryColorOrder='unavailable';this.canvas.dataset.corridorStyle='unavailable';
  }

  setFrame(frame){
    if(this.failed){this.frame=frame;return;}const previous=this.frame,first=!this.origin||!validPoint(previous?.ego?.position||previous?.ego?.center);this.frame=frame;this.canvas.dataset.frameId=String(frame?.frame_id||'unavailable');
    const p=frame?.ego?.position||frame?.ego?.center;if(!validPoint(p)){this._clearGeometry();this.note.textContent=frame?'该帧自车位置无效；已清除上一帧场景':'等待归档场景';this.render();return false;}
    this.ground=number(frame.visualization?.ground_z_m,0);if(!this.origin)this.origin=[p[0],p[1],this.ground];this.hitTargets=[];this._roads();this._topology();this._regions();this._objects();this._trajectory();this._updateEgo();this._applyVisibility();this._updateNote();this.canvas.dataset.frameGeometry='available';const egoPoint=this.geometry.point([p[0],p[1],this.ground]);
    if(first)this.reset();else if(this.userOrbit&&this.lastEgoPoint){const delta=egoPoint.clone().sub(this.lastEgoPoint);this.camera.position.add(delta);this.controls.target.add(delta);}else if(!this.userOrbit)this._followCamera();this.lastEgoPoint=egoPoint;
    this.floor.position.set(egoPoint.x,egoPoint.y-.05,egoPoint.z);this.grid.position.set(Math.round(egoPoint.x/5)*5,egoPoint.y-.035,Math.round(egoPoint.z/5)*5);this.keyLight.position.copy(egoPoint).add(new THREE.Vector3(-23,38,-22));this.keyLight.target.position.copy(egoPoint);this.keyLight.target.updateMatrixWorld();if(this.topdown)this._updateTopdown();this.render();return true;
  }
  _followCamera(){
    if(!this.frame?.ego||!this.origin)return;const ego=this.frame.ego,h=number(ego.heading??ego.yaw),p=ego.position||ego.center,base=this.geometry.point([p[0],p[1],this.ground]),forward=new THREE.Vector3(Math.cos(h),0,-Math.sin(h)),left=new THREE.Vector3(-Math.sin(h),0,-Math.cos(h)),target=base.clone().addScaledVector(forward,8.5);target.y+=.3;
    this.camera.position.copy(base).addScaledVector(forward,-14.5).addScaledVector(left,6.7).add(new THREE.Vector3(0,11.5,0));this.controls.target.copy(target);this.camera.lookAt(target);this.controls.update();
  }
  reset(){if(this.failed)return;this.userOrbit=false;this.orthoSpan=75;this._followCamera();if(this.topdown)this._updateTopdown();this.render();}
  _updateTopdown(){
    if(!this.frame?.ego||!this.origin)return;const ego=this.frame.ego,p=ego.position||ego.center,h=number(ego.heading??ego.yaw),base=this.geometry.point([p[0],p[1],this.ground]),forward=new THREE.Vector3(Math.cos(h),0,-Math.sin(h)),target=base.clone().addScaledVector(forward,9),height=this.orthoSpan||75,aspect=(this.width||1)/(this.height||1);
    this.orthoCamera.left=-height*aspect/2;this.orthoCamera.right=height*aspect/2;this.orthoCamera.top=height/2;this.orthoCamera.bottom=-height/2;this.orthoCamera.position.copy(target).add(new THREE.Vector3(0,180,0));this.orthoCamera.up.copy(forward);this.orthoCamera.lookAt(target);this.orthoCamera.updateProjectionMatrix();
  }
  setTopdown(enabled){if(this.failed)return;this.topdown=enabled;this.activeCamera=enabled?this.orthoCamera:this.camera;this.controls.enabled=!enabled;if(enabled)this._updateTopdown();else this.controls.update();this.render();}
  _applyVisibility(){for(const name of['roads','sidewalk','junction','crosswalk','topology','regions','grid_matches','corridor','trajectory','goal'])this.groups[name].visible=Boolean(this.layers[name]);}
  setLayer(name,enabled){this.layers[name]=enabled;if(this.failed)return;if(['sources','attention','objects','grid_matches'].includes(name)&&this.frame?.ego&&this.origin){this.hitTargets=this.hitTargets.filter(o=>!o.userData.object);this._objects();}this._applyVisibility();this.render();}
  _hover(event){
    if(!this.tooltip||!this.origin||event.buttons||this.failed)return;const rect=this.canvas.getBoundingClientRect();this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);this.raycaster.setFromCamera(this.pointer,this.activeCamera);const hits=this.raycaster.intersectObjects(this.hitTargets.filter(o=>o.parent?.visible!==false&&o.visible!==false),false),hit=hits.find(h=>(h.object.userData.object&&h.object.userData.displayed!==false)||h.object.userData.region||h.object.userData.topology)||hits.find(h=>h.object.userData.object);this.tooltip.hidden=!hit;
    if(hit){const{object,region,topology,displayed}=hit.object.userData;this.tooltip.textContent=object?`${object.id} · ${object.kind}${object.grid_matched?' · 八向 '+(object.grid_roles||[]).map(role=>REGION_NAMES[role]||role).join('/') :''}${object.potential_source?' · 势场来源':''}${object.attention?' · 模型关注':''}${displayed===false?' · 当前框图层已隐藏':''}`:topology?`中心拓扑 ${topology.id||topology.lane_id||'未登记'} · 后继 ${(topology.successors||[]).join(', ')||'未登记'} · 示意连通，不是物理标线`:`${REGION_NAMES[region.name]||region.name}区域 · ${region.object_ids?.length?'代表 '+region.object_ids.join(', '):'无代表对象'} · 沿真实车道链`;this.tooltip.style.left=`${clamp(event.clientX-rect.left+12,7,Math.max(7,rect.width-250))}px`;this.tooltip.style.top=`${Math.max(8,event.clientY-rect.top-32)}px`;}
  }
  _updateLabels(){
    if(!this.width||!this.height||!this.groups)return;const camera=this.activeCamera,labels=[];camera.updateMatrixWorld();this.scene.updateMatrixWorld(true);
    for(const group of Object.values(this.groups)){if(!group.visible)continue;group.traverse(object=>{if(object.userData.labelInfo)labels.push(object);});}
    const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1),occupied=[];let shown=0;
    labels.sort((a,b)=>b.userData.labelInfo.priority-a.userData.labelInfo.priority||camera.position.distanceToSquared(a.userData.labelInfo.anchor)-camera.position.distanceToSquared(b.userData.labelInfo.anchor));
    for(const sprite of labels){const info=sprite.userData.labelInfo,world=sprite.parent.localToWorld(info.anchor.clone()),view=world.clone().applyMatrix4(camera.matrixWorldInverse),projected=world.clone().project(camera);sprite.visible=false;info.leader.visible=false;if(view.z>=-.1||projected.z< -1||projected.z>1||Math.abs(projected.x)>1.1||Math.abs(projected.y)>1.1)continue;
      const h=info.pixelHeight,w=h*info.aspect,cx=(projected.x+1)*this.width/2,cy=(1-projected.y)*this.height/2,candidates=info.allowShift?[[0,0],[0,-h-7],[0,-2*h-14],[-w/2-10,-h/2],[w/2+10,-h/2]]:[[0,0]];
      let chosen=null,best=null,bestOverlap=Infinity;
      for(const[dx,dy]of candidates){const x=clamp(cx+dx,w/2+5,Math.max(w/2+5,this.width-w/2-5)),y=clamp(cy+dy,h/2+5,Math.max(h/2+5,this.height-h/2-5)),rect={left:x-w/2,right:x+w/2,top:y-h/2,bottom:y+h/2},overlap=occupied.reduce((sum,r)=>sum+Math.max(0,Math.min(r.right+4,rect.right)-Math.max(r.left-4,rect.left))*Math.max(0,Math.min(r.bottom+3,rect.bottom)-Math.max(r.top-3,rect.top)),0);if(overlap<bestOverlap){bestOverlap=overlap;best={x,y,rect};}if(!overlap){chosen={x,y,rect};break;}}
      chosen??=info.priority>=100?best:null;if(!chosen)continue;const units=camera.isOrthographicCamera?(camera.top-camera.bottom)/(this.height*camera.zoom):2*(-view.z)*Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)/(this.height*camera.zoom),offsetWorld=right.clone().multiplyScalar((chosen.x-cx)*units).addScaledVector(up,-(chosen.y-cy)*units),position=world.clone().add(offsetWorld);sprite.position.copy(sprite.parent.worldToLocal(position));sprite.scale.set(w*units,h*units,1);sprite.visible=true;occupied.push(chosen.rect);shown++;
      if(offsetWorld.lengthSq()>.01){info.leader.geometry.setFromPoints([info.anchor,sprite.position]);info.leader.visible=true;}
    }
    this.canvas.dataset.labelVisible=String(shown);
  }
  resize(){if(this.failed||this.disposed)return;const rect=this.canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;this.width=rect.width;this.height=rect.height;this.renderer.setSize(rect.width,rect.height,false);this.camera.aspect=rect.width/rect.height;this.camera.updateProjectionMatrix();if(this.topdown)this._updateTopdown();this.render();}
  render(){if(!this.failed&&!this.disposed&&this.renderer&&this.scene){this._updateLabels();this.renderer.render(this.scene,this.activeCamera);}}
  download(name='scene.png'){if(this.failed)return;this.render();const link=document.createElement('a');link.download=name;link.href=this.canvas.toDataURL('image/png');link.click();}
  dispose(){this.disposed=true;this.observer?.disconnect();this.renderer?.setAnimationLoop(null);this.controls?.dispose();this.draco?.dispose();this.canvas.removeEventListener('pointermove',this._move);this.canvas.removeEventListener('pointerleave',this._leave);this.canvas.removeEventListener('wheel',this._wheel);if(this.groups)Object.values(this.groups).forEach(disposeGroup);if(this.egoPivot)disposeGroup(this.egoPivot);this.environment?.dispose();this.renderer?.dispose();this.note?.remove();}
}
