import * as THREE from '/vendor/three.module.js';

export const COLORS={road:0x27323e,ground:0x0b131d,marking:0xdde4e9,path:0x7dffda,source:0x64eea0,attention:0xc499ff,object:0x93b7d3,match:0xffb65d,topology:0x71b7f1,goal:0x69c5ff};
export const REGION_COLORS={left_front:0x64aaca,front:0x60caba,right_front:0x6da7c5,left_side:0x9b9ce5,right_side:0x80afce,left_rear:0x7795ca,rear:0x7799b4,right_rear:0x709aac};
export const REGION_NAMES={front:'前方',rear:'后方',left_front:'左前',right_front:'右前',left_side:'左侧',right_side:'右侧',left_rear:'左后',right_rear:'右后',ego:'自车',center:'自车'};
export const shortId=value=>{const text=String(value??'未知ID').replace(/^carla_/,'');return text.length>16?`${text.slice(0,6)}…${text.slice(-4)}`:text;};
export const BOX_EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
export const BOX_FACES=[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7];
export const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
export const validPoint=p=>Array.isArray(p)&&p.length>=2&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
export const number=(v,fallback=0)=>Number.isFinite(Number(v))?Number(v):fallback;

export function disposeGroup(group){
  const materials=new Set(),geometries=new Set(),textures=new Set();
  group.traverse(object=>{if(object.geometry)geometries.add(object.geometry);for(const material of Array.isArray(object.material)?object.material:[object.material])if(material){materials.add(material);if(material.map)textures.add(material.map);}});
  group.clear();geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());textures.forEach(t=>t.dispose());
}

/** Tube cross sections are rounded; the center follows only the archived
 * piecewise-linear trajectory, without Catmull-Rom turn overshoot. */
class RecordedPolyline extends THREE.Curve{
  constructor(points){super();this.points=points;this.arc=[0];for(let i=1;i<points.length;i++)this.arc.push(this.arc.at(-1)+points[i].distanceTo(points[i-1]));this.total=this.arc.at(-1);}
  getPoint(t,target=new THREE.Vector3()){const distance=clamp(t,0,1)*this.total;let lo=1,hi=this.arc.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(this.arc[mid]<distance)lo=mid+1;else hi=mid;}const part=(distance-this.arc[lo-1])/Math.max(1e-9,this.arc[lo]-this.arc[lo-1]);return target.copy(this.points[lo-1]).lerp(this.points[lo],part);}
}
class TemporalPolyline extends RecordedPolyline{
  constructor(points,progress){super(points);this.progress=progress;}
  // RecordedPolyline already uses arc length as its curve parameter. Avoid
  // remapping it through sampled lengths, then recover the original sample
  // index (time order) independently of unequal geometric segment lengths.
  getUtoTmapping(u,distance){return distance===undefined?u:distance/Math.max(this.total,1e-9);}
  progressAt(t){const distance=clamp(t,0,1)*this.total;let i=1;while(i<this.arc.length-1&&this.arc[i]<distance)i++;const part=(distance-this.arc[i-1])/Math.max(1e-9,this.arc[i]-this.arc[i-1]);return this.progress[i-1]+clamp(part,0,1)*(this.progress[i]-this.progress[i-1]);}
}

export class WorldGeometry{
  constructor(owner){this.owner=owner;}
  point(p,offset=0){const s=this.owner;return new THREE.Vector3(p[0]-s.origin[0],number(p[2],s.ground)-s.origin[2]+offset,-(p[1]-s.origin[1]));}
  polygon(polygon,material,offset=0){
    const s=this.owner,outer=polygon?.outer||polygon;if(!Array.isArray(outer)||outer.length<3||!outer.every(validPoint))return null;
    // Triangulate in map XY while retaining each recorded vertex's elevation.
    // Averaging Z would flatten slopes, raised sidewalks and separate decks.
    const openRing=ring=>ring.length>3&&ring[0][0]===ring.at(-1)[0]&&ring[0][1]===ring.at(-1)[1]?ring.slice(0,-1):ring.slice(),rings=[openRing(outer),...(polygon.holes||[]).filter(r=>r.length>=3&&r.every(validPoint)).map(openRing)],flat=rings.flat(),vectors=rings.map(r=>r.map(p=>new THREE.Vector2(p[0]-s.origin[0],p[1]-s.origin[1]))),faces=THREE.ShapeUtils.triangulateShape(vectors[0],vectors.slice(1));
    if(!faces.length)return null;const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(flat.flatMap(p=>this.point(p,offset).toArray()),3));geometry.setIndex(faces.flat());geometry.computeVertexNormals();const mesh=new THREE.Mesh(geometry,material);mesh.receiveShadow=true;return mesh;
  }
  line(points,color,opacity=1,offset=0,dashed=false){
    if(!Array.isArray(points)||points.length<2||!points.every(validPoint))return null;const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>this.point(p,offset)));
    const material=dashed?new THREE.LineDashedMaterial({color,transparent:true,opacity,dashSize:2.2,gapSize:2.5,depthWrite:false,toneMapped:false}):new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity,depthWrite:false,toneMapped:false});const line=new THREE.Line(geometry,material);if(dashed)line.computeLineDistances();return line;
  }
  tube(points,color,radius,opacity=1,offset=.1){
    if(!Array.isArray(points)||points.length<2)return null;if(points.some(p=>!validPoint(p))){const group=new THREE.Group();let run=[];for(const point of[...points,null]){if(validPoint(point))run.push(point);else{if(run.length>1){const part=this.tube(run,color,radius,opacity,offset);if(part)group.add(part);}run=[];}}return group.children.length?group:null;}
    const vectors=points.map(p=>this.point(p,offset)).filter((p,i,list)=>i===0||p.distanceToSquared(list[i-1])>1e-8);if(vectors.length<2)return null;
    const curve=new RecordedPolyline(vectors),geometry=new THREE.TubeGeometry(curve,Math.min(900,Math.max(24,vectors.length*4)),radius,6,false);return new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity,depthWrite:opacity===1,toneMapped:false}));
  }
  plannedTrajectory(points,radius=.055,offset=.11){
    if(!Array.isArray(points)||!points.length)return null;const group=new THREE.Group(),denominator=Math.max(1,points.length-1),runs=[];let run=[];
    for(let index=0;index<=points.length;index++){const point=points[index];if(validPoint(point))run.push({point:this.point(point,offset),progress:index/denominator});else{if(run.length)runs.push(run);run=[];}}
    const colorAt=progress=>new THREE.Color().setRGB(clamp(progress,0,1),1,0,THREE.SRGBColorSpace);
    for(const entries of runs){const vectors=[],progress=[];for(const entry of entries){if(vectors.length&&entry.point.distanceToSquared(vectors.at(-1))<=1e-8){if(vectors.length>1)progress[progress.length-1]=entry.progress;continue;}vectors.push(entry.point);progress.push(entry.progress);}
      if(vectors.length===1){const dot=new THREE.Mesh(new THREE.SphereGeometry(radius*1.65,10,6),new THREE.MeshBasicMaterial({color:colorAt(points.length===1?0:progress[0]),toneMapped:false,fog:false}));dot.position.copy(vectors[0]);group.add(dot);continue;}
      const curve=new TemporalPolyline(vectors,progress),count=Math.min(1200,Math.max(40,vectors.length*6)),radial=6,geometry=new THREE.TubeGeometry(curve,count,radius,radial,false),colors=[];
      for(let ring=0;ring<=count;ring++){const color=colorAt(curve.progressAt(ring/count));for(let side=0;side<=radial;side++)colors.push(color.r,color.g,color.b);}
      geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));group.add(new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0xffffff,vertexColors:true,toneMapped:false,fog:false})));
    }
    group.userData.trajectoryColorOrder='original_sample_index';return group.children.length?group:null;
  }
  stripe(points,width,color,dashed){
    if(!Array.isArray(points)||points.length<2)return null;const vertices=[],indices=[];let arc=0;const s=this.owner;
    for(let i=0;i<points.length-1;i++){
      const a=points[i],b=points[i+1];if(!validPoint(a)||!validPoint(b))continue;const dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(length<1e-6)continue;const nx=-dy/length*width/2,ny=dx/length*width/2;let local=0;
      while(local<length-1e-6){const period=5.8,phase=(arc+local)%period,on=!dashed||phase<2.7,step=dashed?Math.min(length-local,(phase<2.7?2.7:period)-phase):length-local;
        if(on&&step>1e-6){const start=local/length,end=(local+step)/length,az=number(a[2],s.ground),bz=number(b[2],s.ground),p0=[a[0]+dx*start,a[1]+dy*start,az+(bz-az)*start],p1=[a[0]+dx*end,a[1]+dy*end,az+(bz-az)*end],index=vertices.length/3;
          for(const p of[[p0[0]+nx,p0[1]+ny,p0[2]],[p0[0]-nx,p0[1]-ny,p0[2]],[p1[0]-nx,p1[1]-ny,p1[2]],[p1[0]+nx,p1[1]+ny,p1[2]]])vertices.push(...this.point(p,.025).toArray());indices.push(index,index+1,index+2,index,index+2,index+3);}
        local+=Math.max(step,1e-4);
      }arc+=length;
    }
    if(!vertices.length)return null;const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.setIndex(indices);geometry.computeVertexNormals();return new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide,transparent:true,opacity:.88,depthWrite:false,toneMapped:false}));
  }
  label(group,text,point,color='#b9d5e6',scale=1,options={}){
    const canvas=document.createElement('canvas'),c=canvas.getContext('2d'),lines=String(text).split('\n').slice(0,2);c.font='600 29px "Segoe UI","Microsoft YaHei",sans-serif';canvas.width=Math.ceil(Math.max(160,...lines.map(line=>c.measureText(line).width+42)));canvas.height=lines.length===2?100:64;
    c.font='600 29px "Segoe UI","Microsoft YaHei",sans-serif';c.textAlign='center';c.textBaseline='middle';c.fillStyle='#0c1c2bf2';c.beginPath();c.roundRect(2,2,canvas.width-4,canvas.height-4,10);c.fill();c.strokeStyle=color;c.globalAlpha=.65;c.lineWidth=2;c.stroke();c.globalAlpha=1;c.fillStyle=color;
    lines.forEach((line,index)=>{c.font=`${index?'500 23':'600 29'}px "Segoe UI","Microsoft YaHei",sans-serif`;c.fillText(line,canvas.width/2,lines.length===2?31+index*39:canvas.height/2);});
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false,depthWrite:false,toneMapped:false})),anchor=this.point(point);sprite.position.copy(anchor);sprite.renderOrder=15;
    sprite.userData.labelInfo={anchor,aspect:canvas.width/canvas.height,pixelHeight:options.pixelHeight||(lines.length===2?45:27),priority:options.priority||0,allowShift:options.allowShift!==false};
    const leader=new THREE.Line(new THREE.BufferGeometry().setFromPoints([anchor,anchor]),new THREE.LineBasicMaterial({color,transparent:true,opacity:.65,depthTest:false,depthWrite:false,toneMapped:false}));leader.visible=false;leader.renderOrder=14;sprite.userData.labelInfo.leader=leader;group.add(leader,sprite);return sprite;
  }
}
