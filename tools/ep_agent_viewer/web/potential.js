// Same archived lane potential, two views. No display-only source or target.
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const COLORS={text:'#111827',muted:'#6b7280',paper:'#ffffff',road:'#9ca3af',mark:'#ffffff'};
function range(z){let lo=Infinity,hi=-Infinity;for(const row of z||[])for(const v of row)if(finite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}return [lo,hi];}
function scatterLine(points,color,width=1,dash='solid',name='道路边界'){
  return {type:'scatter',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),
    line:{color,width,dash},name,hoverinfo:'skip',showlegend:false};
}
export class PotentialView {
  constructor(container){
    this.container=container;this.mode='top';this.quantity='raw';this.data=null;this.frame=null;this.sequence=0;
    this.observer=new ResizeObserver(()=>{if(this.container.data)window.Plotly?.Plots.resize(this.container);});
    this.observer.observe(container);
  }
  setData(data,frame){this.data=data;this.frame=frame;this.render();}
  setMode(mode){this.mode=mode;this.render();}
  setQuantity(quantity){this.quantity=quantity;this.render();}
  resize(){if(this.container.data)window.Plotly?.Plots.resize(this.container);}
  render(){
    const Plotly=window.Plotly,data=this.data,frame=this.frame;
    if(!Plotly)return;
    const sequence=++this.sequence;
    if(!data?.available){
      Plotly.purge(this.container);this.container.replaceChildren();
      const empty=document.createElement('div');empty.className='field-empty';
      empty.textContent=data?.reason||'等待当前源帧的归档势场';this.container.append(empty);return;
    }
    const z=this.quantity==='weighted'?(data.weighted_values||data.values):data.values;
    const [minimum,maximum]=range(z);if(!finite(minimum)){this.setData({available:false,reason:'当前展示窗口没有有效势场采样'},frame);return;}
    const is3d=this.mode==='surface',span=maximum-minimum;
    const cmin=span>1e-10?minimum:minimum-.5,cmax=span>1e-10?maximum:maximum+.5;
    const quantity=this.quantity==='weighted'?'MPC 加权势场项':'原始势能 U';
    const colorbar={title:{text:quantity,side:'right',font:{size:11}},thickness:13,len:.82,
      tickfont:{size:10,color:COLORS.text},outlinewidth:0,tickformat:'.3~g'};
    const common={x:data.xs,y:data.ys,z,colorscale:'Viridis',connectgaps:false,colorbar,
      hovertemplate:'x=%{x:.2f} m<br>y=%{y:.2f} m<br>U=%{z:.4f}<extra></extra>',name:quantity};
    const traces=[is3d?{...common,type:'surface',cmin,cmax,
      lighting:{ambient:.7,diffuse:.7,specular:.15,roughness:.8},
      contours:{z:{show:true,usecolormap:true,highlightcolor:'#ffffff',project:{z:true}}}}:
      {...common,type:'contour',zmin:cmin,zmax:cmax,autocontour:true,ncontours:30,
        contours:{coloring:'heatmap',showlabels:false},line:{width:.5,color:'rgba(240,248,255,.36)'},
        hoverongaps:false}];
    const visual=frame?.visualization||{},road=visual.render_geometry?.road_surface;
    const polygons=road?.polygons||data.domain_polygons||[];
    if(!is3d){
      for(const polygon of polygons){for(const ring of [polygon.outer,...(polygon.holes||[])]){
        if(ring?.length)traces.push(scatterLine([...ring,ring[0]],'#465970',1.5));
      }}
      for(const marking of visual.render_geometry?.markings||[]){
        if(marking.points?.length)traces.push(scatterLine(marking.points,'rgba(255,255,255,.8)',1,marking.kind==='broken'?'dash':'solid','车道标线'));
      }
      const effective=new Set((data.sources||[]).filter(s=>s.effective).map(s=>s.id));
      for(const object of visual.objects||[]){
        if(!effective.has(object.id))continue;
        const p=object.center,footprint=object.footprint;
        if(footprint?.length)traces.push(scatterLine([...footprint,footprint[0]],'#ffffff',1.8,'solid',object.id));
        if(p)traces.push({type:'scatter',mode:'markers+text',x:[p[0]],y:[p[1]],text:[String(object.id).replace(/^carla_/, '')],
          textposition:'top right',textfont:{color:'#f6f9fd',size:10},marker:{size:6,color:object.attention_rank?'#d6b1ff':'#ffffff',line:{width:1,color:'#24354a'}},
          showlegend:false,name:'排斥源',hovertemplate:'排斥源 %{text}<extra></extra>'});
      }
      const ego=frame?.ego?.position;
      if(ego)traces.push({type:'scatter',mode:'markers+text',x:[ego[0]],y:[ego[1]],text:['E'],textposition:'bottom center',
        textfont:{color:'#f5f7fb',size:12},marker:{size:11,symbol:'square',color:'#ef5d62',line:{width:1,color:'#ffffff'}},name:'自车中心',showlegend:false});
      if(data.goal)traces.push({type:'scatter',mode:'markers+text',x:[data.goal[0]],y:[data.goal[1]],text:['G'],textposition:'top right',
        marker:{size:16,symbol:'star',color:'#ffe054',line:{width:1.5,color:'#26354b'}},textfont:{color:'#f6f9fd'},name:'活动局部目标',showlegend:false});
    } else {
      for(const source of data.sources||[]){
        if(!source.effective||!source.position)continue;
        const p=source.position;
        let ix=0,iy=0;for(let i=1;i<data.xs.length;i++)if(Math.abs(data.xs[i]-p[0])<Math.abs(data.xs[ix]-p[0]))ix=i;
        for(let i=1;i<data.ys.length;i++)if(Math.abs(data.ys[i]-p[1])<Math.abs(data.ys[iy]-p[1]))iy=i;
        const height=z[iy]?.[ix];if(!finite(height))continue;
        traces.push({type:'scatter3d',mode:'markers+text',x:[p[0]],y:[p[1]],z:[height],text:[source.id],textposition:'top center',
          textfont:{color:'#e3edf8',size:11},marker:{size:3,color:'#ffffff'},showlegend:false,name:'排斥源'});
      }
      if(data.goal){
        const p=data.goal;
        let ix=0,iy=0;for(let i=1;i<data.xs.length;i++)if(Math.abs(data.xs[i]-p[0])<Math.abs(data.xs[ix]-p[0]))ix=i;
        for(let i=1;i<data.ys.length;i++)if(Math.abs(data.ys[i]-p[1])<Math.abs(data.ys[iy]-p[1]))iy=i;
        const height=z[iy]?.[ix];
        if(finite(height))traces.push({type:'scatter3d',mode:'markers+text',x:[p[0]],y:[p[1]],z:[height],text:['G · 活动局部目标'],
          textposition:'top center',textfont:{color:'#35455a',size:11},marker:{size:6,color:'#ffe054',symbol:'diamond',line:{color:'#26354b',width:1}},showlegend:false,name:'活动局部目标'});
      }
    }
    const axis={showgrid:true,gridcolor:'#dce3eb',zeroline:false,tickfont:{color:COLORS.muted,size:10},title:{font:{size:11,color:COLORS.muted}}};
    const b=data.bounds,dx=b[2]-b[0],dy=b[3]-b[1],longest=Math.max(dx,dy);
    // Rotate a tall 2D window by 90 degrees; preserve distances and every
    // original sample, unlike stretching the world to fill the dashboard.
    const rotated=!is3d&&dy>1.3*dx;
    if(rotated){
      traces[0].x=data.ys;traces[0].y=data.xs.slice().reverse().map(v=>-v);
      traces[0].z=data.xs.map((_,ix)=>z.map(row=>row[ix])).reverse();
      traces[0].hovertemplate='y=%{x:.2f} m<br>−x=%{y:.2f} m<br>U=%{z:.4f}<extra></extra>';
      for(const trace of traces.slice(1)){const oldX=trace.x;trace.x=trace.y;trace.y=oldX.map(v=>-v);}
    }
    const plotBounds=rotated?[b[1],-b[2],b[3],-b[0]]:b;
    const surfaceAxis={...axis,tickfont:{color:'#52657b',size:10},title:{font:{size:11,color:'#3f526a'}}};
    const layout={paper_bgcolor:COLORS.paper,plot_bgcolor:'#e9edf3',font:{family:'Segoe UI, Microsoft YaHei, sans-serif',color:COLORS.text},
      margin:{l:53,r:35,t:20,b:48},autosize:true,showlegend:false,
      uirevision:`${frame?.episode_id||''}:${this.mode}:${this.quantity}`,
      xaxis:{...axis,title:{text:rotated?'y / m（世界坐标旋转90°）':'x / m'},range:[plotBounds[0],plotBounds[2]],constrain:'domain'},
      yaxis:{...axis,title:{text:rotated?'−x / m':'y / m'},range:[plotBounds[1],plotBounds[3]],scaleanchor:'x',scaleratio:1,constrain:'domain'},
      scene:{bgcolor:'#e9edf3',xaxis:{...surfaceAxis,title:{text:'x / m',font:{color:'#3f526a'}}},yaxis:{...surfaceAxis,title:{text:'y / m',font:{color:'#3f526a'}}},
        zaxis:{...surfaceAxis,title:{text:quantity,font:{color:'#3f526a'}}},aspectmode:'manual',aspectratio:{x:1.6*dx/longest,y:1.6*dy/longest,z:.65},
        camera:{eye:{x:1.3,y:-1.6,z:1.1}},dragmode:'orbit'},
      annotations:span<1e-10?[{text:'当前场域为常值：'+minimum.toPrecision(3)+'（未添加示意峰谷）',xref:'paper',yref:'paper',x:.01,y:1.04,showarrow:false,font:{size:11,color:COLORS.muted},xanchor:'left'}]:[]};
    const config={responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
      modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'png',filename:`EP-Driver_field_${frame?.index??0}`,scale:2}};
    // Clear the placeholder before the first Plotly creation.
    if(!this.container.data)this.container.replaceChildren();
    Plotly.react(this.container,traces,layout,config).then(()=>{if(sequence===this.sequence)this.resize();}).catch(error=>{
      if(sequence===this.sequence){this.container.textContent='势场绘制失败：'+error.message;}
    });
  }
}
