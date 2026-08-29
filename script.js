// ══════════════════════════════════════════════════
// LAYER REGISTRY
// ══════════════════════════════════════════════════
const LR=[
  {t:'Linear',c:'#3b82f6',d:'Fully connected',df:{in_features:128,out_features:64,bias:true},g:'Core'},
  {t:'Bilinear',c:'#2563eb',d:'Bilinear transform',df:{in1_features:64,in2_features:64,out_features:64},g:'Core'},
  {t:'Conv1d',c:'#38bdf8',d:'1D convolution',df:{in_channels:1,out_channels:32,kernel_size:3,stride:1,padding:0},g:'Convolution'},
  {t:'Conv2d',c:'#10b981',d:'2D convolution',df:{in_channels:3,out_channels:32,kernel_size:3,stride:1,padding:1},g:'Convolution'},
  {t:'Conv3d',c:'#059669',d:'3D convolution',df:{in_channels:1,out_channels:16,kernel_size:3,stride:1,padding:0},g:'Convolution'},
  {t:'ConvTranspose2d',c:'#047857',d:'Transposed conv (upsample)',df:{in_channels:32,out_channels:16,kernel_size:2,stride:2},g:'Convolution'},
  {t:'DepthwiseSepConv',c:'#065f46',d:'Depthwise separable',df:{in_channels:32,out_channels:64,kernel_size:3,padding:1},g:'Convolution'},
  {t:'ReLU',c:'#f97316',d:'Rectified Linear Unit',df:{},g:'Activation'},
  {t:'LeakyReLU',c:'#ea580c',d:'Leaky ReLU',df:{negative_slope:0.01},g:'Activation'},
  {t:'GELU',c:'#f59e0b',d:'Gaussian Error Linear',df:{},g:'Activation'},
  {t:'SiLU',c:'#d97706',d:'Swish / SiLU',df:{},g:'Activation'},
  {t:'Mish',c:'#b45309',d:'Mish self-regularizing',df:{},g:'Activation'},
  {t:'Sigmoid',c:'#eab308',d:'Sigmoid',df:{},g:'Activation'},
  {t:'Tanh',c:'#ca8a04',d:'Hyperbolic tangent',df:{},g:'Activation'},
  {t:'Softmax',c:'#06b6d4',d:'Softmax output layer',df:{dim:1},g:'Activation'},
  {t:'LogSoftmax',c:'#0891b2',d:'Log softmax',df:{dim:1},g:'Activation'},
  {t:'Dropout',c:'#ef4444',d:'Dropout regularization',df:{p:0.5},g:'Regularization'},
  {t:'Dropout2d',c:'#dc2626',d:'Spatial channel dropout',df:{p:0.5},g:'Regularization'},
  {t:'AlphaDropout',c:'#b91c1c',d:'Alpha dropout (SELU nets)',df:{p:0.5},g:'Regularization'},
  {t:'BatchNorm1d',c:'#10b981',d:'1D batch normalization',df:{num_features:64,momentum:0.1},g:'Normalization'},
  {t:'BatchNorm2d',c:'#059669',d:'2D batch normalization',df:{num_features:32,momentum:0.1},g:'Normalization'},
  {t:'LayerNorm',c:'#047857',d:'Layer normalization',df:{normalized_shape:64},g:'Normalization'},
  {t:'GroupNorm',c:'#065f46',d:'Group normalization',df:{num_groups:8,num_channels:64},g:'Normalization'},
  {t:'InstanceNorm2d',c:'#064e3b',d:'Instance normalization',df:{num_features:32},g:'Normalization'},
  {t:'RMSNorm',c:'#14532d',d:'RMS normalization (LLMs)',df:{normalized_shape:64},g:'Normalization'},
  {t:'MaxPool2d',c:'#f59e0b',d:'2D max pooling',df:{kernel_size:2,stride:2},g:'Pooling'},
  {t:'AvgPool2d',c:'#d97706',d:'2D average pooling',df:{kernel_size:2,stride:2},g:'Pooling'},
  {t:'AdaptiveAvgPool2d',c:'#b45309',d:'Global average pool',df:{output_size:1},g:'Pooling'},
  {t:'MaxPool1d',c:'#92400e',d:'1D max pooling',df:{kernel_size:2,stride:2},g:'Pooling'},
  {t:'Flatten',c:'#8b5cf6',d:'Flatten to 1D',df:{},g:'Shape'},
  {t:'Unflatten',c:'#7c3aed',d:'Reshape from 1D',df:{dim:1,sizes:'[8,8]'},g:'Shape'},
  {t:'Permute',c:'#6d28d9',d:'Permute tensor dims',df:{dims:'[0,2,1]'},g:'Shape'},
  {t:'Embedding',c:'#ec4899',d:'Embedding lookup table',df:{num_embeddings:10000,embedding_dim:128},g:'NLP'},
  {t:'MultiheadAttention',c:'#db2777',d:'Multi-head self-attention',df:{embed_dim:128,num_heads:8,dropout:0.1},g:'NLP'},
  {t:'TransformerEncoderLayer',c:'#be185d',d:'Pre-norm transformer block',df:{d_model:128,nhead:8,dim_feedforward:512,dropout:0.1},g:'NLP'},
  {t:'TransformerDecoderLayer',c:'#9d174d',d:'Transformer decoder block',df:{d_model:128,nhead:8,dim_feedforward:512,dropout:0.1},g:'NLP'},
  {t:'LSTM',c:'#06b6d4',d:'Long Short-Term Memory',df:{input_size:128,hidden_size:256,num_layers:1,batch_first:true,dropout:0},g:'Recurrent'},
  {t:'GRU',c:'#0891b2',d:'Gated Recurrent Unit',df:{input_size:128,hidden_size:256,num_layers:1,batch_first:true},g:'Recurrent'},
  {t:'RNN',c:'#0e7490',d:'Vanilla RNN',df:{input_size:128,hidden_size:128,num_layers:1,batch_first:true},g:'Recurrent'},
  {t:'Conv+BN+Act',c:'#10b981',d:'Fused conv block',df:{in_channels:32,out_channels:64,kernel_size:3,padding:1,activation:'ReLU'},g:'Blocks'},
  {t:'Linear+BN+Act',c:'#3b82f6',d:'Fused linear block',df:{in_features:128,out_features:64,activation:'ReLU'},g:'Blocks'},
  {t:'SEBlock',c:'#f97316',d:'Squeeze-Excitation block',df:{channels:64,reduction:16},g:'Blocks'},
  {t:'PositionalEncoding',c:'#8b5cf6',d:'Sinusoidal pos encoding',df:{d_model:128,max_len:512,dropout:0.1},g:'Blocks'},
  {t:'ResidualBlock',c:'#10b981',d:'Residual block (built-in skip)',df:{channels:64,kernel_size:3},g:'Blocks'},
];

const PM={
  in_features:{l:'In features',tp:'n',mn:1,mx:65536},
  out_features:{l:'Out features',tp:'n',mn:1,mx:65536},
  in1_features:{l:'In1 features',tp:'n',mn:1,mx:65536},
  in2_features:{l:'In2 features',tp:'n',mn:1,mx:65536},
  in_channels:{l:'In channels',tp:'n',mn:1,mx:4096},
  out_channels:{l:'Out channels',tp:'n',mn:1,mx:4096},
  channels:{l:'Channels',tp:'n',mn:1,mx:4096},
  kernel_size:{l:'Kernel size',tp:'n',mn:1,mx:15},
  stride:{l:'Stride',tp:'n',mn:1,mx:8},
  padding:{l:'Padding',tp:'n',mn:0,mx:8},
  p:{l:'Dropout prob',tp:'r',mn:0,mx:0.99,st:0.01},
  dropout:{l:'Dropout',tp:'r',mn:0,mx:0.99,st:0.01},
  negative_slope:{l:'Neg slope',tp:'r',mn:0,mx:0.5,st:0.01},
  num_features:{l:'Num features',tp:'n',mn:1,mx:65536},
  num_groups:{l:'Num groups',tp:'n',mn:1,mx:128},
  num_channels:{l:'Num channels',tp:'n',mn:1,mx:65536},
  normalized_shape:{l:'Norm shape',tp:'n',mn:1,mx:65536},
  num_embeddings:{l:'Vocab size',tp:'n',mn:1,mx:2000000},
  embedding_dim:{l:'Embedding dim',tp:'n',mn:1,mx:4096},
  embed_dim:{l:'Embed dim',tp:'n',mn:1,mx:4096},
  num_heads:{l:'Num heads',tp:'n',mn:1,mx:128},
  d_model:{l:'Model dim',tp:'n',mn:1,mx:4096},
  nhead:{l:'Num heads',tp:'n',mn:1,mx:128},
  dim_feedforward:{l:'FFN dim',tp:'n',mn:1,mx:65536},
  input_size:{l:'Input size',tp:'n',mn:1,mx:65536},
  hidden_size:{l:'Hidden size',tp:'n',mn:1,mx:65536},
  num_layers:{l:'Num layers',tp:'n',mn:1,mx:64},
  dim:{l:'Softmax dim',tp:'n',mn:0,mx:4},
  output_size:{l:'Output size',tp:'n',mn:1,mx:2048},
  reduction:{l:'Reduction ratio',tp:'n',mn:2,mx:64},
  max_len:{l:'Max seq length',tp:'n',mn:8,mx:32768},
  momentum:{l:'Momentum',tp:'r',mn:0.001,mx:0.99,st:0.001},
  bias:{l:'Use bias',tp:'b'},
  batch_first:{l:'Batch first',tp:'b'},
  sizes:{l:'Target sizes (JSON)',tp:'tx'},
  dims:{l:'Permute dims (JSON)',tp:'tx'},
  activation:{l:'Activation',tp:'tx'},
};

const BASELINES=[
  {name:'ResNet Residual Block',tag:'CNN',c:'#10b981',
   layers:[
     {t:'Conv2d',p:{in_channels:64,out_channels:64,kernel_size:3,stride:1,padding:1}},
     {t:'BatchNorm2d',p:{num_features:64}},{t:'ReLU',p:{}},
     {t:'Conv2d',p:{in_channels:64,out_channels:64,kernel_size:3,stride:1,padding:1}},
     {t:'BatchNorm2d',p:{num_features:64}},
   ],sk:[{f:0,t:4}],desc:'He et al. 2016 — basic residual block with identity skip'},
  {name:'Bottleneck Block (ResNet-50)',tag:'CNN',c:'#f97316',
   layers:[
     {t:'Conv2d',p:{in_channels:256,out_channels:64,kernel_size:1,stride:1,padding:0}},
     {t:'BatchNorm2d',p:{num_features:64}},{t:'ReLU',p:{}},
     {t:'Conv2d',p:{in_channels:64,out_channels:64,kernel_size:3,stride:1,padding:1}},
     {t:'BatchNorm2d',p:{num_features:64}},{t:'ReLU',p:{}},
     {t:'Conv2d',p:{in_channels:64,out_channels:256,kernel_size:1,stride:1,padding:0}},
     {t:'BatchNorm2d',p:{num_features:256}},
   ],sk:[{f:0,t:7}],desc:'1×1→3×3→1×1 bottleneck with projection skip'},
  {name:'Pre-Norm Transformer (GPT-style)',tag:'NLP',c:'#8b5cf6',
   layers:[
     {t:'LayerNorm',p:{normalized_shape:128}},
     {t:'MultiheadAttention',p:{embed_dim:128,num_heads:8,dropout:0.1}},
     {t:'Dropout',p:{p:0.1}},
     {t:'LayerNorm',p:{normalized_shape:128}},
     {t:'Linear',p:{in_features:128,out_features:512,bias:true}},
     {t:'GELU',p:{}},
     {t:'Dropout',p:{p:0.1}},
     {t:'Linear',p:{in_features:512,out_features:128,bias:true}},
     {t:'Dropout',p:{p:0.1}},
   ],sk:[{f:0,t:3},{f:3,t:8}],desc:'Pre-LayerNorm GPT/ViT block with two residual paths'},
  {name:'MobileNetV2 Inverted Residual',tag:'Efficient',c:'#06b6d4',
   layers:[
     {t:'Conv2d',p:{in_channels:32,out_channels:192,kernel_size:1,stride:1,padding:0}},
     {t:'BatchNorm2d',p:{num_features:192}},{t:'ReLU',p:{}},
     {t:'DepthwiseSepConv',p:{in_channels:192,out_channels:192,kernel_size:3,padding:1}},
     {t:'BatchNorm2d',p:{num_features:192}},{t:'ReLU',p:{}},
     {t:'Conv2d',p:{in_channels:192,out_channels:32,kernel_size:1,stride:1,padding:0}},
     {t:'BatchNorm2d',p:{num_features:32}},
   ],sk:[{f:0,t:7}],desc:'Expansion→DW→Projection with linear bottleneck (stride=1)'},
  {name:'U-Net Skip Encoder Stage',tag:'Seg',c:'#f59e0b',
   layers:[
     {t:'Conv2d',p:{in_channels:1,out_channels:64,kernel_size:3,stride:1,padding:1}},
     {t:'ReLU',p:{}},
     {t:'Conv2d',p:{in_channels:64,out_channels:64,kernel_size:3,stride:1,padding:1}},
     {t:'ReLU',p:{}},
     {t:'MaxPool2d',p:{kernel_size:2,stride:2}},
     {t:'Conv2d',p:{in_channels:64,out_channels:128,kernel_size:3,stride:1,padding:1}},
     {t:'ReLU',p:{}},
   ],sk:[{f:2,t:6}],desc:'U-Net encoder with skip connection saved to decoder'},
  {name:'LSTM Text Classifier',tag:'NLP',c:'#ec4899',
   layers:[
     {t:'Embedding',p:{num_embeddings:10000,embedding_dim:128}},
     {t:'LSTM',p:{input_size:128,hidden_size:256,num_layers:2,batch_first:true,dropout:0.3}},
     {t:'Dropout',p:{p:0.5}},
     {t:'Linear',p:{in_features:256,out_features:64,bias:true}},
     {t:'ReLU',p:{}},
     {t:'Linear',p:{in_features:64,out_features:2,bias:true}},
     {t:'Softmax',p:{dim:1}},
   ],sk:[],desc:'Stacked LSTM with embedding for sentiment/classification'},
];

// ══════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════
let layers=[],skips=[],selIdx=null,nextId=1;
let skipMode=false,skipPend=null;
let saved=[],customLs=[];

// ══════════════════════════════════════════════════
// BUILD PALETTE
// ══════════════════════════════════════════════════
function buildPalette(){
  const all=[...LR,...customLs];
  const gs={};
  all.forEach(l=>{const g=l.g||'Custom';if(!gs[g])gs[g]=[];gs[g].push(l);});
  let h='';
  Object.entries(gs).forEach(([g,items])=>{
    h+=`<div class="glabel">${g}</div>`;
    h+=items.map(l=>`<div class="lt" onclick="addLayer('${l.t||l.type}')">
      <div class="dot" style="background:${l.c||l.color}"></div>
      <div><div class="ln">${l.t||l.type}</div><div class="ld">${l.d||l.desc}</div></div>
    </div>`).join('');
  });
  document.getElementById('lt-layers').innerHTML=h;
}

function buildBaselines(){
  const tc={'CNN':'#10b981','NLP':'#8b5cf6','Efficient':'#06b6d4','Seg':'#f59e0b'};
  document.getElementById('lt-baselines').innerHTML=BASELINES.map((b,i)=>`
    <div class="bl" onclick="loadBaseline(${i})">
      <div class="bn">${b.name}</div>
      <div class="bd">${b.desc}</div>
      <span class="btag" style="background:${tc[b.tag]||'#888'}22;color:${tc[b.tag]||'#888'};border:1px solid ${tc[b.tag]||'#888'}44">${b.tag}</span>
    </div>`).join('');
}

function loadBaseline(i){
  const b=BASELINES[i];
  layers=b.layers.map(l=>({id:nextId++,type:l.t,params:{...l.p}}));
  skips=JSON.parse(JSON.stringify(b.sk||[]));
  selIdx=0;
  document.getElementById('arch-name').value=b.name;
  render();
}

function setLTab(n,el){
  document.querySelectorAll('.ptab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  ['layers','baselines','custom','saved'].forEach(t=>document.getElementById('lt-'+t).style.display=t===n?'block':'none');
}

// ══════════════════════════════════════════════════
// CUSTOM LAYERS
// ══════════════════════════════════════════════════
function regCustom(){
  const n=document.getElementById('cl-n').value.trim()||'MyLayer';
  const c=document.getElementById('cl-c').value||'#8b5cf6';
  const d=document.getElementById('cl-d').value.trim()||'Custom op';
  const ps=document.getElementById('cl-p').value.trim();
  const fwd=document.getElementById('cl-f').value.trim()||'return x';
  const df={};
  if(ps) ps.split(',').forEach(p=>{const[k,tp]=(p.trim()).split(':');if(k)df[k.trim()]=tp==='range'?0.5:64;});
  const ex=customLs.findIndex(c=>c.t===n);
  const def={t:n,type:n,c,color:c,d,desc:d,df,defaults:df,g:'Custom',forwardCode:fwd};
  if(ex>=0)customLs[ex]=def;else customLs.push(def);
  buildPalette();renderCustomList();
}
function renderCustomList(){
  document.getElementById('cl-list').innerHTML=customLs.length?
    '<div class="slabel" style="margin-top:7px">Registered</div>'+
    customLs.map((c,i)=>`<div class="lt">
      <div class="dot" style="background:${c.c}"></div>
      <div style="flex:1"><div class="ln">${c.t}</div><div class="ld">${c.d}</div></div>
      <button class="btn sm" onclick="customLs.splice(${i},1);buildPalette();renderCustomList()">×</button>
    </div>`).join(''):'';
}

// ══════════════════════════════════════════════════
// LAYER MANAGEMENT
// ══════════════════════════════════════════════════
function getDef(t){return LR.find(l=>l.t===t)||customLs.find(l=>l.t===t)||{t,c:'#666',d:'',df:{},g:'Custom'};}
function getColor(t){const d=getDef(t);return d.c||d.color||'#666';}

function addLayer(type){
  const d=getDef(type);
  layers.push({id:nextId++,type,params:{...(d.df||d.defaults||{})}});
  selIdx=layers.length-1;
  render();
}
function removeLayer(idx){
  layers.splice(idx,1);
  skips=skips.filter(s=>s.f!==idx&&s.t!==idx)
    .map(s=>({...s,f:s.f>idx?s.f-1:s.f,t:s.t>idx?s.t-1:s.t}));
  if(selIdx>=layers.length)selIdx=layers.length-1;
  render();
}
function moveLayer(idx,dir){
  const ni=idx+dir;if(ni<0||ni>=layers.length)return;
  [layers[idx],layers[ni]]=[layers[ni],layers[idx]];
  skips=skips.map(s=>{
    let{f,t}=s;
    if(f===idx)f=ni;else if(f===ni)f=idx;
    if(t===idx)t=ni;else if(t===ni)t=idx;
    return{...s,f,t};
  });
  selIdx=ni;render();
}
function selectLayer(idx){
  if(skipMode){
    if(skipPend===null){skipPend=idx;document.getElementById('skip-btn').textContent='⤳ Pick dst';}
    else{
      if(skipPend!==idx&&idx>skipPend&&!skips.some(s=>s.f===skipPend&&s.t===idx))
        skips.push({f:skipPend,t:idx});
      skipPend=null;skipMode=false;
      const b=document.getElementById('skip-btn');
      b.textContent='⤳ Skip';b.style.borderColor='';b.style.color='';
    }
  }else{selIdx=idx;}
  render();
}
function removeSkip(i){skips.splice(i,1);render();}
function toggleSkip(){
  skipMode=!skipMode;skipPend=null;
  const b=document.getElementById('skip-btn');
  b.style.borderColor=skipMode?'var(--cy)':'';
  b.style.color=skipMode?'var(--cy)':'';
  b.textContent=skipMode?'⤳ Pick src':'⤳ Skip';
}
function setParam(idx,key,val){
  layers[idx].params[key]=val;
  renderCanvas();renderValidate();renderCode();renderStats();
}

// ══════════════════════════════════════════════════
// SHAPE INFERENCE
// ══════════════════════════════════════════════════
function inferShape(l){
  const p=l.params;
  switch(l.type){
    case 'Linear':case 'Linear+BN+Act':return`[B,${p.out_features||'?'}]`;
    case 'Bilinear':return`[B,${p.out_features||'?'}]`;
    case 'Conv1d':return`[B,${p.out_channels||'?'},L']`;
    case 'Conv2d':case 'Conv+BN+Act':case 'DepthwiseSepConv':return`[B,${p.out_channels||'?'},H',W']`;
    case 'Conv3d':return`[B,${p.out_channels||'?'},D',H',W']`;
    case 'ConvTranspose2d':return`[B,${p.out_channels||'?'},H↑,W↑]`;
    case 'MaxPool2d':case 'AvgPool2d':return`[B,C,H/${p.kernel_size||2},W/${p.kernel_size||2}]`;
    case 'MaxPool1d':return`[B,C,L/${p.kernel_size||2}]`;
    case 'AdaptiveAvgPool2d':return`[B,C,${p.output_size||1},${p.output_size||1}]`;
    case 'Flatten':return`[B,C×H×W]`;
    case 'Embedding':return`[B,S,${p.embedding_dim||'?'}]`;
    case 'MultiheadAttention':return`[B,S,${p.embed_dim||'?'}]`;
    case 'TransformerEncoderLayer':case 'TransformerDecoderLayer':return`[B,S,${p.d_model||'?'}]`;
    case 'PositionalEncoding':return`[B,S,${p.d_model||'?'}]`;
    case 'LSTM':case 'GRU':case 'RNN':return`[B,S,${p.hidden_size||'?'}]`;
    case 'BatchNorm1d':return`[B,${p.num_features||'?'}]`;
    case 'BatchNorm2d':case 'InstanceNorm2d':return`[B,${p.num_features||'?'},H,W]`;
    case 'LayerNorm':case 'RMSNorm':return`[…,${p.normalized_shape||'?'}]`;
    case 'GroupNorm':return`[B,${p.num_channels||'?'},H,W]`;
    case 'SEBlock':case 'ResidualBlock':return`[B,${p.channels||'?'},H,W]`;
    default:return null;
  }
}

// ══════════════════════════════════════════════════
// RENDER CANVAS
// ══════════════════════════════════════════════════
function renderCanvas(){
  const wrap=document.getElementById('cvwrap');
  if(!layers.length){
    wrap.innerHTML=`<div class="empty"><div class="eyebrow">Model Workbench</div><div class="ei">◈</div><p>Build and inspect a neural network architecture in the workspace.</p><p class="sub">Add a layer or load a baseline to begin</p></div>`;
    return;
  }
  const issues=validate();
  const eM={},wM={};
  issues.forEach(i=>{if(i.layer!=null){if(i.sev==='error')eM[i.layer]=true;if(i.sev==='warn')wM[i.layer]=true;}});
  const sSrc=new Set(skips.map(s=>s.f)),sDst=new Set(skips.map(s=>s.t));
  let h='';
  layers.forEach((l,i)=>{
    const col=getColor(l.type);
    const isSel=(selIdx===i&&!skipMode)||(skipMode&&skipPend===i);
    const chips=Object.entries(l.params).filter(([k,v])=>v!=null&&v!=='').slice(0,5)
      .map(([k,v])=>`<div class="chip">${k}=${typeof v==='boolean'?(v?'T':'F'):v}</div>`).join('');
    const mySkips=skips.filter(s=>s.f===i);
    const sb2=mySkips.map(s=>`<span class="sk-lbl">→L${s.t+1}</span>`).join(' ');
    const shape=inferShape(l);
    const cls=['lc',isSel?'sel':'',eM[i]?'err':'',wM[i]&&!eM[i]?'wrn':'',
      sSrc.has(i)?'sksr':'',sDst.has(i)&&!sSrc.has(i)?'skds':''].filter(Boolean).join(' ');
    h+=`<div class="${cls}" onclick="selectLayer(${i})">
      <div class="ll">
        <div class="ldot" style="background:${col}"></div>
        <div class="lname">${l.type}</div>
        <div class="lidx">#${i+1}</div>
        ${sb2}
        <div class="cbtns">
          <button class="cb" onclick="event.stopPropagation();moveLayer(${i},-1)" title="Up">↑</button>
          <button class="cb" onclick="event.stopPropagation();moveLayer(${i},1)" title="Down">↓</button>
          <button class="cb del" onclick="event.stopPropagation();removeLayer(${i})">×</button>
        </div>
      </div>
      <div class="chips">${chips||'<span style="font-size:7px;color:var(--t3)">no params</span>'}</div>
      ${shape?`<div class="lshape">out → ${shape}</div>`:''}
    </div>`;
    if(i<layers.length-1){
      const through=skips.filter(s=>s.f<=i&&s.t>i);
      h+=`<div class="arr"><div class="al"></div><div class="at"></div>${through.length?`<div class="skb">${through.map(s=>`L${s.f+1}→L${s.t+1}`).join(' ')}</div>`:''}</div>`;
    }
  });
  wrap.innerHTML=h;
}

// ══════════════════════════════════════════════════
// INSPECTOR
// ══════════════════════════════════════════════════
function renderInspector(){
  const el=document.getElementById('rp-inspector');
  if(selIdx===null||selIdx>=layers.length){
    el.innerHTML=`<div style="color:var(--t3);font-size:8px;text-align:center;padding:18px 0">Select a layer to inspect</div>`;
    return;
  }
  const l=layers[selIdx];const col=getColor(l.type);
  const def=getDef(l.type);
  let h=`<div class="ih">
    <div style="width:9px;height:9px;border-radius:50%;background:${col};flex-shrink:0"></div>
    <div class="iname">${l.type}</div>
    <div class="ilyr">L${selIdx+1} / ${layers.length}</div>
  </div>`;
  if(def.forwardCode)h+=`<div class="f"><label>Forward stub</label><div class="cb2" style="font-size:7px;padding:7px;max-height:60px">${def.forwardCode}</div></div>`;
  const keys=Object.keys(l.params);
  if(!keys.length)h+=`<div style="color:var(--t3);font-size:8px">No configurable parameters</div>`;
  keys.forEach(k=>{
    const m=PM[k]||{l:k,tp:'n',mn:0,mx:9999999};const v=l.params[k];
    if(m.tp==='b')h+=`<div class="f"><label>${m.l} <input type="checkbox" ${v?'checked':''} onchange="setParam(${selIdx},'${k}',this.checked)" style="width:auto;margin-left:5px;accent-color:var(--acc)"></label></div>`;
    else if(m.tp==='r')h+=`<div class="f"><label>${m.l}<span class="rv" id="rv_${k}">${parseFloat(v).toFixed(2)}</span></label>
      <input type="range" min="${m.mn}" max="${m.mx}" step="${m.st||0.01}" value="${v}"
        oninput="document.getElementById('rv_${k}').textContent=parseFloat(this.value).toFixed(2);setParam(${selIdx},'${k}',parseFloat(this.value))">
    </div>`;
    else if(m.tp==='tx')h+=`<div class="f"><label>${m.l}</label><input type="text" value="${v}" oninput="setParam(${selIdx},'${k}',this.value)"></div>`;
    else h+=`<div class="f"><label>${m.l}</label><input type="number" min="${m.mn||0}" max="${m.mx||9999999}" value="${v}" oninput="setParam(${selIdx},'${k}',parseInt(this.value)||0)"></div>`;
  });
  const mySkips=skips.filter(s=>s.f===selIdx||s.t===selIdx);
  if(mySkips.length){
    h+=`<div class="sep"></div><div class="slabel">Skip / Residual Connections</div>`;
    h+=mySkips.map(s=>`<div class="vi in" style="justify-content:space-between;align-items:center">
      <span>L${s.f+1} (${layers[s.f]?.type}) → L${s.t+1} (${layers[s.t]?.type})</span>
      <button class="btn sm" onclick="removeSkip(${skips.indexOf(s)})">Remove</button></div>`).join('');
  }
  el.innerHTML=h;
}

// ══════════════════════════════════════════════════
// VALIDATION
// ══════════════════════════════════════════════════
function validate(){
  const issues=[];if(!layers.length)return issues;
  const learnable=['Linear','Linear+BN+Act','Bilinear','Conv1d','Conv2d','Conv3d','ConvTranspose2d','DepthwiseSepConv','Conv+BN+Act','SEBlock','ResidualBlock','Embedding','LSTM','GRU','RNN','MultiheadAttention','TransformerEncoderLayer','TransformerDecoderLayer','PositionalEncoding'];
  if(!layers.some(l=>learnable.includes(l.type)))issues.push({sev:'warn',layer:null,msg:'No learnable layers found.'});
  const acts=['ReLU','LeakyReLU','GELU','SiLU','Mish','Sigmoid','Tanh','Softmax','LogSoftmax'];
  layers.forEach((l,i)=>{
    if(i>0&&acts.includes(l.type)&&acts.includes(layers[i-1].type))issues.push({sev:'warn',layer:i,msg:'Two activations back-to-back — likely unintentional.'});
    if(l.type==='Linear'&&i>0){
      const p=layers[i-1];
      if(p.type==='Linear'&&p.params.out_features!==l.params.in_features)issues.push({sev:'error',layer:i,msg:`Dimension mismatch: L${i} out=${p.params.out_features} ≠ L${i+1} in=${l.params.in_features}`});
      if(p.type==='BatchNorm1d'&&p.params.num_features!==l.params.in_features)issues.push({sev:'error',layer:i,msg:`BN1d(${p.params.num_features}) ≠ Linear in(${l.params.in_features})`});
    }
    if(l.type==='BatchNorm1d'&&i>0){const p=layers[i-1];if(p.type==='Linear'&&p.params.out_features!==l.params.num_features)issues.push({sev:'error',layer:i,msg:`BN1d features=${l.params.num_features} ≠ prev out=${p.params.out_features}`});}
    if(l.type==='Conv2d'&&i>0){
      const p=layers[i-1];
      if(p.type==='Conv2d'&&p.params.out_channels!==l.params.in_channels)issues.push({sev:'error',layer:i,msg:`Conv2d: L${i} out_ch=${p.params.out_channels} ≠ L${i+1} in_ch=${l.params.in_channels}`});
      if(p.type==='BatchNorm2d'&&p.params.num_features!==l.params.in_channels)issues.push({sev:'error',layer:i,msg:`BN2d(${p.params.num_features}) ≠ Conv2d in_ch(${l.params.in_channels})`});
    }
    if(l.type==='BatchNorm2d'&&i>0){const p=layers[i-1];if(p.type==='Conv2d'&&p.params.out_channels!==l.params.num_features)issues.push({sev:'error',layer:i,msg:`BN2d features=${l.params.num_features} ≠ Conv2d out_ch=${p.params.out_channels}`});}
    if((l.type==='Dropout'||l.type==='Dropout2d')&&(l.params.p<=0||l.params.p>=1))issues.push({sev:'warn',layer:i,msg:`Dropout p=${l.params.p} unusual — typical: 0.1–0.5`});
    if(l.type==='Softmax'&&i<layers.length-1)issues.push({sev:'warn',layer:i,msg:'Softmax is typically the final layer.'});
    if(l.type==='Embedding'&&i>0)issues.push({sev:'warn',layer:i,msg:'Embedding is typically the first layer.'});
    if((l.type==='LSTM'||l.type==='GRU'||l.type==='RNN')&&i>0){const p=layers[i-1];if(p.type==='Embedding'&&p.params.embedding_dim!==l.params.input_size)issues.push({sev:'error',layer:i,msg:`${l.type} input_size=${l.params.input_size} ≠ Embedding dim=${p.params.embedding_dim}`});}
    if(l.type==='MultiheadAttention'&&(l.params.embed_dim||0)%(l.params.num_heads||1)!==0)issues.push({sev:'error',layer:i,msg:`embed_dim(${l.params.embed_dim}) not divisible by num_heads(${l.params.num_heads})`});
    if((l.type==='TransformerEncoderLayer'||l.type==='TransformerDecoderLayer')&&(l.params.d_model||0)%(l.params.nhead||1)!==0)issues.push({sev:'error',layer:i,msg:`d_model(${l.params.d_model}) not divisible by nhead(${l.params.nhead})`});
    if(l.type==='GroupNorm'&&(l.params.num_channels||0)%(l.params.num_groups||1)!==0)issues.push({sev:'error',layer:i,msg:`num_channels(${l.params.num_channels}) not divisible by num_groups(${l.params.num_groups})`});
  });
  let hasConv=false,hasFl=false;
  layers.forEach((l,i)=>{
    if(['Conv2d','Conv3d','MaxPool2d','AvgPool2d','AdaptiveAvgPool2d','SEBlock'].includes(l.type))hasConv=true;
    if(l.type==='Flatten')hasFl=true;
    if(l.type==='Linear'&&hasConv&&!hasFl)issues.push({sev:'error',layer:i,msg:'Linear after Conv2d requires a Flatten layer first.'});
  });
  skips.forEach(s=>{if(s.t<=s.f)issues.push({sev:'error',layer:s.f,msg:`Skip L${s.f+1}→L${s.t+1} is backward — skips must go forward.`});});
  return issues;
}

function renderValidate(){
  const el=document.getElementById('rp-validate');
  const issues=validate();
  const hasE=issues.some(i=>i.sev==='error'),hasW=issues.some(i=>i.sev==='warn');
  if(!layers.length)return el.innerHTML='<div style="color:var(--t3);font-size:8px">Add layers to validate.</div>';
  let h='';
  if(!hasE&&!hasW)h=`<div class="vi ok"><span>✓</span><span>Architecture is valid — no errors or warnings.</span></div>`;
  issues.forEach(iss=>{
    const cls=iss.sev==='error'?'er':iss.sev==='warn'?'wn':'in';
    const icon=iss.sev==='error'?'✗':iss.sev==='warn'?'⚠':'ℹ';
    const loc=iss.layer!=null?`<b>[L${iss.layer+1}·${layers[iss.layer]?.type}]</b> `:'';
    h+=`<div class="vi ${cls}"><span>${icon}</span><span>${loc}${iss.msg}</span></div>`;
  });
  el.innerHTML=h;
}

// ══════════════════════════════════════════════════
// LIVE TENSOR FLOW TRACE
// ══════════════════════════════════════════════════
function runTrace(){
  const n=+document.getElementById('tn').value||1;
  const c=+document.getElementById('tc').value||3;
  const h2=+document.getElementById('th').value||32;
  const w2=+document.getElementById('tw').value||32;
  let cur=[n,c,h2,w2];
  const rows=[];
  layers.forEach((l,i)=>{
    const prev=[...cur];let ok=true,err='';const p=l.params;
    try{
      switch(l.type){
        case 'Linear':case 'Linear+BN+Act':
          if(cur[cur.length-1]!==p.in_features){err=`need ${p.in_features} got ${cur[cur.length-1]}`;ok=false;}
          else cur[cur.length-1]=p.out_features;break;
        case 'Conv2d':case 'Conv+BN+Act':
          if(cur[1]!==p.in_channels){err=`need ${p.in_channels}ch got ${cur[1]}`;ok=false;}
          else{const ks=p.kernel_size||3,st=p.stride||1,pd=p.padding||0;cur=[cur[0],p.out_channels,Math.floor((cur[2]+2*pd-ks)/st+1),Math.floor((cur[3]+2*pd-ks)/st+1)];}break;
        case 'DepthwiseSepConv':
          {const ks=p.kernel_size||3,st=p.stride||1,pd=p.padding||0;cur=[cur[0],p.out_channels||cur[1],Math.floor((cur[2]+2*pd-ks)/st+1),Math.floor((cur[3]+2*pd-ks)/st+1)];}break;
        case 'ConvTranspose2d':cur=[cur[0],p.out_channels,cur[2]*(p.stride||2),cur[3]*(p.stride||2)];break;
        case 'MaxPool2d':case 'AvgPool2d':cur=[cur[0],cur[1],Math.floor(cur[2]/(p.kernel_size||2)),Math.floor(cur[3]/(p.kernel_size||2))];break;
        case 'AdaptiveAvgPool2d':cur=[cur[0],cur[1],p.output_size||1,p.output_size||1];break;
        case 'Flatten':cur=[cur[0],cur.slice(1).reduce((a,b)=>a*b,1)];break;
        case 'BatchNorm1d':if(cur[1]!==p.num_features){err=`need ${p.num_features} got ${cur[1]}`;ok=false;}break;
        case 'BatchNorm2d':case 'InstanceNorm2d':if(cur[1]!==p.num_features){err=`need ${p.num_features}ch got ${cur[1]}`;ok=false;}break;
        case 'SEBlock':case 'ResidualBlock':if(cur[1]!==p.channels){err=`need ${p.channels}ch got ${cur[1]}`;ok=false;}break;
        default:break;
      }
    }catch(e){ok=false;err=e.message;}
    rows.push({l,i,in_:[...prev],out_:[...cur],ok,err});
  });
  const allOk=rows.every(r=>r.ok);
  const el=document.getElementById('trace-out');
  let h=`<div class="slabel" style="margin-bottom:4px">Input [${n},${c},${h2},${w2}] → Output [${cur.join(',')}]</div>
    <div class="tt">
      <div class="tr hdr"><span>Layer</span><span>Input</span><span>Output</span><span></span></div>`;
  rows.forEach(r=>{
    const params=countLayerParams(r.l);
    h+=`<div class="tr" style="${r.ok?'':'background:rgba(239,68,68,.04)'}">
      <span class="trn">${r.i+1}. ${r.l.type}</span>
      <span class="tri">[${r.in_.join(',')}]</span>
      <span class="tro">[${r.out_.join(',')}]</span>
      <span class="${r.ok?'tok':'ter'}">${r.ok?'✓':('✗')}</span>
    </div>`;
    if(!r.ok) h+=`<div class="tr" style="background:rgba(239,68,68,.04);grid-template-columns:1fr"><span class="ter" style="padding-left:4px">↳ ${r.err}</span></div>`;
  });
  h+='</div>';
  h+=`<div class="vi ${allOk?'ok':'er'}" style="margin-top:5px"><span>${allOk?'✓':'✗'}</span><span>${allOk?`Tensor flows correctly through all ${layers.length} layers.`:'Shape mismatch detected — check highlighted layers.'}</span></div>`;

  // Live flow visualization
  h+=`<div class="as" style="margin-top:10px"><div class="ah">Live Tensor Flow</div>
    <div id="flow-vis" style="position:relative;padding:0"></div>
    <button class="btn" onclick="toggleLiveFlow()" id="flow-btn" style="width:100%;justify-content:center;margin-top:5px">▶ Animate Flow</button>
  </div>`;
  el.innerHTML=h;
  buildFlowVis(rows);
}

function buildFlowVis(rows){
  const el=document.getElementById('flow-vis');
  if(!el||!rows.length)return;
  const nh=22,gap=6,W=el.offsetWidth||320;
  const totalH=rows.length*(nh+gap)+10;
  let svg=`<svg width="100%" viewBox="0 0 ${W} ${totalH}" style="display:block">`;
  rows.forEach((r,i)=>{
    const y=i*(nh+gap);
    const col=getColor(r.l.type);
    const ok=r.ok;
    const shp=`[${r.out_.join(',')}]`;
    svg+=`<rect x="0" y="${y}" width="${W-2}" height="${nh}" rx="3" fill="${ok?'#fbfbf7':'#f8e9e5'}" stroke="${ok?col:'#c75b57'}" stroke-width="${ok?'0.6':'1'}" stroke-opacity="0.75"/>`;
    svg+=`<rect x="0" y="${y}" width="4" height="${nh}" rx="1" fill="${col}" fill-opacity="0.7"/>`;
    svg+=`<text x="12" y="${y+nh/2+1}" dominant-baseline="central" font-size="8" font-family="DM Sans, sans-serif" fill="${ok?'#292b29':'#a44743'}" font-weight="500">${r.i+1}. ${r.l.type}</text>`;
    svg+=`<text x="${W-6}" y="${y+nh/2+1}" text-anchor="end" dominant-baseline="central" font-size="6" font-family="DM Sans, sans-serif" fill="${ok?'#46777b':'#a44743'}">${shp}</text>`;
    if(i<rows.length-1)svg+=`<line x1="${W/2}" y1="${y+nh}" x2="${W/2}" y2="${y+nh+gap}" stroke="#cfd1c7" stroke-width="1"/>`;
  });
  svg+='</svg>';
  el.innerHTML=svg;
}

let flowActive=false,flowTimer=null;
function toggleLiveFlow(){
  flowActive=!flowActive;
  const btn=document.getElementById('flow-btn');
  if(!btn)return;
  if(flowActive){
    btn.textContent='⏹ Stop Flow';
    animateFlow();
  }else{
    btn.textContent='▶ Animate Flow';
    clearTimeout(flowTimer);
  }
}
function animateFlow(){
  if(!flowActive)return;
  const el=document.getElementById('flow-vis');
  if(!el){flowActive=false;return;}
  const svgEl=el.querySelector('svg');if(!svgEl){flowActive=false;return;}
  // Animate a glowing dot down the center of the SVG
  const W=svgEl.viewBox.baseVal.width||320;
  const H=svgEl.viewBox.baseVal.height||200;
  const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
  dot.setAttribute('r','3');dot.setAttribute('fill','#3b82f6');dot.setAttribute('opacity','0');
  dot.setAttribute('cx',W/2);dot.setAttribute('cy','0');
  svgEl.appendChild(dot);
  let t=0;const dur=1200;const start=performance.now();
  const anim=ts=>{
    const p=Math.min(1,(ts-start)/dur);
    dot.setAttribute('cy',p*H);
    const fade=p<0.1?p/0.1:p>0.9?(1-p)/0.1:1;
    dot.setAttribute('opacity',fade*0.9);
    if(p<1)requestAnimationFrame(anim);
    else{dot.remove();flowTimer=setTimeout(()=>{if(flowActive)animateFlow();},300);}
  };
  requestAnimationFrame(anim);
}

// ══════════════════════════════════════════════════
// PARAMETER + GRADIENT ANALYSIS
// ══════════════════════════════════════════════════
function countLayerParams(l){
  const p=l.params;
  switch(l.type){
    case 'Linear':case 'Linear+BN+Act':return(p.in_features||0)*(p.out_features||0)+(p.bias?(p.out_features||0):0);
    case 'Conv1d':return(p.in_channels||0)*(p.out_channels||0)*(p.kernel_size||1)+(p.out_channels||0);
    case 'Conv2d':case 'Conv+BN+Act':return(p.in_channels||0)*(p.out_channels||0)*(p.kernel_size||3)**2+(p.out_channels||0);
    case 'DepthwiseSepConv':return(p.in_channels||0)*(p.kernel_size||3)**2+(p.in_channels||0)*(p.out_channels||p.in_channels||0);
    case 'ConvTranspose2d':return(p.in_channels||0)*(p.out_channels||0)*(p.kernel_size||2)**2;
    case 'BatchNorm1d':case 'BatchNorm2d':case 'InstanceNorm2d':return(p.num_features||0)*2;
    case 'LayerNorm':case 'RMSNorm':return(p.normalized_shape||0)*2;
    case 'GroupNorm':return(p.num_channels||0)*2;
    case 'Embedding':return(p.num_embeddings||0)*(p.embedding_dim||0);
    case 'MultiheadAttention':return 4*(p.embed_dim||0)**2;
    case 'TransformerEncoderLayer':return 4*(p.d_model||0)**2+(p.d_model||0)*(p.dim_feedforward||0)*2;
    case 'TransformerDecoderLayer':return 6*(p.d_model||0)**2+(p.d_model||0)*(p.dim_feedforward||0)*2;
    case 'LSTM':{const{input_size:is=0,hidden_size:hs=0,num_layers:nl=1}=p;return 4*(is*hs+hs*hs+2*hs)*nl;}
    case 'GRU':{const{input_size:is=0,hidden_size:hs=0,num_layers:nl=1}=p;return 3*(is*hs+hs*hs+2*hs)*nl;}
    case 'RNN':{const{input_size:is=0,hidden_size:hs=0,num_layers:nl=1}=p;return(is*hs+hs*hs+2*hs)*nl;}
    case 'SEBlock':return 2*(p.channels||0)*(p.channels||0)/(p.reduction||16);
    case 'PositionalEncoding':return(p.max_len||512)*(p.d_model||128);
    default:return 0;
  }
}
function countParams(){return layers.reduce((t,l)=>t+countLayerParams(l),0);}
function estimateFlops(){
  return layers.reduce((t,l)=>{
    const p=l.params;
    switch(l.type){
      case 'Linear':return t+2*(p.in_features||0)*(p.out_features||0);
      case 'Conv2d':return t+2*(p.in_channels||0)*(p.out_channels||0)*(p.kernel_size||3)**2*32*32;
      case 'LSTM':{const{input_size:is=0,hidden_size:hs=0,num_layers:nl=1}=p;return t+8*(is+hs)*hs*nl*50;}
      case 'MultiheadAttention':{const ed=p.embed_dim||0;return t+4*ed*ed+2*ed*512;}
      default:return t;
    }
  },0);
}

function renderAnalysis(){
  const el=document.getElementById('rp-analysis');
  if(!layers.length){el.innerHTML='<div style="color:var(--t3);font-size:8px">Add layers to see analysis.</div>';return;}
  const total=countParams();const flops=estimateFlops();const mem=total*4/1024/1024;
  const issues=validate();const hasE=issues.some(i=>i.sev==='error');
  const depth=layers.length;
  const hasSkips=skips.length>0;
  const hasNorm=layers.some(l=>['BatchNorm1d','BatchNorm2d','LayerNorm','GroupNorm','RMSNorm','InstanceNorm2d'].includes(l.type));
  const hasDropout=layers.some(l=>['Dropout','Dropout2d','AlphaDropout'].includes(l.type));
  // Per-layer breakdown
  const lp=layers.map((l,i)=>({n:l.type,i,p:countLayerParams(l)})).filter(x=>x.p>0).sort((a,b)=>b.p-a.p).slice(0,6);
  const maxLP=Math.max(...lp.map(x=>x.p),1);
  // Grad risk
  const gradRisk=depth>=12&&!hasSkips?'High':depth>=7&&!hasNorm?'Medium':'Low';
  const gradColor=gradRisk==='High'?'var(--re)':gradRisk==='Medium'?'var(--ye)':'var(--gr)';
  // Regularization score
  const regScore=hasDropout&&hasNorm?'Good':hasDropout||hasNorm?'Moderate':'Weak';
  const regColor=regScore==='Good'?'var(--gr)':regScore==='Moderate'?'var(--ye)':'var(--re)';
  el.innerHTML=`
    <div class="as"><div class="ah">Parameter Overview</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px">
        <div class="rc"><div class="rk">Total Params</div><div class="rv">${total>1e6?(total/1e6).toFixed(2)+'M':total>1e3?(total/1e3).toFixed(1)+'K':total}</div></div>
        <div class="rc"><div class="rk">Memory (FP32)</div><div class="rv">${mem.toFixed(2)} MB</div></div>
        <div class="rc"><div class="rk">Est. FLOPs</div><div class="rv">${flops>1e9?(flops/1e9).toFixed(1)+'G':flops>1e6?(flops/1e6).toFixed(1)+'M':(flops/1e3).toFixed(0)+'K'}</div></div>
        <div class="rc"><div class="rk">Learnable Layers</div><div class="rv">${lp.length}</div></div>
      </div>
      <div class="slabel">Param Distribution by Layer</div>
      ${lp.map(x=>`<div class="bar">
        <div class="bk"><span>L${x.i+1} ${x.n}</span><span class="bv">${x.p>1e6?(x.p/1e6).toFixed(2)+'M':x.p>1e3?(x.p/1e3).toFixed(1)+'K':x.p}</span></div>
        <div class="btrack"><div class="bfill" style="width:${(x.p/maxLP*100).toFixed(0)}%;background:var(--acc)"></div></div>
      </div>`).join('')}
    </div>
    <div class="as"><div class="ah">Gradient Flow Analysis</div>
      <div class="rc" style="margin-bottom:6px"><div class="rk">Vanishing Gradient Risk</div><div class="rv" style="color:${gradColor}">${gradRisk}</div></div>
      <div class="vi ${hasSkips?'ok':'wn'}"><span>${hasSkips?'✓':'⚠'}</span><span>${hasSkips?`${skips.length} residual connection${skips.length>1?'s':''} — gradient highway active.`:'No skip connections — gradients must pass through all layers.'}</span></div>
      <div class="vi ${hasNorm?'ok':'wn'}"><span>${hasNorm?'✓':'⚠'}</span><span>${hasNorm?'Normalization layers stabilize activations and gradients.':'No normalization — consider BatchNorm or LayerNorm.'}</span></div>
      <div class="vi ${depth<8?'ok':depth<14?'wn':'er'}"><span>${depth<8?'✓':depth<14?'⚠':'⚠'}</span><span>Network depth: ${depth} layers${depth>=14?' — use residual connections for stability':depth>=8?' — monitor for gradient issues':''}.</span></div>
    </div>
    <div class="as"><div class="ah">Regularization Health</div>
      <div class="rc" style="margin-bottom:6px"><div class="rk">Regularization Score</div><div class="rv" style="color:${regColor}">${regScore}</div></div>
      <div class="vi ${hasDropout?'ok':'wn'}"><span>${hasDropout?'✓':'⚠'}</span><span>${hasDropout?'Dropout present — reduces overfitting.':'No dropout — may overfit on small datasets.'}</span></div>
      <div class="vi ${!hasE?'ok':'er'}"><span>${!hasE?'✓':'✗'}</span><span>${!hasE?'No structural errors — architecture can be compiled.':'Structural errors prevent training.'}</span></div>
    </div>`;
}

// ══════════════════════════════════════════════════
// PUBLICATION-READY DIAGRAM
// ══════════════════════════════════════════════════
function renderDiagram(){
  const el=document.getElementById('diag-svg');
  if(!layers.length){el.innerHTML='<div style="color:var(--t3);font-size:8px;text-align:center;padding:16px">Add layers to generate diagram</div>';return;}
  const archName=document.getElementById('arch-name').value||'Architecture';
  const W=430,nh=36,nodeW=280,gap=10,ox=(W-nodeW)/2,sy=30;
  const totalH=layers.length*(nh+gap)+sy+30;
  const skipCols=['#06b6d4','#a78bfa','#34d399','#fb923c'];

  // compute skip lane x positions, fanning left
  const laneW=12;
  const skipLanes=skips.map((s,i)=>({...s,x:ox-8-(i%3)*laneW}));

  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${totalH}" width="${W}" height="${totalH}" style="font-family:'DM Sans',sans-serif">
  <rect width="${W}" height="${totalH}" fill="#f7f7f2"/>
  <text x="${W/2}" y="16" text-anchor="middle" font-size="7" fill="#62665f" letter-spacing="0.15em" font-weight="500">${archName.toUpperCase()}</text>`;

  // Skip connection lanes
  skipLanes.forEach((s,si)=>{
    const col=skipCols[si%4];
    const y1=sy+s.f*(nh+gap)+nh;
    const y2=sy+s.t*(nh+gap);
    const lx=s.x;
    svg+=`<path d="M ${ox} ${y1-2} L ${lx} ${y1-2} L ${lx} ${y2+2} L ${ox} ${y2+2}"
      fill="none" stroke="${col}" stroke-width="0.8" stroke-opacity="0.8" stroke-dasharray="3,2"/>
      <polygon points="${ox-1},${y2+5} ${ox+1},${y2+5} ${ox},${y2+1}" fill="${col}" fill-opacity="0.8"/>
      <text x="${lx-2}" y="${(y1+y2)/2+3}" text-anchor="end" font-size="6" fill="${col}" fill-opacity="0.8">skip</text>`;
  });

  layers.forEach((l,i)=>{
    const y=sy+i*(nh+gap);const col=getColor(l.type);
    const shape=inferShape(l);
    const shapeText=shape?shape.slice(0,34):'';
    const params=countLayerParams(l);
    const pstr=params>1e6?(params/1e6).toFixed(1)+'M':params>1e3?(params/1e3).toFixed(0)+'K':params>0?params+'':' ';
    // connector
    if(i>0)svg+=`<line x1="${W/2}" y1="${y-gap}" x2="${W/2}" y2="${y+1}" stroke="#c4c6bd" stroke-width="1"/>
      <polygon points="${W/2-2},${y} ${W/2+2},${y} ${W/2},${y+3}" fill="#aeb2a9"/>`;
    // node
    svg+=`<rect x="${ox}" y="${y}" width="${nodeW}" height="${nh}" rx="3" fill="#fbfbf7" stroke="${col}" stroke-width="0.7" stroke-opacity="0.8"/>
      <rect x="${ox}" y="${y}" width="4" height="${nh}" rx="1" fill="${col}" fill-opacity="0.75"/>
      <text x="${ox+11}" y="${y+14}" dominant-baseline="central" font-size="8" fill="#292b29" font-weight="500">${l.type}</text>`;
    if(shapeText)svg+=`<text x="${ox+11}" y="${y+27}" dominant-baseline="central" font-size="6" fill="#46777b">out → ${shapeText}</text>`;
    if(params>0)svg+=`<text x="${ox+nodeW+10}" y="${y+18}" dominant-baseline="central" font-size="6" fill="#62665f">${pstr}</text>`;
    svg+=`<text x="${ox-7}" y="${y+nh/2+1}" text-anchor="end" dominant-baseline="central" font-size="6" fill="#62665f">${i+1}</text>`;
  });

  // Legend
  const legy=totalH-18;
  svg+=`<text x="8" y="${legy}" font-size="6" fill="#62665f">■ layer</text>
    <line x1="44" y1="${legy-2}" x2="58" y2="${legy-2}" stroke="#528b90" stroke-width="0.8" stroke-dasharray="2,1" stroke-opacity="0.8"/>
    <text x="62" y="${legy}" font-size="6" fill="#62665f">skip / residual</text>
    <text x="${W-6}" y="${legy}" text-anchor="end" font-size="6" fill="#62665f">${countParams()>1e6?(countParams()/1e6).toFixed(1)+'M params':countParams()>1e3?(countParams()/1e3).toFixed(0)+'K params':countParams()+' params'}</text>`;
  svg+='</svg>';
  el.innerHTML=svg;
}

function exportSVG(){
  renderDiagram();
  const name=document.getElementById('arch-name').value||'architecture';
  const svg=document.getElementById('diag-svg').innerHTML;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));
  a.download=name.replace(/\s+/g,'_')+'.svg';a.click();
}

function exportPNG(){
  renderDiagram();
  const svgEl=document.getElementById('diag-svg').querySelector('svg');
  if(!svgEl)return;
  const svgStr=new XMLSerializer().serializeToString(svgEl);
  const img=new Image();
  const url=URL.createObjectURL(new Blob([svgStr],{type:'image/svg+xml'}));
  img.onload=()=>{
    const c=document.createElement('canvas');
    const sc=3;// 3× for publication quality
    const width=svgEl.viewBox.baseVal.width;
    const height=svgEl.viewBox.baseVal.height;
    c.width=width*sc;c.height=height*sc;
    const ctx=c.getContext('2d');
    ctx.scale(sc,sc);ctx.drawImage(img,0,0);
    c.toBlob(blob=>{
      const a=document.createElement('a');
      a.download=(document.getElementById('arch-name').value||'architecture').replace(/\s+/g,'_')+'.png';
      a.href=URL.createObjectURL(blob);a.click();
      setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    },'image/png');
    URL.revokeObjectURL(url);
  };
  img.src=url;
}

// ══════════════════════════════════════════════════
// CODE GENERATION
// ══════════════════════════════════════════════════
function genCode(){
  const kw=s=>`<span class="kw">${s}</span>`;
  const fn=s=>`<span class="fn">${s}</span>`;
  const nu=s=>`<span class="nu">${s}</span>`;
  const cm=s=>`<span class="cm">${s}</span>`;
  const s2=s=>`<span class="st2">${s}</span>`;
  const name=(document.getElementById('arch-name').value||'MyModel').replace(/[^a-zA-Z0-9]/g,'_');
  const hasSkips=skips.length>0;
  const total=countParams();
  const lines=[];
  lines.push(cm('# ══════════════════════════════════════'));
  lines.push(cm(`# ${name}`));
  lines.push(cm(`# Parameters: ~${total>1e6?(total/1e6).toFixed(2)+'M':(total/1e3).toFixed(1)+'K'}`));
  lines.push(cm(`# Memory (FP32): ~${(total*4/1024/1024).toFixed(2)} MB`));
  if(skips.length)lines.push(cm(`# Residual: ${skips.map(s=>`L${s.f+1}→L${s.t+1}`).join(', ')}`));
  lines.push(cm('# ══════════════════════════════════════'));
  lines.push(kw('import')+' torch');
  lines.push(kw('import')+' torch.nn '+kw('as')+' nn');
  lines.push(kw('import')+' torch.nn.functional '+kw('as')+' F');
  lines.push('');
  lines.push(kw('class ')+fn(name)+'(nn.Module):');
  lines.push(`    ${kw('def ')}${fn('__init__')}(${kw('self')}):`);
  lines.push(`        ${fn('super')}().${fn('__init__')}()`);
  if(!hasSkips){
    lines.push('        self.net = nn.Sequential(');
    layers.forEach((l,i)=>lines.push(`            ${buildLayerCode(l,nu,s2)}`+(i<layers.length-1?',':'')));
    lines.push('        )');
    lines.push('');
    lines.push(`    ${kw('def ')}${fn('forward')}(${kw('self')}, x):`);
    lines.push(`        ${kw('return')} self.net(x)`);
  }else{
    layers.forEach((l,i)=>lines.push(`        self.l${i+1} = ${buildLayerCode(l,nu,s2)}`));
    lines.push('');
    lines.push(`    ${kw('def ')}${fn('forward')}(${kw('self')}, x):`);
    layers.forEach((l,i)=>{
      skips.filter(s=>s.t===i).forEach(s=>lines.push(`        x = x + out${s.f+1}  ${cm('# residual from L'+(s.f+1))}`));
      if(skips.some(s=>s.f===i)){lines.push(`        out${i+1} = x`);lines.push(`        x = self.l${i+1}(x)`);}
      else lines.push(`        x = self.l${i+1}(x)`);
    });
    lines.push(`        ${kw('return')} x`);
  }
  lines.push('');
  lines.push(cm('# ── Instantiate & verify ──────────────'));
  lines.push(`model = ${fn(name)}()`);
  lines.push(`n_params = ${fn('sum')}(p.${fn('numel')}() ${kw('for')} p ${kw('in')} model.${fn('parameters')}() ${kw('if')} p.requires_grad)`);
  lines.push(`${fn('print')}(${s2(`f"${name}: {n_params:,} trainable parameters"`)})`);
  return lines.join('\n');
}

function buildLayerCode(l,nu,s2){
  const p=l.params;
  const T=s=>s;
  switch(l.type){
    case 'Linear':return`nn.Linear(${nu(p.in_features)}, ${nu(p.out_features)}, bias=${p.bias?s2('True'):s2('False')})`;
    case 'Bilinear':return`nn.Bilinear(${nu(p.in1_features)}, ${nu(p.in2_features)}, ${nu(p.out_features)})`;
    case 'Conv1d':return`nn.Conv1d(${nu(p.in_channels)}, ${nu(p.out_channels)}, ${nu(p.kernel_size)}, stride=${nu(p.stride)}, padding=${nu(p.padding)})`;
    case 'Conv2d':return`nn.Conv2d(${nu(p.in_channels)}, ${nu(p.out_channels)}, ${nu(p.kernel_size)}, stride=${nu(p.stride)}, padding=${nu(p.padding)})`;
    case 'Conv3d':return`nn.Conv3d(${nu(p.in_channels)}, ${nu(p.out_channels)}, ${nu(p.kernel_size)}, stride=${nu(p.stride)}, padding=${nu(p.padding)})`;
    case 'ConvTranspose2d':return`nn.ConvTranspose2d(${nu(p.in_channels)}, ${nu(p.out_channels)}, ${nu(p.kernel_size)}, stride=${nu(p.stride)})`;
    case 'DepthwiseSepConv':return`nn.Sequential(nn.Conv2d(${nu(p.in_channels)}, ${nu(p.in_channels)}, ${nu(p.kernel_size)}, padding=${nu(p.padding)}, groups=${nu(p.in_channels)}), nn.Conv2d(${nu(p.in_channels)}, ${nu(p.out_channels||p.in_channels)}, ${nu(1)}))`;
    case 'ReLU':return`nn.ReLU(inplace=${s2('True')})`;
    case 'LeakyReLU':return`nn.LeakyReLU(negative_slope=${nu(p.negative_slope)})`;
    case 'GELU':return`nn.GELU()`;case 'SiLU':return`nn.SiLU()`;case 'Mish':return`nn.Mish()`;
    case 'Sigmoid':return`nn.Sigmoid()`;case 'Tanh':return`nn.Tanh()`;
    case 'Softmax':case 'LogSoftmax':return`nn.${l.type}(dim=${nu(p.dim)})`;
    case 'Dropout':case 'Dropout2d':case 'AlphaDropout':return`nn.${l.type}(p=${nu(parseFloat(p.p).toFixed(2))})`;
    case 'BatchNorm1d':case 'BatchNorm2d':case 'InstanceNorm2d':return`nn.${l.type}(${nu(p.num_features)})`;
    case 'LayerNorm':case 'RMSNorm':return`nn.${l.type}(${nu(p.normalized_shape)})`;
    case 'GroupNorm':return`nn.GroupNorm(${nu(p.num_groups)}, ${nu(p.num_channels)})`;
    case 'MaxPool1d':case 'MaxPool2d':case 'AvgPool2d':return`nn.${l.type}(kernel_size=${nu(p.kernel_size)}, stride=${nu(p.stride)})`;
    case 'AdaptiveAvgPool2d':return`nn.AdaptiveAvgPool2d(output_size=${nu(p.output_size)})`;
    case 'Flatten':return`nn.Flatten(start_dim=${nu(1)})`;
    case 'Permute':return`# nn.Permute (use in forward: x.permute(${p.dims||'[0,2,1]'}))`;
    case 'Embedding':return`nn.Embedding(${nu(p.num_embeddings)}, ${nu(p.embedding_dim)})`;
    case 'MultiheadAttention':return`nn.MultiheadAttention(${nu(p.embed_dim)}, ${nu(p.num_heads)}, dropout=${nu(p.dropout||0)})`;
    case 'TransformerEncoderLayer':return`nn.TransformerEncoderLayer(d_model=${nu(p.d_model)}, nhead=${nu(p.nhead)}, dim_feedforward=${nu(p.dim_feedforward)}, dropout=${nu(p.dropout||0.1)}, batch_first=${s2('True')})`;
    case 'TransformerDecoderLayer':return`nn.TransformerDecoderLayer(d_model=${nu(p.d_model)}, nhead=${nu(p.nhead)}, dim_feedforward=${nu(p.dim_feedforward)}, dropout=${nu(p.dropout||0.1)}, batch_first=${s2('True')})`;
    case 'LSTM':case 'GRU':case 'RNN':return`nn.${l.type}(${nu(p.input_size)}, ${nu(p.hidden_size)}, num_layers=${nu(p.num_layers)}, batch_first=${p.batch_first?s2('True'):s2('False')})`;
    case 'Conv+BN+Act':return`nn.Sequential(nn.Conv2d(${nu(p.in_channels)}, ${nu(p.out_channels)}, ${nu(p.kernel_size||3)}, padding=${nu(p.padding||1)}), nn.BatchNorm2d(${nu(p.out_channels)}), nn.ReLU(inplace=${s2('True')}))`;
    case 'Linear+BN+Act':return`nn.Sequential(nn.Linear(${nu(p.in_features)}, ${nu(p.out_features)}), nn.BatchNorm1d(${nu(p.out_features)}), nn.ReLU(inplace=${s2('True')}))`;
    case 'SEBlock':return`# SEBlock(channels=${p.channels}, reduction=${p.reduction}) — implement separately`;
    case 'PositionalEncoding':return`# PositionalEncoding(d_model=${p.d_model}, max_len=${p.max_len}) — implement separately`;
    case 'ResidualBlock':return`# ResidualBlock(channels=${p.channels}) — implement with skip connection`;
    default:return`nn.Identity()  # ${l.type}`;
  }
}

function renderCode(){
  const el=document.getElementById('rp-code');
  if(!layers.length){el.innerHTML='<div style="color:var(--t3);font-size:8px">Add layers to generate code.</div>';return;}
  const code=genCode();
  el.innerHTML=`<pre class="cb2">${code}</pre>
    <div style="display:flex;gap:5px;margin-top:6px">
      <button class="btn" style="flex:1;justify-content:center" onclick="copyCode()">⎘ Copy</button>
      <button class="btn gc" style="flex:1;justify-content:center" onclick="exportCode()">⬇ Export .py</button>
    </div>`;
}
function copyCode(){
  navigator.clipboard.writeText(document.querySelector('.cb2')?.textContent||'');
}
function exportCode(){
  const name=document.getElementById('arch-name').value||'model';
  const code=genCode().replace(/<[^>]+>/g,'');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([code],{type:'text/plain'}));
  a.download=name.replace(/\s+/g,'_')+'.py';a.click();
  closeOv('ov-train');
}

// ══════════════════════════════════════════════════
// STATS
// ══════════════════════════════════════════════════
function renderStats(){
  const p=countParams();const f=estimateFlops();
  const issues=validate();
  const hasE=issues.some(i=>i.sev==='error'),hasW=issues.some(i=>i.sev==='warn');
  document.getElementById('sl').textContent=layers.length;
  document.getElementById('sp').textContent=p>1e6?(p/1e6).toFixed(2)+'M':p>1e3?(p/1e3).toFixed(1)+'K':p;
  document.getElementById('sm').textContent=(p*4/1024/1024).toFixed(2)+' MB';
  document.getElementById('sf').textContent=f>1e9?(f/1e9).toFixed(1)+'G':f>1e6?(f/1e6).toFixed(1)+'M':(f/1e3).toFixed(0)+'K';
  document.getElementById('sd').textContent=layers.length+skips.length;
  const st=document.getElementById('ss'),sb=document.getElementById('status-badge');
  if(!layers.length){st.textContent='—';st.className='sv';sb.className='badge iv';sb.textContent='Empty';}
  else if(hasE){st.textContent='Errors';st.className='sv r';sb.className='badge iv';sb.textContent='Invalid';}
  else if(hasW){st.textContent='Warnings';st.className='sv y';sb.className='badge w';sb.textContent='Warnings';}
  else{st.textContent='Valid';st.className='sv g';sb.className='badge v';sb.textContent='Valid';}
}

// ══════════════════════════════════════════════════
// MODE + TABS
// ══════════════════════════════════════════════════
function setMode(m,el){
  document.querySelectorAll('.htab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  const map={design:'inspector',analyze:'analysis',diagram:'diagram',compare:'code'};
  const tid=map[m]||'inspector';
  const rtel=document.querySelector(`.rtab[onclick*="'${tid}'"]`);
  if(rtel)setRTab(tid,rtel);
}
function setRTab(n,el){
  document.querySelectorAll('.rtab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  document.querySelectorAll('.rp').forEach(p=>p.classList.remove('on'));
  document.getElementById('rp-'+n).classList.add('on');
  if(n==='analysis')renderAnalysis();
  if(n==='diagram')renderDiagram();
  if(n==='code')renderCode();
  if(n==='validate')renderValidate();
}

// ══════════════════════════════════════════════════
// SAVE / LOAD
// ══════════════════════════════════════════════════
function openSave(){
  if(!layers.length){alert('Add layers first.');return;}
  document.getElementById('sv-n').value=document.getElementById('arch-name').value||'Architecture';
  document.getElementById('sv-notes').value='';
  document.getElementById('ov-save').classList.add('open');
}
function doSave(){
  const n=document.getElementById('sv-n').value.trim()||'Untitled';
  const notes=document.getElementById('sv-notes').value.trim();
  const p=countParams();
  saved.push({name:n,notes,layers:JSON.parse(JSON.stringify(layers)),skips:JSON.parse(JSON.stringify(skips)),
    params:p,mem:(p*4/1024/1024).toFixed(2)+'MB',ts:new Date().toLocaleTimeString()});
  closeOv('ov-save');renderSavedList();
}
function renderSavedList(){
  const el=document.getElementById('saved-list');
  if(!saved.length){el.innerHTML='<div style="color:var(--t3);font-size:8px;text-align:center;padding:14px 0">No saved architectures.</div>';return;}
  el.innerHTML=saved.map((a,i)=>`<div class="si">
    <div class="sn">${a.name}</div>
    <div class="sm">${a.layers.length}L · ${(a.params/1e3).toFixed(1)}K params · ${a.ts}</div>
    ${a.notes?`<div class="sm" style="font-style:italic;margin-top:2px">${a.notes}</div>`:''}
    <div style="display:flex;gap:4px;margin-top:5px">
      <button class="btn sm" onclick="loadSaved(${i})" style="flex:1;justify-content:center">Load</button>
      <button class="btn sm" onclick="saved.splice(${i},1);renderSavedList()" style="border-color:var(--re);color:var(--re)">×</button>
    </div>
  </div>`).join('');
}
function loadSaved(i){
  const a=saved[i];
  layers=JSON.parse(JSON.stringify(a.layers));
  skips=JSON.parse(JSON.stringify(a.skips));
  document.getElementById('arch-name').value=a.name;
  selIdx=0;render();
}

// ══════════════════════════════════════════════════
// TRAINING
// ══════════════════════════════════════════════════
function openTrain(){
  document.getElementById('tlog').innerHTML='';
  document.getElementById('pf').style.width='0%';
  document.getElementById('tres').style.display='none';
  document.getElementById('tbtn').disabled=false;
  document.getElementById('ov-train').classList.add('open');
}
function buildTfjsModel(){
  if(!window.tf)return{error:'TensorFlow.js is not loaded. Check your network connection and reload.'};
  if(!layers.length)return{error:'Add at least one layer before training.'};
  const unsupported=layers.find(l=>['Bilinear','Conv1d','Conv3d','ConvTranspose2d','DepthwiseSepConv','MultiheadAttention','TransformerEncoderLayer','TransformerDecoderLayer','LSTM','GRU','RNN','GroupNorm','InstanceNorm2d','RMSNorm','SEBlock','PositionalEncoding','ResidualBlock','Unflatten','Permute'].includes(l.type));
  if(unsupported)return{error:`${unsupported.type} cannot be mapped faithfully to TensorFlow.js yet. Use Conv2d, Dense, activations, pooling, normalization, Flatten, and Dropout layers for a real run.`};
  const first=layers[0];
  const imageInput=['Conv2d','BatchNorm2d','MaxPool2d','AvgPool2d','AdaptiveAvgPool2d'].includes(first.type);
  const inputShape=imageInput?[32,32,first.params.in_channels||3]:[first.params.in_features||128];
  const model=tf.sequential();
  model.add(tf.layers.inputLayer({inputShape}));
  layers.forEach(l=>{
    const p=l.params;
    switch(l.type){
      case 'Linear':case 'Linear+BN+Act':model.add(tf.layers.dense({units:p.out_features||64,useBias:p.bias!==false,activation:l.type==='Linear+BN+Act'?'relu':undefined}));break;
      case 'Conv2d':model.add(tf.layers.conv2d({filters:p.out_channels||32,kernelSize:p.kernel_size||3,strides:p.stride||1,padding:(p.padding||0)>0?'same':'valid',activation:undefined}));break;
      case 'BatchNorm1d':case 'BatchNorm2d':model.add(tf.layers.batchNormalization());break;
      case 'LayerNorm':model.add(tf.layers.layerNormalization({axis:-1}));break;
      case 'ReLU':model.add(tf.layers.activation({activation:'relu'}));break;
      case 'LeakyReLU':model.add(tf.layers.leakyReLU({alpha:p.negative_slope||0.01}));break;
      case 'GELU':model.add(tf.layers.activation({activation:'gelu'}));break;
      case 'SiLU':model.add(tf.layers.activation({activation:'swish'}));break;
      case 'Sigmoid':model.add(tf.layers.activation({activation:'sigmoid'}));break;
      case 'Tanh':model.add(tf.layers.activation({activation:'tanh'}));break;
      case 'Softmax':model.add(tf.layers.activation({activation:'softmax'}));break;
      case 'LogSoftmax':model.add(tf.layers.activation({activation:'softmax'}));break;
      case 'Dropout':case 'Dropout2d':case 'AlphaDropout':model.add(tf.layers.dropout({rate:p.p||0.5}));break;
      case 'MaxPool2d':case 'AvgPool2d':model.add(l.type==='MaxPool2d'?tf.layers.maxPooling2d({poolSize:p.kernel_size||2,strides:p.stride||2}):tf.layers.averagePooling2d({poolSize:p.kernel_size||2,strides:p.stride||2}));break;
      case 'AdaptiveAvgPool2d':model.add(tf.layers.globalAveragePooling2d());break;
      case 'Flatten':model.add(tf.layers.flatten());break;
      default:break;
    }
  });
  const outputShape=model.outputs[0].shape;
  if(outputShape[outputShape.length-1]!==2)model.add(tf.layers.dense({units:2,activation:'softmax'}));
  return{model,inputShape};
}
async function runTrain(){
  const button=document.getElementById('tbtn'),log=document.getElementById('tlog'),pf=document.getElementById('pf'),result=document.getElementById('tres');
  button.disabled=true;log.textContent='';pf.style.width='0%';result.style.display='none';
  let model,x,y;
  const write=(tag,message,color='#62665f')=>{log.innerHTML+=`<span style="color:${color}">${tag}</span> <span style="color:#292d2a">${message}</span>\n`;log.scrollTop=log.scrollHeight;};
  try{
    if(!window.tf)throw new Error('TensorFlow.js is not loaded. Check your network connection and reload.');
    const epochs=Math.min(500,Math.max(1,+document.getElementById('ep').value||30));
    const batchSize=Math.max(1,+document.getElementById('bs').value||32);
    const lr=Math.max(0.000001,+document.getElementById('lr').value||0.001);
    const built=buildTfjsModel();if(built.error)throw new Error(built.error);model=built.model;
    const sampleCount=Math.max(64,Math.min(2048,batchSize*8));
    x=tf.randomNormal([sampleCount,...built.inputShape]);
    const signal=x.mean(built.inputShape.length===3?[1,2,3]:[1]);
    y=tf.oneHot(signal.greater(0).toInt(),2);
    const optimizer=document.getElementById('opt').value;
    const opt=optimizer==='SGD'?tf.train.sgd(lr):optimizer==='RMSprop'?tf.train.rmsprop(lr):tf.train.adam(lr);
    model.compile({optimizer:opt,loss:'categoricalCrossentropy',metrics:['accuracy']});
    write('[TFJS]',`TensorFlow.js ${tf.version.tfjs} · WebGL backend: ${tf.getBackend()}`,'#46777b');
    write('[MODEL]',`${model.countParams().toLocaleString()} trainable parameters · input [${built.inputShape.join(',')}]`,'#46777b');
    write('[DATA]',`${sampleCount} generated samples · binary target from tensor signal · batch_size=${batchSize}`,'#46777b');
    write('[RUN]','Starting real gradient updates with model.fit(...)','#46777b');
    const history=await model.fit(x,y,{epochs,batchSize,validationSplit:0.2,shuffle:true,callbacks:{onEpochEnd:async(epoch,logs)=>{
      const loss=(logs.loss??0).toFixed(4),acc=((logs.acc??logs.accuracy??0)*100).toFixed(2),valLoss=(logs.val_loss??0).toFixed(4),valAcc=((logs.val_acc??logs.val_accuracy??0)*100).toFixed(2);
      write(`[EPOCH ${String(epoch+1).padStart(3)}/${epochs}]`,`loss=${loss} · acc=${acc}% · val_loss=${valLoss} · val_acc=${valAcc}%`, '#62665f');pf.style.width=`${((epoch+1)/epochs*100).toFixed(0)}%`;
      await tf.nextFrame();
    }}});
    const last=history.history,valLoss=last.val_loss.at(-1),valAcc=(last.val_accuracy||last.val_acc).at(-1);
    result.style.display='grid';result.style.gridTemplateColumns='1fr 1fr 1fr';result.style.gap='5px';
    result.innerHTML=`<div class="rc"><div class="rk">Val Acc</div><div class="rv" style="color:var(--gr)">${(valAcc*100).toFixed(2)}%</div></div><div class="rc"><div class="rk">Val Loss</div><div class="rv" style="color:var(--cy)">${valLoss.toFixed(4)}</div></div><div class="rc"><div class="rk">Backend</div><div class="rv">${tf.getBackend()}</div></div>`;
    write('[DONE]','Experiment complete. Metrics above came from TensorFlow.js training.','#4f8e78');pf.style.width='100%';
  }catch(error){write('[ERROR]',error.message,'#a44743');write('[ABORT]','No simulated metrics were generated.','#a44743');}
  finally{if(x)x.dispose();if(y)y.dispose();if(model)model.dispose();button.disabled=false;}
}
function runMsgs(msgs,showRes){
  const log=document.getElementById('tlog'),pf=document.getElementById('pf');
  let i=0;
  const step=()=>{
    if(i>=msgs.length){
      if(showRes){
        const last=msgs.filter(m=>m.t==='[EPOCH]').pop();
        if(last){
          const acc=last.m.match(/val_acc=([\d.]+)/)?.[1]||'—';
          const vl=last.m.match(/val_loss=([\d.]+)/)?.[1]||'—';
          const res=document.getElementById('tres');
          res.style.display='grid';res.style.gridTemplateColumns='1fr 1fr 1fr';res.style.gap='5px';
          res.innerHTML=`<div class="rc"><div class="rk">Val Acc</div><div class="rv" style="color:var(--gr)">${acc}%</div></div>
            <div class="rc"><div class="rk">Val Loss</div><div class="rv" style="color:var(--cy)">${vl}</div></div>
            <div class="rc"><div class="rk">Params</div><div class="rv">${(countParams()/1e3).toFixed(1)}K</div></div>`;
        }
      }
      return;
    }
    const{t,m}=msgs[i];
    const col=t==='[ERROR]'||t==='[ABORT]'?'#ef4444':t==='[WARN]'?'#f59e0b':t==='[EPOCH]'?'#3b82f6':t==='[DONE]'?'#10b981':'#6b748f';
    log.innerHTML+=`<span style="color:${col}">${t} </span><span style="color:#c8cde8">${m}</span>\n`;
    log.scrollTop=log.scrollHeight;
    pf.style.width=((i+1)/msgs.length*100).toFixed(0)+'%';
    i++;setTimeout(step,t==='[EPOCH]'?55:25);
  };
  step();
}

function closeOv(id){document.getElementById(id).classList.remove('open');}
function resetNet(){
  layers=[];skips=[];selIdx=null;nextId=1;
  document.getElementById('arch-name').value='Untitled Architecture';
  render();
}

// ══════════════════════════════════════════════════
// MASTER RENDER
// ══════════════════════════════════════════════════
function render(){
  renderCanvas();renderInspector();renderValidate();renderCode();renderStats();
  const active=document.querySelector('.rp.on');
  if(active){
    const id=active.id.replace('rp-','');
    if(id==='analysis')renderAnalysis();
    if(id==='diagram')renderDiagram();
  }
}

// BOOT
buildPalette();
buildBaselines();
renderCustomList();
renderSavedList();
render();