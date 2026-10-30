import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {SelectionCoordinator,inventorySignature} from './selection.mjs';

const $=id=>document.getElementById(id), stage=$('stage'), status=$('status');
const coordinator=new SelectionCoordinator(), loader=new GLTFLoader(), cache=new Map(), transforms=new Map();
let assets=[], signature='', info, head, currentHair, active=null, pending=null, job=null, initialized=false, refreshing=false;
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(35,1,.0001,2);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
renderer.setClearColor(0x000000,0);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.0;stage.prepend(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.025;controls.maxDistance=.15;controls.target.set(0,.001,0);camera.position.set(0,.002,.05);
scene.add(new THREE.HemisphereLight(0xe9f1ff,0x718080,1.5));
const key=new THREE.DirectionalLight(0xfff4e7,2.0);key.position.set(-.025,.045,.06);scene.add(key);
const fill=new THREE.DirectionalLight(0xd8e6ff,.65);fill.position.set(.04,.02,.015);scene.add(fill);
const resize=()=>{const r=stage.getBoundingClientRect();renderer.setSize(r.width,r.height);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe(stage);resize();
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
function message(text,error=false){status.textContent=text;status.classList.toggle('error',error);}
async function request(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
function download(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000);}
function sourceMaterials(root){root.traverse(o=>{if(o.isMesh&&!o.userData.sourceMaterial)o.userData.sourceMaterial=o.material;});}
function applyMaterial(root){if(!root)return;const mode=$('material').value;root.traverse(o=>{if(!o.isMesh)return;const original=o.userData.sourceMaterial;if(o.userData.overrideMaterial){o.userData.overrideMaterial.dispose();delete o.userData.overrideMaterial;}if(mode==='original'){o.material=original;return;}const m=Array.isArray(original)?original[0]:original;const options={color:mode==='gray'?0xbfc6c8:m.color,side:m.side,vertexColors:mode!=='gray'&&Boolean(o.geometry.attributes.color)};if(mode==='unlit')options.map=m.map;o.material=mode==='gray'?new THREE.MeshStandardMaterial({...options,roughness:.85,metalness:0}):new THREE.MeshBasicMaterial(options);o.userData.overrideMaterial=o.material;});}
function dispose(root){root?.traverse(o=>{if(!o.isMesh)return;o.geometry.dispose();const list=[o.material,o.userData.sourceMaterial].flat();for(const m of new Set(list.filter(Boolean))){for(const v of Object.values(m)){if(v?.isTexture){v.dispose();v.source?.data?.close?.();}}m.dispose();}});}
function values(){return {scale:Number($('scale').value)/100,height_mm:Number($('height').value),yaw_deg:Number($('yaw').value)};}
function showValues(t){$('scale').value=String(t.scale*100);$('height').value=String(t.height_mm);$('yaw').value=String(t.yaw_deg);updateTransform();}
function updateTransform(){const t=values();$('scale-value').textContent=Math.round(t.scale*100)+'%';$('height-value').textContent=t.height_mm.toFixed(1)+' mm';$('yaw-value').textContent=t.yaw_deg+'°';if(currentHair){const p=info.pivot_m;currentHair.position.set(p[0],p[1]+t.height_mm/1000,p[2]);currentHair.rotation.y=THREE.MathUtils.degToRad(t.yaw_deg);currentHair.scale.setScalar(t.scale);}if(active)transforms.set(active.id+active.revision,t);}
function paintCatalog(){const box=$('catalog');box.replaceChildren();const none=document.createElement('button');none.className='asset'+(!active&&!pending?' selected':'');none.textContent='Không tóc';none.onclick=()=>selectNone();box.append(none);
 for(const a of assets){const b=document.createElement('button');b.className='asset'+((pending||active)?.id===a.id?' selected':'');b.disabled=a.state!=='available';const name=document.createElement('strong');name.textContent=a.name;const filename=document.createElement('span');filename.className='file';filename.textContent=a.filename;const tag=document.createElement('small');tag.textContent=a.state==='error'?a.error:a.fit;b.append(name,filename,tag);b.onclick=()=>choose(a);box.append(b);}
 $('count').textContent=assets.filter(a=>a.state==='available').length+' mẫu';}
function selectNone(){coordinator.invalidate();if(currentHair)scene.remove(currentHair);currentHair=null;active=null;pending=null;job=null;$('export').disabled=false;$('save').disabled=false;paintCatalog();message('Đang xem đầu nguồn, không có tóc.');}
async function choose(asset){const intent=coordinator.epoch+1;pending=asset;paintCatalog();message('Đang chuẩn bị '+asset.name+'…');$('export').disabled=true;$('save').disabled=true;
 try{await coordinator.choose(async isCurrent=>{
  let next=await request('/api/load',{id:asset.id,revision:asset.revision});
  while(next.state==='loading'||next.state==='queued'){if(!isCurrent())return null;await new Promise(r=>setTimeout(r,250));next=await request('/api/jobs/'+next.id);}
  if(!isCurrent())return null;if(next.state==='error')throw Error(next.error);
  let wrapper=cache.get(next.id);
  if(!wrapper){const model=await loader.loadAsync(next.model_url);if(!isCurrent()){dispose(model.scene);return null;}sourceMaterials(model.scene);wrapper=new THREE.Group();model.scene.position.sub(new THREE.Vector3(...info.pivot_m));wrapper.add(model.scene);cache.set(next.id,wrapper);}
  return {wrapper,job:next,asset};
 },result=>{if(currentHair)scene.remove(currentHair);currentHair=result.wrapper;scene.add(currentHair);active=result.asset;pending=null;job=result.job;showValues(transforms.get(active.id+active.revision)||{scale:1,height_mm:0,yaw_deg:0});applyMaterial(currentHair);paintCatalog();$('fit-note').textContent=job.fit+' · Bản xem trước, chưa kiểm tra in.';message('Đang xem: '+active.name+' · Mặt và màu da được giữ nguyên.');$('export').disabled=false;$('save').disabled=false;
  while(cache.size>3){const key=[...cache.keys()].find(k=>cache.get(k)!==currentHair);const old=cache.get(key);cache.delete(key);dispose(old);}
 });}catch(e){if(coordinator.isCurrent(intent)&&pending?.id===asset.id&&pending?.revision===asset.revision){pending=null;paintCatalog();message('Không nạp được '+asset.name+': '+e.message,true);$('export').disabled=!currentHair;$('save').disabled=!currentHair;}}
}
async function refresh(){if(refreshing)return;refreshing=true;try{const data=await request('/api/assets');const nextSignature=inventorySignature(data.assets);assets=data.assets;if(signature!==nextSignature){signature=nextSignature;paintCatalog();
 if(pending&&!assets.some(a=>a.id===pending.id)){coordinator.invalidate();pending=null;paintCatalog();message('File đang nạp đã bị xóa; giữ bản xem trước hiện tại.');}
 if(active){const replacement=assets.find(a=>a.id===active.id);if(!replacement){selectNone();message('File tóc đã được xóa; đang xem đầu không tóc.');}else if(replacement.state==='available'&&replacement.revision!==active.revision)choose(replacement);}
 if(!initialized){const first=assets.find(a=>a.mode==='current')||assets.find(a=>a.state==='available');if(first){initialized=true;choose(first);}}
 }}catch(e){message('Mất kết nối server nội bộ. Hãy giữ PowerShell chạy và thử Làm mới.',true);}finally{refreshing=false;}}
$('refresh').onclick=refresh;for(const id of ['scale','height','yaw'])$(id).oninput=updateTransform;
$('reset-fit').onclick=()=>showValues({scale:1,height_mm:0,yaw_deg:0});
$('material').onchange=()=>{applyMaterial(head);applyMaterial(currentHair);};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>{const a=THREE.MathUtils.degToRad(Number(b.dataset.view));camera.position.set(Math.sin(a)*.05,.002,Math.cos(a)*.05);controls.target.set(0,.001,0);controls.update();};
$('rotate').onclick=()=>{controls.autoRotate=!controls.autoRotate;$('rotate').textContent=controls.autoRotate?'Dừng xoay':'Tự xoay';};
$('save').onclick=()=>download(new Blob([JSON.stringify({version:info.version,head_sha256:info.head_sha256,asset:active?{filename:active.filename,sha256:active.sha256,revision:active.revision}:null,transform:values(),preview_only:true,ready_for_machine:false},null,2)],{type:'application/json'}),'hair-selection.json');
$('export').onclick=async()=>{if(!job){download(await (await fetch(info.head_url)).blob(),'head-preview.glb');return;}try{message('Đang xuất GLB xem trước…');const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job:job.id,transform:values()})});if(!r.ok)throw Error((await r.json()).error);download(await r.blob(),'hair-studio-preview.glb');message('Đã xuất GLB xem trước. Chưa kiểm tra in cho lựa chọn này.');}catch(e){message(e.message,true);}};
try{info=await request('/api/info');$('run-name').textContent='Nguồn: '+info.run;const model=await loader.loadAsync(info.head_url);head=model.scene;sourceMaterials(head);scene.add(head);applyMaterial(head);message('Đầu nguồn đã sẵn sàng. Đang quét thư viện tóc…');await refresh();setInterval(refresh,info.poll_ms);}catch(e){message('Không mở được mô hình: '+e.message,true);}
// Read-only diagnostics for local regression tests; no app control is exposed.
window.hairStudioState=()=>({head_uuid:head?.uuid,head_position:head?.position.toArray(),active_id:active?.id,active_revision:active?.revision,hair_count:scene.children.filter(x=>x===currentHair).length,camera:camera.position.toArray(),mode:$('material').value,status:status.textContent});
