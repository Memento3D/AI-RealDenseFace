import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const $ = id => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
$('canvas').appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color('#26272d');
scene.add(new THREE.HemisphereLight(0xffffff, 0x555566, 2));
for (const [x,y,z,power] of [[-1,2,3,2],[2,1,0,1]]) {
  const light = new THREE.DirectionalLight(0xffffff, power); light.position.set(x,y,z); scene.add(light);
}
const camera = new THREE.OrthographicCamera(-.2,.2,.2,-.2,.01,20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.minZoom = .25; controls.maxZoom = 12;
let model, request = 0, span = .36;
function resize() {
  const { width, height } = $('canvas').getBoundingClientRect();
  renderer.setSize(width,height); const aspect=width/height;
  camera.left=-span*aspect/2;camera.right=span*aspect/2;camera.top=span/2;camera.bottom=-span/2;camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($('canvas'));
function view(name) {
  const full = name === 'full'; span = full ? 1.2 : .36;
  const target = new THREE.Vector3(-.02, full ? .52 : .90, 0);
  const directions = { front:[0,0,2], quarter:[-1,.05,2], side:[2,0,0], back:[0,0,-2], full:[0,.1,3] };
  camera.position.copy(target).add(new THREE.Vector3(...directions[name])); camera.zoom=1;
  controls.target.copy(target); controls.update(); resize();
}
function dispose(root) {
  const geometries=new Set(),materials=new Set(),textures=new Set();
  root.traverse(o=>{ if(o.geometry)geometries.add(o.geometry); for(const m of o.material ? [].concat(o.material) : [])materials.add(m); });
  for(const m of materials){for(const v of Object.values(m))if(v?.isTexture)textures.add(v);m.dispose();}
  for(const t of textures)t.dispose();for(const g of geometries)g.dispose();
}
function visibility() {
  model?.traverse(o=>{ if(o.isMesh && /Body|Hand|Wrist|Neck/.test(o.name))o.visible=!$('hideBody').checked; });
}
const manifest = await fetch('./manifest.json').then(r=>{if(!r.ok)throw Error('Manifest unavailable');return r.json();});
for(const [i,item] of manifest.cases.entries()) {
  const option=document.createElement('option');option.value=i;option.textContent=item.title;$('case').appendChild(option);
  const card=document.createElement('button');card.className='card';
  const img=document.createElement('img');img.src=item.thumbnail;img.alt=item.title;img.loading='lazy';
  const label=document.createElement('span');label.textContent=item.title;card.append(img,label);
  card.onclick=()=>{$('case').value=i;load(i);};$('cards').appendChild(card);
}
async function load(index) {
  const ticket=++request,item=manifest.cases[index];$('status').textContent='Đang tải bản thử…';
  if(model){scene.remove(model);dispose(model);model=null;}
  $('notes').textContent=item.notes;
  $('glb').href=item.glb;$('report').href=item.report;
  try {
    const gltf=await new GLTFLoader().loadAsync(item.glb);
    if(ticket!==request){dispose(gltf.scene);return;}
    model=gltf.scene;scene.add(model);visibility();view('front');
    $('status').textContent=`${item.size_mib.toFixed(1)} MiB · ${item.warning_points === null ? "chưa đo lại va chạm" : item.warning_points + " điểm cảnh báo theo phép đo mẫu"} · chưa duyệt tích hợp`;
  } catch(error) { if(ticket===request)$('status').textContent=`Không tải được GLB: ${error.message}`; }
}
$('case').onchange=()=>load(Number($('case').value));$('hideBody').onchange=visibility;
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
view('front');load(0);
renderer.setAnimationLoop(()=>{if(!document.hidden){controls.update();renderer.render(scene,camera);}});
window.addEventListener('pagehide',()=>{if(model)dispose(model);controls.dispose();renderer.dispose();});
