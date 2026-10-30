import test from 'node:test';
import assert from 'node:assert/strict';
import {SelectionCoordinator,inventorySignature} from '../assets/hair-studio/selection.mjs';
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise};};
test('late old load never replaces newest hairstyle',async()=>{const c=new SelectionCoordinator(),slow=deferred();let displayed=null;let discarded=[];const first=c.choose(()=>slow.promise,x=>displayed=x,x=>discarded.push(x));await c.choose(async()=> 'new',x=>displayed=x);slow.resolve('old');assert.equal(await first,false);assert.equal(displayed,'new');assert.deepEqual(discarded,['old']);});
test('removal invalidates a pending model',async()=>{const c=new SelectionCoordinator(),d=deferred();let committed=false;const p=c.choose(()=>d.promise,()=>committed=true);c.invalidate();d.resolve({hair:'deleted'});assert.equal(await p,false);assert.equal(committed,false);});
test('inventory revision changes invalidate UI signature',()=>{const a=[{id:'same',revision:'old',state:'available',name:'Hair'}];assert.notEqual(inventorySignature(a),inventorySignature([{...a[0],revision:'new'}]));});
