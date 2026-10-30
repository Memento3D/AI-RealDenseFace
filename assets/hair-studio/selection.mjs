// Last-request-wins: a slower old hair load cannot replace the latest selection.
export class SelectionCoordinator {
  constructor(){this.epoch=0;}
  invalidate(){return ++this.epoch;}
  isCurrent(epoch){return epoch===this.epoch;}
  async choose(load,commit,discard=()=>{}){
    const epoch=this.invalidate();
    const result=await load(()=>this.isCurrent(epoch));
    if(!this.isCurrent(epoch)){if(result)discard(result);return false;}
    if(result)commit(result);return Boolean(result);
  }
}
export function inventorySignature(assets){return assets.map(a=>[a.id,a.revision,a.state,a.name].join('|')).join('\n');}
