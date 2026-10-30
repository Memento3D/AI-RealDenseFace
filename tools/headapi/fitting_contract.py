"""Variable-view RDF joint registration contract; never repeat missing observations."""

def fit_shared_identity(optimizer,vm,vpm,coords,depths,cv,dv,initial,states,count):
    if count not in (1,2,3):raise ValueError('Expected one to three real views')
    if tuple(vm.shape[:2])!=(count,1) or tuple(vpm.shape[:2])!=(count,1):raise ValueError('Independent per-view cameras required')
    if any(a.shape[0]!=count for a in (coords,depths,cv,dv,states)):raise ValueError('Observation/view count mismatch')
    if count==1:return initial,states
    return optimizer._register(vm,vpm,coords,depths,cv,dv,init_identity=initial,init_x=states,num_iters=10)
