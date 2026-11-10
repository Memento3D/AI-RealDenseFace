"""Versioned RDF fitting policy with objective acceptance and retained camera states.

No identity, pose or camera is borrowed from another subject. Vendor stays intact.
"""
import math
import numpy as np
import torch
from tracker import GNFlameOptimizer

POLICY='rdf_objective_acceptance_v1'

class StableRDFOptimizer(GNFlameOptimizer):
    def __init__(self,config):
        super().__init__(config)
        self.audit=[]

    def _geometry_valid(self,identity,x):
        if not torch.isfinite(identity).all() or not torch.isfinite(x).all():return False
        canonical,joints=self._calc_canonical_flame(identity)
        v,_=self._calc_posed_flame(canonical,joints,x)
        if not torch.isfinite(v).all():return False
        extent=(v.amax(dim=-2)-v.amin(dim=-2)).norm(dim=-1)
        reference=(self.v_template.amax(0)-self.v_template.amin(0)).norm()
        # Broad numerical sanity, not an identity/likeness gate. Existing gates remain.
        return bool(((extent>reference*.1)&(extent<reference*10)).all()
                    and (v.abs().amax()<reference*10000))

    def _x_system(self,vm,vpm,coords,depths,cv,dv,identity,x,pose_only=False):
        canonical,joints=self._calc_canonical_flame(identity)
        v,j,vj,jj=self.solver_ops.expression_jacobian(x,canonical,joints,self.expr_dirs,self.pose_dirs,self.lbs_weights)
        return self.solver_ops.assemble_x_system(vm,vpm,v,j,coords,depths,cv,dv,x,vj,jj,
            1.0 if pose_only else self.config.correspondence_weight,
            0.0 if pose_only else self.config.relative_depth_weight,
            0.0 if pose_only else self.config.expression_regularization,
            0.0 if pose_only else self.config.pose_regularization,self.key_vertex_ids)

    def _objective(self,vm,vpm,coords,depths,cv,dv,identity,x,origin,pose_only=False):
        if not self._geometry_valid(identity,x):return math.inf
        residual,_=self._x_system(vm,vpm,coords,depths,cv,dv,identity,x,pose_only)
        value=residual.double().square().sum()
        if not pose_only:value=value+((identity-origin).double()*self.config.identity_regularization).square().sum()
        result=float(value.item())
        return result if math.isfinite(result) else math.inf

    def _accepted_step(self,base,jacobian,residual,damping,score,current):
        for _ in range(3):
            delta=self.solver_ops.solve_normal_equation(self.solver_context,jacobian,residual,damping)
            if torch.isfinite(delta).all():
                for scale in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
                    candidate=base+scale*delta
                    value=score(candidate)
                    if math.isfinite(value) and value<current:
                        return candidate,value,max(damping/2,1e-8),True
            damping*=10
        return base,current,damping,False

    def _register_pose(self,vm,vpm,coords,depths,cv,dv,canonical,joints,init_x=None,num_iters=None):
        # Canonical geometry here is supplied by the vendor camera-search contract.
        shape=(self.dim_x,) if coords.ndim==2 or (coords.ndim==3 and coords.shape[0]==vm.shape[0]) else (coords.shape[0],self.dim_x)
        x=torch.zeros(shape,device=self.device) if init_x is None else init_x.to(self.device).reshape(shape).clone()
        dims=torch.tensor([self.pose_start+i for i in range(3)]+[self.trans_start+i for i in range(3)],device=self.device)
        def system(state):
            v,j,vj,jj=self.solver_ops.expression_jacobian(state,canonical,joints,self.expr_dirs,self.pose_dirs,self.lbs_weights)
            return self.solver_ops.assemble_x_system(vm,vpm,v,j,coords,depths,cv,dv,state,vj,jj,1.,0.,0.,0.,self.key_vertex_ids)
        def score(values):
            candidate=x.clone();candidate[...,dims]=values
            if not torch.isfinite(candidate).all():return math.inf
            v,_=self.solver_ops.forward_vertices(candidate,canonical,joints,self.expr_dirs,self.pose_dirs,self.lbs_weights)
            reference=(self.v_template.amax(0)-self.v_template.amin(0)).norm()
            if not torch.isfinite(v).all() or v.abs().amax()>reference*10000:return math.inf
            r,_=system(candidate);value=float(r.double().square().sum().item())
            return value if math.isfinite(value) else math.inf
        damping=self.config.pose_damping
        before=score(x[...,dims]);current=before
        if not math.isfinite(current):raise ValueError('FIT_DIVERGED: invalid pose initialization')
        accepted=0
        for _ in range(self.config.pose_registration_iterations if num_iters is None else num_iters):
            r,j=system(x)
            values,current,damping,ok=self._accepted_step(x[...,dims],j[...,dims],r,damping,score,current)
            if not ok:break
            x[...,dims]=values;accepted+=1
        self.audit.append({'phase':'pose','initial_objective':before,'final_objective':current,'accepted_steps':accepted})
        return x

    def _register(self,vm,vpm,coords,depths,cv=None,dv=None,init_identity=None,init_x=None,num_iters=None):
        shape=(self.dim_x,) if coords.ndim==2 or (coords.ndim==3 and coords.shape[0]==vm.shape[-3]) else (coords.shape[0],self.dim_x)
        origin=torch.zeros(300,device=self.device) if init_identity is None else init_identity.to(self.device).reshape(300).clone()
        identity=origin.clone()
        x=torch.zeros(shape,device=self.device) if init_x is None else init_x.to(self.device).reshape(shape).clone()
        score=lambda b,state:self._objective(vm,vpm,coords,depths,cv,dv,b,state,origin)
        before=score(identity,x);current=before
        if not math.isfinite(current):raise ValueError('FIT_DIVERGED: invalid registration initialization')
        dx=di=self.config.damping;accepted=0
        for _ in range(self.config.registration_iterations if num_iters is None else num_iters):
            old=current
            r,j=self._x_system(vm,vpm,coords,depths,cv,dv,identity,x)
            x,current,dx,ok=self._accepted_step(x,j,r,dx,lambda state:score(identity,state),current);accepted+=int(ok)
            canonical,joints=self._calc_canonical_flame(identity)
            v,joint,vj,jj=self.solver_ops.identity_jacobian(x,canonical,joints,self.joints_dirs,self.shape_dirs,self.expr_dirs,self.pose_dirs,self.lbs_weights)
            r,j=self.solver_ops.assemble_identity_system(vm,vpm,v,joint,coords,depths,cv,dv,identity-origin,vj,jj,
                self.config.correspondence_weight,self.config.relative_depth_weight,self.config.identity_regularization,self.key_vertex_ids)
            identity,current,di,ok=self._accepted_step(identity,j,r,di,lambda beta:score(beta,x),current);accepted+=int(ok)
            if old-current<=max(1e-12,abs(old)*1e-7):break
        self.audit.append({'phase':'registration','initial_objective':before,'final_objective':current,'accepted_steps':accepted})
        return identity,x

    def _register_wi_fov_search(self,width,height,coords,depths,cv,dv,init_identity=None,init_x=None,pose_init=True):
        origin=torch.zeros(300,device=self.device) if init_identity is None else init_identity.to(self.device)
        canonical,joints=self._calc_canonical_flame(origin)
        candidates={}
        def evaluate(fov):
            if fov in candidates:return
            camera=self._build_config_camera(width,height,fov);vm,vpm=self._single_camera_matrices(camera)
            try:
                pose=self._register_pose(vm,vpm,coords,depths,cv,dv,canonical,joints,init_x=init_x) if pose_init else init_x
                beta,x=self._register(vm,vpm,coords,depths,cv,dv,init_identity=origin,init_x=pose,num_iters=self.config.fov_registration_iterations)
                energy=self._objective(vm,vpm,coords,depths,cv,dv,beta,x,origin)
                if math.isfinite(energy):candidates[fov]=(energy,beta.clone(),x.clone(),pose.clone())
                self.audit.append({'phase':'camera','fov':fov,'objective':energy if math.isfinite(energy) else None})
            except ValueError as exc:
                if not str(exc).startswith('FIT_DIVERGED:'):raise
                self.audit.append({'phase':'camera','fov':fov,'objective':None,'rejected':str(exc)})
        grid=np.linspace(self.config.fov_y_min,self.config.fov_y_max,5).tolist()
        for fov in grid:evaluate(fov)
        if not candidates:raise ValueError('FIT_DIVERGED: no valid camera candidate')
        best=min(candidates,key=lambda fov:candidates[fov][0])
        spacing=(self.config.fov_y_max-self.config.fov_y_min)/4
        for fov in (max(self.config.fov_y_min,best-spacing/2),min(self.config.fov_y_max,best+spacing/2)):evaluate(float(fov))
        best=min(candidates,key=lambda fov:candidates[fov][0]);energy,beta,x,pose=candidates[best]
        camera=self._build_config_camera(width,height,best);vm,vpm=self._single_camera_matrices(camera)
        refined_b,refined_x=self._register(vm,vpm,coords,depths,cv,dv,init_identity=beta,init_x=x)
        # Compare refinement using the original common objective, not a new identity origin.
        if self._objective(vm,vpm,coords,depths,cv,dv,refined_b,refined_x,origin)<energy:beta,x=refined_b,refined_x
        self.audit.append({'phase':'selected_camera','fov':best,'at_search_boundary':best in (self.config.fov_y_min,self.config.fov_y_max)})
        return best,beta,x,pose
