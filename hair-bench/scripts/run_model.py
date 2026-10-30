"""One GPU model per process. No face generation, recoloring or template fitting."""
import argparse,json,sys,time,types,traceback
from pathlib import Path

def install_cpu_mc():
    # TripoSR expects torchmcubes XYZ ordering before its own ZYX swap.
    # Use a scoped CPU Lewiner extractor to avoid a CUDA toolkit build on a 6GB laptop.
    import torch
    from skimage.measure import marching_cubes
    module=types.ModuleType('torchmcubes')
    def mc(level,threshold):
        v,f,_,_=marching_cubes(level.detach().float().cpu().numpy(),float(threshold))
        return torch.tensor(v[:,::-1].copy(),dtype=torch.float32),torch.tensor(f.copy(),dtype=torch.int64)
    module.marching_cubes=mc;sys.modules['torchmcubes']=module

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--subject',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);result={'model':a.model,'status':'STARTED','native_only':True,'ready_for_machine':False};start=time.perf_counter()
    try:
        import torch
        from PIL import Image
        if not torch.cuda.is_available():raise RuntimeError('CUDA not visible; CPU fallback disabled for this benchmark')
        torch.cuda.reset_peak_memory_stats();torch.manual_seed(42);sys.path.insert(0,str(a.repo));s=json.loads(a.subject.read_text())
        result.update(gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability()),face_sha256=s['face_sha256'])
        if a.model=='triposr':
            install_cpu_mc();from tsr.system import TSR
            import tsr.models.tokenizers.image as tokenizer
            original_download=tokenizer.hf_hub_download
            def locked_config(repo_id,filename,**kw):
                if repo_id=='facebook/dino-vitb16' and filename=='config.json':return str(a.weights/'dino-config/config.json')
                return original_download(repo_id,filename,**kw)
            tokenizer.hf_hub_download=locked_config
            model=TSR.from_pretrained(str(a.weights),config_name='config.yaml',weight_name='model.ckpt');model.renderer.set_chunk_size(1024);model.to('cuda')
            with torch.inference_mode():
                code=model([Image.open(s['hair_rgb']).convert('RGB')],device='cuda');mesh=model.extract_mesh(code,False,resolution=128)[0]
            result['settings']={'mc_resolution':128,'chunk_size':1024,'surface_extraction':'CPU skimage Lewiner, adapted axis ordering','input':'hair-only RGB, gray background'}
        elif a.model=='hunyuan-mini':
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            pipe=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(a.weights),subfolder='hunyuan3d-dit-v2-mini',device='cpu',dtype=torch.float16,use_safetensors=True,variant='fp16')
            pipe.enable_model_cpu_offload(device='cuda');pipe.device=torch.device('cuda')
            with torch.inference_mode():mesh=pipe(image=Image.open(s['hair_rgba']).convert('RGBA'),num_inference_steps=20,octree_resolution=128,num_chunks=1024,mc_algo='mc',generator=torch.Generator(device='cuda').manual_seed(42),output_type='trimesh')[0]
            if mesh is None:raise RuntimeError('Model returned no mesh')
            result['settings']={'octree_resolution':128,'chunks':1024,'steps':20,'cpu_offload':True,'shape_only':True,'input':'hair-only RGBA','runtime_execution_device':'cuda after CPU offload hooks'}
        else:raise ValueError('DiffLocks is gated/experimental: use its audited upstream inference only after checkpoints and CUDA-kernel setup are verified; no fake substitute generated.')
        mesh.export(a.output/'native.glb');mesh.export(a.output/'native.ply');torch.cuda.synchronize()
        result.update(status='INFERRED_NATIVE_MESH',native_glb=str(a.output/'native.glb'),triangles=len(mesh.faces),watertight=bool(mesh.is_watertight),
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            memory_scope='PyTorch allocator only, not whole GPU/driver peak',fitted_to_personal_head=False)
    except Exception as exc:
        result.update(status='FAILED',error=str(exc),oom='out of memory' in str(exc).lower());(a.output/'traceback.txt').write_text(traceback.format_exc())
    result['seconds']=time.perf_counter()-start;(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['status']=='INFERRED_NATIVE_MESH' else 2
if __name__=='__main__':raise SystemExit(main())
