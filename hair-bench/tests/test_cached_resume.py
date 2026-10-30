import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import finalize_triposr_cache as f

class CachedResumeTests(unittest.TestCase):
    def test_link_does_not_duplicate_weights(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'blob';src.write_bytes(b'checkpoint');dst=Path(d)/'weights/model.ckpt'
            self.assertEqual(f.link_existing(src,dst),'linked');self.assertTrue(dst.is_symlink());self.assertTrue(dst.samefile(src))
            self.assertEqual(f.link_existing(src,dst),'reused')
    def test_conflicting_existing_weight_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';b=Path(d)/'existing';a.write_bytes(b'new');b.write_bytes(b'old')
            with self.assertRaises(RuntimeError):f.link_existing(a,b)
            self.assertEqual(b.read_bytes(),b'old')
    def test_dangling_link_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'source';a.write_bytes(b'new');b=Path(d)/'broken';b.symlink_to(Path(d)/'missing')
            with self.assertRaises(RuntimeError):f.link_existing(a,b)
            self.assertTrue(b.is_symlink())
    def test_lfs_blob_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            raw=b'weights';name=hashlib.sha256(raw).hexdigest();p=Path(d)/name;p.write_bytes(raw)
            self.assertEqual(f.verify_cache_blob(p),name);p.write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):f.verify_cache_blob(p)
    def test_failed_import_does_not_create_success_record(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'state';kit=Path(d)/'kit';kit.mkdir();state.mkdir()
            python=state/'envs/triposr/bin/python';entry={'weights_repo':'stabilityai/TripoSR','weights_revision':'pin','commit':'source','dino_config_revision':'dino'}
            (kit/'models.lock.json').write_text(json.dumps({'models':{'triposr':entry}}))
            source=Path(d)/'blob';source.write_bytes(b'data')
            with patch.object(f.sys,'executable',str(python)),patch.object(f,'validate_repo'),patch.object(f,'cached_files',return_value={'config.yaml':source,'model.ckpt':source}),patch.object(f,'fetch_dino'),patch.object(f,'smoke_check',side_effect=RuntimeError('import failed')):
                with self.assertRaisesRegex(RuntimeError,'import failed'):f.complete(kit,state)
            self.assertFalse((state/'triposr.setup.json').exists())
    def test_smoke_check_uses_offline_subprocess(self):
        with patch.object(f.subprocess,'run') as run:
            f.smoke_check(Path('/repo'),Path('/scripts'),Path('/state'))
            env=run.call_args.kwargs['env'];self.assertEqual(env['HF_HUB_OFFLINE'],'1');self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
            self.assertTrue(run.call_args.kwargs['check'])
if __name__=='__main__':unittest.main()
