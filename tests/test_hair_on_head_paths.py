"""Project-layout regressions. No Windows execution is claimed by these tests."""
import os,sys,subprocess,tempfile,unittest,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from memento3d_hair_on_head.entrypoint import project_root

class RootLayoutTests(unittest.TestCase):
    def test_root_comes_from_module_location(self):
        self.assertEqual(project_root(),ROOT)
    def test_expected_flat_paths_exist(self):
        for name in ['scripts/run_on_head.py','scripts/serve_hair_on_head.py','configs/hair-on-head.json','requirements/hair-on-head-cpu.txt','viewer/hair-on-head/index.html','src/memento3d_hair_on_head/runtime/pipeline.py']:
            self.assertTrue((ROOT/name).is_file(),name)
    def test_launcher_uses_flat_entrypoint_and_scoped_manifest(self):
        s=(ROOT/'RUN-HAIR-ON-HEAD.ps1').read_text()
        self.assertIn('$project=$PSScriptRoot',s)
        self.assertIn('"scripts\\run_on_head.py"',s)
        self.assertIn('"hair-on-head-code.json"',s)
        self.assertNotIn('"hair-on-head\\scripts\\',s)
        self.assertIn('requirements\\hair-on-head-cpu.txt',s)
    def test_python_cli_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory(prefix='foreign cwd ') as folder:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_on_head.py'),'--help'],cwd=folder,env=os.environ.copy(),capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--bench',p.stdout);self.assertIn('--workspace',p.stdout)
    def test_profile_and_viewer_are_relative_to_actual_project(self):
        s=(ROOT/'src/memento3d_hair_on_head/runtime/pipeline.py').read_text()
        self.assertIn("ROOT/'configs/hair-on-head.json'",s)
        self.assertIn("ROOT/'viewer/hair-on-head'",s)
    def test_viewer_wrapper_points_to_project_src(self):
        s=(ROOT/'scripts/serve_hair_on_head.py').read_text()
        self.assertIn("PROJECT/'src'",s)
        self.assertIn('memento3d_hair_on_head.entrypoint',s)
    def test_manifest_covers_feature_owned_paths_only(self):
        p=ROOT/'hair-on-head-code.json'
        if not p.exists():self.skipTest('Manifest generated during packaging')
        m=json.loads(p.read_text())
        for e in m['files']:
            name=e['path']
            self.assertNotIn('src/memento3d_head/',name)
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),e['sha256'])
            self.assertFalse(name.startswith('hair-on-head/'))

if __name__=='__main__':unittest.main()
