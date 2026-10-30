import json,os,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import control

class HotfixTests(unittest.TestCase):
    def test_interrupt_terminates_only_owned_process_group(self):
        child=MagicMock();child.pid=321;child.wait.side_effect=[KeyboardInterrupt(),-15]
        with tempfile.TemporaryDirectory() as folder,patch.object(control.subprocess,'Popen',return_value=child),patch.object(control.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):control.command(['python','-c','pass'],Path(folder)/'test.log',env={})
            kill.assert_called_once_with(321,signal.SIGTERM)
            self.assertIn('cached files retained',(Path(folder)/'test.log').read_text())
    def test_run_reuses_local_huggingface_assets_without_setup_downloads(self):
        with patch.dict(os.environ,{},clear=True):
            env=control.inference_environment('run',Path('/tmp/state'))
            self.assertEqual(env['HF_HUB_OFFLINE'],'1');self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
            self.assertNotIn('HF_HUB_OFFLINE',os.environ)
            self.assertNotIn('HF_HUB_OFFLINE',control.inference_environment('all',Path('/tmp/state')))
    def test_stubborn_child_is_killed_and_reaped(self):
        child=MagicMock();child.pid=322;child.wait.side_effect=[subprocess.TimeoutExpired('python',10),-9]
        with patch.object(control.os,'killpg') as kill:control.terminate_group(child)
        self.assertEqual([x.args for x in kill.call_args_list],[(322,signal.SIGTERM),(322,signal.SIGKILL)])
    def test_low_physical_disk_blocks_before_launch(self):
        usage=MagicMock();usage.free=2*1024**3
        with tempfile.TemporaryDirectory() as folder,patch.object(control.shutil,'disk_usage',return_value=usage),patch.object(control.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_HOST_DISK'):
                control.command(['python','-c','pass'],Path(folder)/'test.log',env={'MEMENTO_HAIRBENCH_HOST_DISKS':'["/mnt/d"]'})
            launch.assert_not_called()
    def test_physical_disk_location_is_explicit(self):
        usage=MagicMock();usage.free=8*1024**3
        with patch.dict(os.environ,{'MEMENTO_HAIRBENCH_HOST_DISKS':'["/mnt/d/"]'}),patch.object(control.shutil,'disk_usage',return_value=usage):
            self.assertEqual(control.host_disks(),[{'path':'/mnt/d/','free_gib':8.}])
    def test_main_interrupt_is_reported_as_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'out';state=root/'state';request=root/'request.json'
            request.write_text(json.dumps({'output_linux':str(out),'state_linux':str(state),'mode':'setup','models':['hunyuan-mini'],'host_disks_linux':['/mnt/d']}))
            hw={'gpus':[{'total_mib':6144,'free_mib':6000}],'free_disk_gib':30,'available_ram_mib':8000,'physical_host_disk_checked':True}
            with patch.object(sys,'argv',['control','--request',str(request)]),patch.object(control,'probe',return_value=hw),patch.object(control,'setup',side_effect=KeyboardInterrupt()),patch.dict(os.environ,{},clear=False):
                self.assertEqual(control.main(),130)
            report=json.loads((out/'report.json').read_text());self.assertTrue(report['interrupted']);self.assertFalse(report['any_inference_succeeded']);self.assertEqual(report['runs'][0]['status'],'INTERRUPTED')
    def test_main_uses_host_capacity_instead_of_virtual_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);request=root/'request.json';out=root/'out'
            request.write_text(json.dumps({'output_linux':str(out),'state_linux':str(root/'state'),'mode':'setup','models':['hunyuan-mini'],'host_disks_linux':['/mnt/d']}))
            hw={'gpus':[{'total_mib':6144}],'free_disk_gib':8.11,'virtual_disk_free_gib':900,'available_ram_mib':8000,'physical_host_disk_checked':True}
            with patch.object(sys,'argv',['control','--request',str(request)]),patch.object(control,'probe',return_value=hw),patch.object(control,'setup') as setup,patch.dict(os.environ,{},clear=False):
                self.assertEqual(control.main(),2);setup.assert_not_called()
            self.assertIn('10 GiB',(out/'report.json').read_text())

if __name__=='__main__':unittest.main()
