import importlib.util
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'execution_kit'))
from test_control import FakeBus,CAL,INITIAL
spec=importlib.util.spec_from_file_location('candidate_core',HERE/'control_core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

class TestStartupFix(unittest.TestCase):
    def test_goal_write_side_effect_is_detected_and_cleanup_armed(self):
        class SideEffectBus(FakeBus):
            def sync_write(self,name,values,**kwargs):
                super().sync_write(name,values,**kwargs)
                if name=='Goal_Position':self.data['Torque_Enable']=[1]*6
        bus=SideEffectBus();arm=core.BoundedArm('fake',CAL,True,lambda:bus);arm.connect()
        with self.assertRaises(RuntimeError):arm.enable_aligned(INITIAL)
        self.assertTrue(arm.may_be_enabled)
        self.assertNotIn('write_Torque_Enable',bus.events)
        self.assertTrue(arm.hold()['all_torque_enabled'])
    def test_initial_write_failure_still_arms_cleanup(self):
        class BrokenBus(FakeBus):
            def sync_write(self,*args,**kwargs):raise ConnectionError('TX state unknown')
        bus=BrokenBus();arm=core.BoundedArm('fake',CAL,True,lambda:bus);arm.connect()
        with self.assertRaises(ConnectionError):arm.enable_aligned(INITIAL)
        self.assertTrue(arm.may_be_enabled)

if __name__=='__main__':unittest.main(verbosity=2)
