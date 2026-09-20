"""Validate final simulation against delivered code, then create the small Windows kit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
KIT=ROOT/'execution_kit'
SIM=ROOT/'developer_simulation_final_bf16'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

r=json.loads((SIM/'summary.json').read_text())
m=json.loads((SIM/'inference_summary.json').read_text())
assert r['status']=='completed' and r['mode']=='simulate'
assert m['status']=='stopped' and m['model']['strict_load'] and m['model']['precision']=='bf16'
assert r['worker_exitcode']==0 and r['stop_hold']['goal_readback_confirmed']
for name,expected in r['code_sha256'].items():assert sha(KIT/name)==expected,name
rows=[json.loads(line) for line in (SIM/'control.jsonl').read_text().splitlines()]
commands=np.array([row['raw_command'] for row in rows])
initial=np.array(r['execution_initial_raw'])
assert len(rows)>200
assert (np.abs(np.diff(np.vstack([initial,commands]),axis=0))<=1).all()
assert (commands[:,5]==initial[5]).all()
assert (np.abs(commands[:,:5]-initial[:5])*360/4095<=8).all()
lo=np.array(r['execution_envelope']['min']);hi=np.array(r['execution_envelope']['max'])
assert ((commands>=lo)&(commands<=hi)).all()
with np.load(SIM/'predictions.npz',allow_pickle=False) as z:
    assert z['actions'].shape[1:]==(50,6) and np.isfinite(z['actions']).all()
with zipfile.ZipFile(SIM.with_name(SIM.name+'_return.zip')) as z:assert z.testzip() is None
text=(ROOT/'TEST_RESULTS.txt').read_text()
assert re.search(r'Ran 18 tests',text) and text.rstrip().endswith('OK'),text
(KIT/'TEST_RESULTS.txt').write_text(text,encoding='utf-8')
validation={
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'unit_and_integration_tests':{'count':18,'status':'passed',
       'covers':['all unauthorized opcodes blocked','actual Feetech SDK 1.0.0 exact packet authorization',
                 'one-shot write permit','startup alignment readback before torque',
                 'alignment failure and requested stop do not enable torque','wrong mode/existing torque rejection',
                 'readonly main sends no writes','normal completion holds before teardown',
                 'injected serial fault holds before teardown','release does not load model or cameras',
                 'integer slew/excursion/calibration limits and fixed gripper','stale action skipping/expiry',
                 'normalization matches installed LeRobot implementation','shared memory latest observation']},
    'sdk_note':'Developer SDK unpacked in /tmp; normalization test injects that actual SDK module past distribution-metadata checks. Production runtime has no such bypass.',
    'real_model_simulation':{'status':r['status'],'gpu':m['model']['gpu'],'precision':m['model']['precision'],
        'model_checkpoint':m['model']['checkpoint'],'strict_load':True,'warmup_chunks':m['warmup_chunks'],
        'chunks':m['chunks'],'active_seconds':r['active_seconds'],'control_ticks':r['control_ticks'],
        'timing':r['timing'],'memory':m['memory'],'hardware_write_packets':0,
        'all_actions_finite':True,'all_limiter_invariants_passed':True,'return_zip_crc':'passed'},
    'windows_hardware_execution':'NOT performed on developer machine; requires user return',
    'powershell_runtime_test':'NOT performed on Linux; no PowerShell interpreter available',
    'simulation_limits':'Fixed prerecorded images, instantaneous fake position following; no dynamics/collision/grasp validation.',
    'code_sha256':{f.name:sha(f) for f in KIT.iterdir() if f.suffix in ('.py','.ps1')},
    'previous_engine_and_readonly_adapter_unchanged':all(sha(KIT/name)==sha(ROOT.parent/'lerobot_live_readonly_20260919/live_readonly_kit'/name) for name in ('live_probe.py','readonly_bus.py')),
}
write(KIT/'DEVELOPER_VALIDATION.json',validation)
files=sorted(f for f in KIT.iterdir() if f.is_file() and f.suffix in ('.py','.ps1','.json','.md','.txt') and f.name!='MANIFEST.json')
write(KIT/'MANIFEST.json',{'files':{f.name:{'size_bytes':f.stat().st_size,'sha256':sha(f)} for f in files}})
files.append(KIT/'MANIFEST.json')
archive=ROOT/'lerobot_bounded_execution_kit.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,Path('execution_kit')/f.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for f in files:assert hashlib.sha256(z.read('execution_kit/'+f.name)).hexdigest()==sha(f)
write(ROOT/'DELIVERY.json',{'file':str(archive),'size_bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'crc':'passed'})
print(json.dumps({'archive':str(archive),'size_bytes':archive.stat().st_size,'tests':18,'simulation_ticks':len(rows)},indent=2))
