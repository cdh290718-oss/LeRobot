from pathlib import Path
import json,hashlib,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
KIT=ROOT/'live_readonly_kit'
log=(ROOT/'developer_tests.log').read_text()
assert 'Ran 8 tests' in log and '\nOK\n' in log
record={'created_utc':datetime.now(timezone.utc).isoformat(),'unit_tests':{'count':8,'status':'passed',
        'coverage':['all 256 motor opcodes','real Feetech SDK 1.0.0 read/write packet interception','calibration mismatch',
                    'disconnect without torque changes','stale camera/state rejection','headless HTTP preview and camera role/color order']},
        'windows_hardware_test':'not performed; user will test actual cameras/COM port',
        'powershell_execution_on_windows':'not performed on developer Linux machine',
        'python_syntax_check':'passed','replay_tests':{},'code_sha256':{}}
for precision in ('bf16','fp16'):
    directory=ROOT/f'developer_replay_{precision}'
    report=json.loads((directory/'summary.json').read_text())
    assert report['status']=='passed' and report['observations']==3
    assert not report['motor_action_execution']
    rows=[json.loads(s) for s in (directory/'iterations.jsonl').read_text().splitlines()]
    assert len(rows)==3 and all(r['chunk_ms']>0 and r['pipeline_ms']>=r['chunk_ms'] for r in rows)
    assert (directory/'gpu_telemetry.csv').stat().st_size>0
    with zipfile.ZipFile(ROOT/f'developer_replay_{precision}_return.zip') as z:assert z.testzip() is None
    record['replay_tests'][precision]={'status':'passed','gpu':report['model']['gpu'],
        'observations':3,'warmup':5,'strict_load':report['model']['strict_load'],
        'unnormalizer_probe':report['model']['unnormalizer_probe'],'timing_summary':report['timing_summary'],
        'return_zip_crc':'passed','gpu_telemetry':'recorded'}
for p in KIT.iterdir():
    if p.is_file() and p.suffix in ('.py','.ps1'):record['code_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(KIT/'DEVELOPER_VALIDATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(KIT/'TEST_RESULTS.txt').write_text(log,encoding='utf-8')
archive=ROOT/'lerobot_live_readonly_kit.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(KIT.iterdir()):
        if p.is_file():z.write(p,Path('live_readonly_kit')/p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,expected in record['code_sha256'].items():
        assert hashlib.sha256(z.read('live_readonly_kit/'+name)).hexdigest()==expected
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'crc':'passed'},indent=2))
