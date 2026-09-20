from pathlib import Path
import csv,hashlib,io,json,zipfile
from datetime import datetime,timezone,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
FILES={'inputs':'20260919_121454_372_Inputs_bf16_return.zip','live':'20260919_121541_727_Live_bf16_return.zip'}

def quantile(v):
 v=np.asarray(v,dtype=float)
 return dict(count=len(v),min=float(v.min()),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),max=float(v.max()),mean=float(v.mean()))

result={}
for tag,name in FILES.items():
 path=PROJECT/'cui_local_computer_files/results'/name
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  summary=json.loads(z.read('summary.json'))
  assert summary['status']=='passed'
  assert not summary['motor_action_execution']
  for f,h in summary['source_sha256'].items():assert hashlib.sha256((ROOT.parent/'live_readonly_kit'/f).read_bytes()).hexdigest()==h
  iterations=[json.loads(s) for s in z.read('iterations.jsonl').decode().splitlines()]
  states=[json.loads(s) for s in z.read('states.jsonl').decode().splitlines()]
  assert len(iterations)==summary['observations'] and len(states)==summary['state_reads']['count']
  times=np.array([s['timestamp'] for s in states]);intervals=np.diff(times)*1000
  statearray=np.array([s['state'] for s in states]);assert np.isfinite(statearray).all()
  entry={'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'crc_and_code_hash':'passed',
         'summary':summary,'actual_state_poll_hz':float((len(times)-1)/(times[-1]-times[0])),
         'state_poll_interval_ms':quantile(intervals),'state_span_per_joint':np.ptp(statearray,axis=0).tolist()}
  rows=list(csv.DictReader(io.StringIO(z.read('gpu_telemetry.csv').decode())))
  # Infer the telemetry's local UTC offset from the explicitly UTC run creation time and first sample.
  utc_start=datetime.fromisoformat(summary['created_utc']).timestamp()
  first_wall=datetime.strptime(rows[0]['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
  offset=round((first_wall-utc_start)/3600)*3600
  assert abs(first_wall-offset-utc_start)<10
  gpu=[]
  for row in rows:
   row={k.strip():v.strip() for k,v in row.items()}
   t=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()-offset
   gpu.append({'t':t,'pstate':row['pstate'],**{k:float(row[k].split()[0]) for k in ('utilization.gpu [%]','memory.used [MiB]','power.draw [W]','temperature.gpu','clocks.current.graphics [MHz]','clocks.current.memory [MHz]')}})
  t0=iterations[0]['wall_time_ns']/1e9
  t1=iterations[-1]['wall_time_ns']/1e9+iterations[-1]['loop_work_ms']/1000
  active=[g for g in gpu if t0<=g['t']<=t1]
  entry['gpu_active_period']={k:quantile([g[k] for g in active]) for k in active[0] if k not in ('t','pstate')}
  entry['gpu_telemetry_samples']=len(gpu)
  entry['telemetry_utc_offset_hours_inferred']=offset/3600
  if tag=='live':
   with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as n:
    pred=n['actions'];inp=n['states']
   assert pred.shape==(76,50,6) and np.isfinite(pred).all()
   np.testing.assert_allclose(inp,np.array([r['state'] for r in iterations]),rtol=1e-6)
   np.testing.assert_allclose(pred[:,0,:],np.array([r['first_predicted_action'] for r in iterations]),rtol=1e-6)
   delta=pred[:,0,:]-inp
   entry['action_validation']={'shape':list(pred.shape),'finite':True,
     'first_step_abs_delta_p95_per_joint':np.percentile(np.abs(delta),95,axis=0).tolist(),
     'first_step_abs_delta_max_per_joint':np.max(np.abs(delta),axis=0).tolist(),
     'first_step_std_per_joint':pred[:,0,:].std(axis=0).tolist()}
   entry['prediction_loop_hz']=len(iterations)/summary['elapsed_capture_seconds']
   entry['per_call_latency_ms']=quantile([r['chunk_ms'] for r in iterations])
   blocks=[]
   for start in range(0,60,10):
    group=[r for r in iterations if start<=r['elapsed_s']<start+10]
    ggroup=[g for g in active if t0+start<=g['t']<t0+start+10]
    blocks.append({'seconds':[start,start+10],'calls':len(group),'chunk_ms':quantile([r['chunk_ms'] for r in group]),
       'gpu_util_mean':float(np.mean([g['utilization.gpu [%]'] for g in ggroup])),
       'graphics_mhz_mean':float(np.mean([g['clocks.current.graphics [MHz]'] for g in ggroup]))})
   entry['ten_second_blocks']=blocks
  result[tag]=entry
(ROOT/'ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:{a:v[a] for a in ('actual_state_poll_hz','state_poll_interval_ms','state_span_per_joint','gpu_active_period')} for k,v in result.items()},indent=2))
print(json.dumps(result['live']['ten_second_blocks'],indent=2))
print(json.dumps(result['live']['action_validation'],indent=2))
