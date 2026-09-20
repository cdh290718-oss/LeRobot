from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
INPUT=PROJECT/'cui_local_computer_files/results/return1'
KIT=ROOT.parent/'execution_kit'
MOTORS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
D=360/4095
results=[];traces=[];pictures=[]
def stats(a):
 a=np.asarray(a,float)
 return {'count':a.size,'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),'max':float(np.max(a))} if a.size else None
for f in sorted(INPUT.glob('*.zip')):
 with zipfile.ZipFile(f) as z:
  assert z.testzip() is None
  r=json.loads(z.read('summary.json'));rows=[json.loads(s) for s in z.read('control.jsonl').decode().splitlines()]
  mismatches=[n for n,h in r['code_sha256'].items() if hashlib.sha256((KIT/n).read_bytes()).hexdigest()!=h]
  assert not mismatches
  item={'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'crc':'passed','code_sha_match':True,
        'mode':r['mode'],'status':r['status'],'error':r.get('error'),'control_ticks':len(rows),'bus':r['bus'],
        'stop_hold':r.get('stop_hold'),'initial_raw':r.get('execution_initial_raw',r.get('initial_raw')),
        'elapsed_seconds':r.get('active_seconds'),'last_logged_elapsed_seconds':rows[-1]['elapsed_s'] if rows else None}
  if 'inference_summary.json' in z.namelist():item['model']=json.loads(z.read('inference_summary.json'))
  if 'chunks.jsonl' in z.namelist():
   chunks=[json.loads(s) for s in z.read('chunks.jsonl').decode().splitlines()]
   item['pipeline_ms']=stats([s['timings']['pipeline_ms'] for s in chunks])
   item['predict_chunk_ms']=stats([s['timings']['chunk_ms'] for s in chunks])
  if 'predictions.npz' in z.namelist():
   import io
   with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as a:
    assert a['actions'].shape[1:]==(50,6) and np.isfinite(a['actions']).all()
    item['finite_predictions']=True
  if rows:
   initial=np.array(r['execution_initial_raw']);measured=np.array([s['raw_present'] for s in rows]);command=np.array([s['raw_command'] for s in rows])
   prior=np.vstack([initial,command[:-1]])
   item.update(control_hz=1000/np.mean([s['interval_ms'] for s in rows]),
     interval_ms=stats([s['interval_ms'] for s in rows]),work_ms=stats([s['work_ms'] for s in rows]),
     action_observation_age_ms=stats([s['observation_age_ms'] for s in rows]),
     camera_age_ms={k:stats([s['camera_age_ms'][k] for s in rows]) for k in ('1','3')},
     tracking_error_deg_per_joint_max=np.max(abs(measured-prior),axis=0)[:5].__mul__(D).tolist(),
     measured_excursion_deg_per_joint_max=(np.max(abs(measured-initial),axis=0)[:5]*D).tolist(),
     command_excursion_deg_per_joint_max=(np.max(abs(command-initial),axis=0)[:5]*D).tolist(),
     last_measured_displacement_deg=((measured[-1]-initial)[:5]*D).tolist(),
     last_command_displacement_deg=((command[-1]-initial)[:5]*D).tolist(),
     clipped_percent=(np.mean([s['clipped'] for s in rows],axis=0)*100).tolist(),
     max_raw_step=int(np.max(abs(command-prior))),gripper_command_fixed=bool(np.all(command[:,5]==initial[5])),
     expired_plan_ticks=sum(s['plan_expired_hold'] for s in rows),
     command_bounds_pass=bool(np.all(command>=r['execution_envelope']['min']) and np.all(command<=r['execution_envelope']['max'])))
   assert item['max_raw_step']<=1 and item['gripper_command_fixed'] and item['command_bounds_pass']
   if r['mode']=='execute':
    hold=np.array(r['stop_hold']['raw_goal'])
    item['hold_minus_last_command_deg']=((hold-command[-1])[:5]*D).tolist()
    traces.append((f.name[9:15],rows,initial,item))
    panels=[]
    for name in ('camera_before.jpg','camera_last_observation.jpg'):
     data=z.read(name);(ROOT/(f.name[:19]+'_'+name)).write_bytes(data)
     import io
     panels.append(Image.open(io.BytesIO(data)).convert('RGB').resize((640,240)))
    pictures.append((f.name[9:15],panels))
  results.append(item)
(ROOT/'ANALYSIS.json').write_text(json.dumps({'motor_order':MOTORS,'runs':results},ensure_ascii=False,indent=2))
fig,axs=plt.subplots(2,2,figsize=(12,7))
for ax,(name,rows,initial,item) in zip(axs.flat,traces):
 t=[s['elapsed_s'] for s in rows]
 ax.plot(t,[(s['raw_command'][2]-initial[2])*D for s in rows],label='Command')
 ax.plot(t,[(s['raw_present'][2]-initial[2])*D for s in rows],label='Measured')
 ax.set(title=f"{name} / {item['status']}",xlabel='Time since enable (s)',ylabel='Elbow change from start (deg)')
 ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'elbow_tracking.png',dpi=160);plt.close(fig)
canvas=Image.new('RGB',(1280,270*len(pictures)),'white');draw=ImageDraw.Draw(canvas)
for i,(name,panels) in enumerate(pictures):
 draw.text((10,270*i+8),name+'  BEFORE (external | wrist)                         LAST OBSERVATION (external | wrist)',fill='black')
 for j,p in enumerate(panels):canvas.paste(p,(640*j,270*i+30))
canvas.save(ROOT/'camera_comparison.jpg',quality=90)
for r in results:
 print(r['file'][9:15],r['status'],'ticks',r['control_ticks'],'hz',round(r.get('control_hz',0),2),'pipeline',r.get('pipeline_ms'))
