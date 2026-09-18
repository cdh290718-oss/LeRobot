from pathlib import Path
import json, time
import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image, ImageDraw

root=Path(r'G:\LeRobot\data\ChomCUI')
out=Path(r'D:\桌面\萌芽杯\training_audit'); out.mkdir(exist_ok=True)
report=[]; tiles=[]
for info_path in sorted(root.glob('*/meta/info.json')):
 p=info_path.parent.parent; info=json.loads(info_path.read_text())
 r={'dataset':p.name,'episodes':info.get('total_episodes'),'fps':info['fps'],'frames':info['total_frames'],'features':info['features'],'videos':[]}
 try:
  r['tasks']=pq.read_table(p/'meta/tasks.parquet').to_pandas().index.astype(str).tolist()
  data=[pq.read_table(q).to_pandas() for q in (p/'data').rglob('*.parquet')]
  r['data_rows']=sum(len(d) for d in data)
  r['nonfinite']={}
  for col in ['action','observation.state']:
   a=np.concatenate([np.stack(d[col].to_numpy()) for d in data])
   r['nonfinite'][col]=int((~np.isfinite(a)).sum());r[col+'_min']=a.min(axis=0).tolist();r[col+'_max']=a.max(axis=0).tolist()
  for v in sorted((p/'videos').rglob('*.mp4')):
   item={'path':str(v.relative_to(p))};r['videos'].append(item)
   try:
    with av.open(str(v)) as c:
     n=0
     for frame in c.decode(video=0):
      if n==0:
       im=frame.to_image();im.thumbnail((320,240));tile=Image.new('RGB',(340,275),'white');tile.paste(im,(10,25));ImageDraw.Draw(tile).text((8,5),p.name.split('_')[0]+' / '+v.parents[1].name,fill='black');tiles.append(tile)
      n+=1
     item['decoded_frames']=n;item['matches_rows']=n==r['data_rows']
   except Exception as e:item['error']=str(e)
 except Exception as e:r['error']=str(e)
 report.append(r);print(p.name,'episodes',r['episodes'],'videos',r['videos'],flush=True)
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if tiles:
 sheet=Image.new('RGB',(680,275*((len(tiles)+1)//2)), '#ddd')
 for i,t in enumerate(tiles):sheet.paste(t,((i%2)*340,(i//2)*275))
 sheet.save(out/'camera_contact_sheet.jpg')
print('DONE',out,flush=True)
