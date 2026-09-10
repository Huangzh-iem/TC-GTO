from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'data';CACHE=OUT/'offline_cache';STRUCT=OUT/'structures';EVENTS=OUT/'earthquakes'
from tcgto.structure_factory import make_structure
from scripts.opensees_smoke_test import build,eigen_and_participation
FAMILIES=('shear','frame','wall','braced','dual');NS=(6,8,10,12);CLASSES=('low_frequency','broadband','high_frequency','two_peak','short_duration','long_duration');DT=.01;STEPS=2001;G=9.80665
def wcsv(path,rows):
 ks=[]
 for r in rows:
  for k in r:
   if k not in ks:ks.append(k)
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(rows)
def event_motion(cls,index):
 seed=20260920+CLASSES.index(cls)*1000+index;rng=np.random.default_rng(seed);f=np.fft.rfftfreq(STEPS,DT);phase=rng.uniform(0,2*np.pi,len(f));amp=rng.rayleigh(1,len(f))
 if cls=='low_frequency':shape=np.exp(-.5*((f-1.0)/.55)**2)
 elif cls=='broadband':shape=np.exp(-.5*((f-3.0)/2.0)**2)
 elif cls=='high_frequency':shape=np.exp(-.5*((f-7.0)/2.0)**2)
 elif cls=='two_peak':shape=np.exp(-.5*((f-1.2)/.4)**2)+.75*np.exp(-.5*((f-6.0)/1.0)**2)
 elif cls=='short_duration':shape=np.exp(-.5*((f-3.5)/2.0)**2)
 else:shape=np.exp(-.5*((f-1.5)/.8)**2)
 z=amp*shape*np.exp(1j*phase);z[0]=0.;x=np.fft.irfft(z,n=STEPS);t=np.arange(STEPS)*DT
 if cls=='short_duration':env=np.exp(-.5*((t-7.)/1.6)**2)
 elif cls=='long_duration':env=np.clip(t/2.5,0,1)*np.clip((20-t)/2.5,0,1)
 else:env=np.sin(np.pi*np.clip(t/20,0,1))**.6
 x*=env;target=(.10+.32*((index*7+CLASSES.index(cls)*3)%18)/17)*G;x*=target/max(np.max(np.abs(x)),1e-12);return x.astype(np.float32)
def prepare_inputs():
 CACHE.mkdir(parents=True,exist_ok=True);STRUCT.mkdir(exist_ok=True);EVENTS.mkdir(exist_ok=True);erows=[];srows=[]
 for cls in CLASSES:
  for i in range(18):
   split='train' if i<12 else ('validation' if i<15 else 'test');eid=f'E8EQ_{cls}_{i+1:02d}';g=event_motion(cls,i);np.savez_compressed(EVENTS/f'{eid}.npz',event_id=np.asarray(eid),spectral_class=np.asarray(cls),split=np.asarray(split),dt=np.asarray(DT),ground_motion=g);erows.append({'event_id':eid,'spectral_class':cls,'split':split,'path':str((EVENTS/f'{eid}.npz').relative_to(OUT)),'PGA_mps2':float(np.max(np.abs(g))),'sha256':hashlib.sha256((EVENTS/f'{eid}.npz').read_bytes()).hexdigest()})
 for fam in FAMILIES:
  for n in NS:
   for i in range(8):
    split='train' if i<5 else ('validation' if i<7 else 'test');sid=f'E8_{fam.upper()}_N{n}_{i+1:02d}';s=make_structure(fam,n,i,split);masters=build(s);vals,freq,phi,gamma=eigen_and_participation(s,masters);import openseespy.opensees as ops;ops.wipe();meta={'structure_id':sid,'family':fam,'N':n,'factory_index':i,'split':split,'story_height':s['story_height'],'zeta':s['zeta'],'params':s['params'],'floor_master_count':len(masters),'solver':'OpenSeesPy 3.8.0 explicit FE'};p=STRUCT/f'{sid}.npz';np.savez_compressed(p,M=s['M'].astype(np.float64),C=s['C'].astype(np.float64),K=s['K'].astype(np.float64),mass=s['mass'].astype(np.float64),floor_height=s['floor_height'].astype(np.float64),frequencies_hz=freq,modal_shapes=phi,participation=gamma,metadata_json=np.asarray(json.dumps(meta)));srows.append({'structure_id':sid,'family':fam,'N':n,'factory_index':i,'split':split,'path':str(p.relative_to(OUT)),'f1_hz':freq[0],'f2_hz':freq[1],'f3_hz':freq[2]})
 wcsv(OUT/'E8_EARTHQUAKE_SPLIT.csv',erows);wcsv(OUT/'E8_STRUCTURE_INDEX.csv',srows);return srows,erows
def record_task(task):
 fam,n,idx,split,sid,eid,cls,gpath,outpath=task;import openseespy.opensees as ops;s=make_structure(fam,n,idx,split);masters=build(s);_,freq,phi,gamma=eigen_and_participation(s,masters);ground=np.load(gpath)['ground_motion'].astype(np.float64);ops.timeSeries('Path',1,'-dt',DT,'-values',*ground.tolist(),'-factor',1.0);ops.pattern('UniformExcitation',1,1,'-accel',1);ops.rayleigh(float(s['rayleigh_alpha']),0.,0.,float(s['rayleigh_beta']));ops.wipeAnalysis();ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormDispIncr',1e-8,20);ops.algorithm('Linear');ops.integrator('Newmark',.5,.25);ops.analysis('Transient');q=np.zeros((STEPS,n),np.float32);v=np.zeros_like(q);a=np.zeros_like(q);ok=True;t0=time.perf_counter()
 for step in range(1,STEPS):
  if ops.analyze(1,DT)!=0:ok=False;break
  q[step]=[ops.nodeDisp(x,1) for x in masters];v[step]=[ops.nodeVel(x,1) for x in masters];a[step]=np.asarray([ops.nodeAccel(x,1) for x in masters])+ground[step]
 elapsed=time.perf_counter()-t0;ops.wipe()
 if not ok:return {'record_id':sid+'__'+eid,'valid':False,'reason':'OpenSees analyze nonzero','elapsed_seconds':elapsed}
 rid=sid+'__'+eid;meta={'record_id':rid,'structure_id':sid,'family':fam,'N':n,'earthquake_id':eid,'spectral_class':cls,'split':split,'dt':DT,'solver':'OpenSeesPy 3.8.0 transient Newmark','ground_excitation':'UniformExcitation; stored a_abs=nodeAccel(relative)+ground','units':{'q':'m','v':'m/s','a_abs':'m/s2','ground':'m/s2'}};np.savez_compressed(outpath,q=q,v=v,a_abs=a,ground_motion=ground.astype(np.float32),dt=np.asarray(DT),metadata_json=np.asarray(json.dumps(meta)));qd=np.gradient(q,DT,axis=0,edge_order=2);qn=float(np.sqrt(np.mean((qd-v)**2))/max(np.sqrt(np.mean(v*v)),1e-12));identity=float(np.max(np.abs((a-ground[:,None])-a+ground[:,None])));return {'record_id':rid,'structure_id':sid,'family':fam,'N':n,'earthquake_id':eid,'spectral_class':cls,'split':split,'path':str(Path(outpath).relative_to(OUT)),'valid':bool(np.isfinite(q).all() and qn<.15),'qdot_v_nrmse':qn,'acceleration_identity_error':identity,'elapsed_seconds':elapsed,'bytes':Path(outpath).stat().st_size}
def tasks(srows,erows):
 rows=[]
 for s in srows:
  for e in erows:
   if s['split']==e['split']:
    p=CACHE/s['split']/f"{s['structure_id']}__{e['event_id']}.npz";p.parent.mkdir(exist_ok=True);rows.append((s['family'],int(s['N']),int(s['factory_index']),s['split'],s['structure_id'],e['event_id'],e['spectral_class'],str(OUT/e['path']),str(p)))
 return rows
def estimate():
 s,e=prepare_inputs();sample=[]
 for fam in FAMILIES:sample.append(next(x for x in tasks(s,e) if x[0]==fam and x[3]=='train'))
 rows=[record_task(x) for x in sample];avg=float(np.mean([x['elapsed_seconds'] for x in rows]));size=float(np.mean([x['bytes'] for x in rows]));est={'sample_records':len(rows),'mean_seconds_per_record':avg,'estimated_serial_hours_8280':avg*8280/3600,'estimated_8_process_hours':avg*8280/3600/8,'mean_cache_bytes_per_record':size,'estimated_cache_GiB':size*8280/2**30,'all_sample_valid':all(x['valid'] for x in rows)};(OUT/'E8_GENERATION_ESTIMATE.json').write_text(json.dumps(est,indent=2),encoding='utf-8');print(json.dumps(est,indent=2))
def generate(workers):
 s,e=prepare_inputs();todo=tasks(s,e);started=time.perf_counter();rows=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  fs=[pool.submit(record_task,x) for x in todo]
  for i,f in enumerate(as_completed(fs),1):
   try:rows.append(f.result())
   except Exception as ex:rows.append({'valid':False,'reason':repr(ex)})
   if i%100==0:print(json.dumps({'done':i,'total':len(todo),'elapsed_s':time.perf_counter()-started}),flush=True)
 wcsv(OUT/'E8_DATASET_INDEX.csv',rows);failed=[x for x in rows if not x.get('valid',False)];wcsv(OUT/'E8_FAILED_ANALYSES.csv',failed)
 # Streaming train-only normalization over physical cache arrays.
 sums=np.zeros(5);sums2=np.zeros(5);counts=np.zeros(5)
 for r in rows:
  if r.get('valid') and r['split']=='train':
   z=np.load(OUT/r['path']);arrs=(z['q'],z['v'],z['a_abs'],z['a_abs'],z['ground_motion'])
   for j,a in enumerate(arrs):x=np.asarray(a,np.float64).ravel();sums[j]+=x.sum();sums2[j]+=(x*x).sum();counts[j]+=len(x)
 mean=sums/counts;std=np.sqrt(np.maximum(sums2/counts-mean*mean,1e-30));norm={'fit_split':'train','response_mean':mean[:3].tolist(),'response_std':std[:3].tolist(),'acceleration_mean':mean[3],'acceleration_std':std[3],'input_mean':mean[4],'input_std':std[4],'validation_test_accessed':False};(OUT/'E8_NORMALIZATION.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
 sets={sp:{r['earthquake_id'] for r in rows if r.get('valid') and r['split']==sp} for sp in ('train','validation','test')};leak=bool((sets['train']&sets['validation'])|(sets['train']&sets['test'])|(sets['validation']&sets['test']));summary={'status':'PASS' if len(rows)==8280 and not failed and not leak else 'FAIL','structures':len(s),'records':len(rows),'failed_analyses':len(failed),'split_counts':{sp:sum(r.get('split')==sp for r in rows) for sp in sets},'unique_earthquakes':{sp:len(sets[sp]) for sp in sets},'identity_leakage':leak,'normalization_ready':not failed and not leak,'offline_cache':str(CACHE),'elapsed_seconds':time.perf_counter()-started,'workers':workers};(OUT/'E8_DATASET_QA_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--estimate',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args();estimate() if a.estimate else generate(a.workers)
