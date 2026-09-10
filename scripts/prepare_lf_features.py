"""E8-LF-PREP: pair-independent, train-only LF scaling and 20-epoch layout bank."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1] / 'data'; CACHE=ROOT/'offline_cache'; OUT=ROOT/'lf_cache'; SEED=20260930
EPOCHS=20; NFREQ=7; PMAX=20; CHANNELS=4; NPER=256; STEP=128; NFFT=256; FS=100.
WINDOW=np.hanning(NPER).astype(np.float64); FREQ=np.fft.rfftfreq(NFFT,1/FS); KEEP=np.where((FREQ>=.2)&(FREQ<=3.0))[0]
assert len(KEEP)==NFREQ
def rows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def stable_seed(record_id,epoch):return int.from_bytes(hashlib.sha256(f'{SEED}|{epoch}|{record_id}'.encode()).digest()[:8],'little')%(2**32)
def sensors_for(record_id,epoch,n,ns):
 rng=np.random.default_rng(stable_seed(record_id,epoch));return np.sort(np.r_[0,rng.choice(np.arange(1,n),ns-1,replace=False)]).astype(np.int16)
def canonical(n):return np.unique(np.rint(np.linspace(0,n-1,4)).astype(np.int16))
def fft_segments(x):
 starts=range(0,len(x)-NPER+1,STEP);a=np.stack([x[s:s+NPER]-x[s:s+NPER].mean(0,keepdims=True) for s in starts]);return np.fft.rfft(a*WINDOW[None,:,None],n=NFFT,axis=1)
def signature(F,sensors,n):
 pairs=np.asarray([(int(i),int(j)) for i in sensors for j in sensors if i!=j],np.int16);sig=np.zeros((len(pairs),NFREQ,CHANNELS),np.float32);geom=np.zeros((len(pairs),4),np.float32);z=np.linspace(0,1,n)
 for p,(i,j) in enumerate(pairs):
  xi,xj=F[:,:,i],F[:,:,j];sij=np.mean(np.conj(xi)*xj,axis=0);sii=np.mean(np.abs(xi)**2,axis=0);sjj=np.mean(np.abs(xj)**2,axis=0);t=sij/(sjj+1e-10);coh=np.clip(np.abs(sij)**2/(sii*sjj+1e-10),0,1);tt=t[KEEP];sig[p,:,0]=np.log(np.abs(tt)+1e-10);sig[p,:,1]=np.cos(np.angle(tt));sig[p,:,2]=np.sin(np.angle(tt));sig[p,:,3]=coh[KEEP];geom[p]=[z[i],z[j],z[j]-z[i],abs(z[j]-z[i])]
 return sig,pairs,geom
def allocate(prefix,count):
 OUT.mkdir(parents=True,exist_ok=True);return {
  'signature':np.lib.format.open_memmap(OUT/f'{prefix}_signature.npy','w+',np.float32,(EPOCHS if prefix=='train' else 1,count,PMAX,NFREQ,CHANNELS)),
  'sensors':np.lib.format.open_memmap(OUT/f'{prefix}_sensors.npy','w+',np.int16,(EPOCHS if prefix=='train' else 1,count,5)),
  'pairs':np.lib.format.open_memmap(OUT/f'{prefix}_pairs.npy','w+',np.int16,(EPOCHS if prefix=='train' else 1,count,PMAX,2)),
  'geometry':np.lib.format.open_memmap(OUT/f'{prefix}_geometry.npy','w+',np.float32,(EPOCHS if prefix=='train' else 1,count,PMAX,4)),
  'valid':np.lib.format.open_memmap(OUT/f'{prefix}_valid_mask.npy','w+',np.bool_,(EPOCHS if prefix=='train' else 1,count,PMAX)),
  'ns':np.lib.format.open_memmap(OUT/f'{prefix}_Ns.npy','w+',np.int8,(EPOCHS if prefix=='train' else 1,count)),
  'seed':np.lib.format.open_memmap(OUT/f'{prefix}_seed.npy','w+',np.uint32,(EPOCHS if prefix=='train' else 1,count))}
def fill_slot(bank,e,i,sig,sensors,pairs,geom,seed):
 p=len(pairs);bank['signature'][e,i,:p]=sig;bank['sensors'][e,i,:]=-1;bank['sensors'][e,i,:len(sensors)]=sensors;bank['pairs'][e,i,:,:]=-1;bank['pairs'][e,i,:p]=pairs;bank['geometry'][e,i,:p]=geom;bank['valid'][e,i,:]=False;bank['valid'][e,i,:p]=True;bank['ns'][e,i]=len(sensors);bank['seed'][e,i]=seed
def main():
 started=time.perf_counter();index=rows(ROOT/'E8_DATASET_INDEX.csv');train=[r for r in index if r['split']=='train'];val=[r for r in index if r['split']=='validation'];assert len(train)==7200 and len(val)==720
 tb=allocate('train',len(train));vb=allocate('validation',len(val));sums=np.zeros((NFREQ,3));sums2=np.zeros_like(sums);count=0
 # Exact 216/144 Ns=4/5 allocation inside every family x N cell each epoch.
 cell={}
 for i,r in enumerate(train):cell.setdefault((r['family'],int(r['N'])),[]).append(i)
 for epoch in range(EPOCHS):
  nsmap=np.empty(len(train),np.int8)
  for key,idx in cell.items():
   order=np.asarray(idx)[np.random.default_rng(SEED+epoch*1009+sum(map(ord,key[0]))+key[1]).permutation(len(idx))];nsmap[order[:216]]=4;nsmap[order[216:]]=5
  for i,r in enumerate(train):
   with np.load(ROOT/r['path']) as z:x=z['a_abs'].astype(np.float64)
   F=fft_segments(x);ns=int(nsmap[i]);seed=stable_seed(r['record_id'],epoch+1);sens=sensors_for(r['record_id'],epoch+1,int(r['N']),ns);sig,pairs,geom=signature(F,sens,int(r['N']));fill_slot(tb,epoch,i,sig,sens,pairs,geom,seed);sums+=sig[:,:,:3].sum(0);sums2+=(sig[:,:,:3]**2).sum(0);count+=len(pairs)
  print(json.dumps({'epoch_bank':epoch+1,'Ns4':int((nsmap==4).sum()),'Ns5':int((nsmap==5).sum()),'elapsed_s':time.perf_counter()-started}),flush=True)
 mean=sums/count;std=np.sqrt(np.maximum(sums2/count-mean*mean,1e-12));scale={'kind':'pair_independent','fit_split':'train','mean':mean.tolist(),'std':std.tolist(),'valid_pairs_observed':count,'frequency_hz':FREQ[KEEP].tolist(),'channels':['log_abs_T','cos_phase_T','sin_phase_T'],'coherence_unscaled':True};(OUT/'E8_LF_SCALING.json').write_text(json.dumps(scale,indent=2),encoding='utf-8')
 # Normalize train bank in-place; coherence remains in [0,1].
 for e in range(EPOCHS):
  for lo in range(0,len(train),256):tb['signature'][e,lo:lo+256,:,:,:3]=(tb['signature'][e,lo:lo+256,:,:,:3]-mean)/(std+1e-8)
 # Validation is application-only with one fixed canonical Ns=4 layout.
 for i,r in enumerate(val):
  with np.load(ROOT/r['path']) as z:x=z['a_abs'].astype(np.float64)
  sens=canonical(int(r['N']));assert len(sens)==4;sig,pairs,geom=signature(fft_segments(x),sens,int(r['N']));sig[:,:,:3]=(sig[:,:,:3]-mean)/(std+1e-8);fill_slot(vb,0,i,sig,sens,pairs,geom,stable_seed(r['record_id'],0))
 # Flush memmaps before independent QA reads.
 for b in (tb,vb):
  for a in b.values():a.flush()
 tr=np.load(OUT/'train_signature.npy',mmap_mode='r');tv=np.load(OUT/'train_valid_mask.npy',mmap_mode='r');tn=np.load(OUT/'train_Ns.npy',mmap_mode='r');vs=np.load(OUT/'validation_signature.npy',mmap_mode='r');vv=np.load(OUT/'validation_valid_mask.npy',mmap_mode='r')
 trainvals=tr[tv];valvals=vs[vv];extreme_std=bool(np.any(std<1e-3)|np.any(std>50));qa={'status':'PASS','train_records':len(train),'validation_records':len(val),'test_records_accessed':0,'epochs':EPOCHS,'pair_independent_shape':list(mean.shape),'train_finite':bool(np.isfinite(trainvals).all()),'validation_finite':bool(np.isfinite(valvals).all()),'std_min':float(std.min()),'std_max':float(std.max()),'extreme_std':extreme_std,'validation_abs_max':float(np.max(np.abs(valvals[:,:,:3]))),'validation_clipped':False,'Ns4_occurrences':int((tn==4).sum()),'Ns5_occurrences':int((tn==5).sum()),'each_epoch_Ns4':4320,'each_epoch_Ns5':2880,'Pmax':PMAX,'Ns4_valid_pairs':12,'Ns5_valid_pairs':20,'trace_fields':['signature','sensor indices','pair indices','geometry descriptors','valid mask','Ns','seed'],'elapsed_seconds':time.perf_counter()-started};qa['status']='PASS' if qa['train_finite'] and qa['validation_finite'] and not extreme_std else 'FAIL';(OUT/'E8_LF_PREP_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8');manifest={'status':qa['status'],'seed':SEED,'record_order_file':'../E8_DATASET_INDEX.csv','train_record_order':'rows where split=train in CSV order','validation_record_order':'rows where split=validation in CSV order','files':sorted(p.name for p in OUT.glob('*.npy'))+['E8_LF_SCALING.json'],'test_accessed':False};(OUT/'E8_LF_CACHE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(qa,indent=2))
 if qa['status']!='PASS':raise RuntimeError('E8-LF-PREP QA failed')
if __name__=='__main__':main()
