"""E8-Scratch formal training: random TC-GTO, no E6 ancestry, test sealed."""
from __future__ import annotations
import argparse,csv,json,math,random,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
import sys;sys.path.insert(0,str(ROOT))
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcgto import model_config
from tcgto.valid_mask_g1r4a_model import ValidMaskG1R4A
from tcgto.anchored_q_model import AnchoredQModel

def training_device(allow_cpu: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if allow_cpu:
        return torch.device('cpu')
    raise RuntimeError('CUDA is not available; rerun on a CUDA-enabled PyTorch installation.')

def physical_slice(array, start, length):
    array=np.asarray(array);result=np.zeros((length,)+array.shape[1:],dtype=np.float32)
    source_lo=max(0,start);source_hi=min(len(array),start+length)
    if source_hi>source_lo:
        dest_lo=source_lo-start;result[dest_lo:dest_lo+source_hi-source_lo]=array[source_lo:source_hi]
    return result

BASE=ROOT/'data';LFC=BASE/'lf_cache';OUT=ROOT/'runs';SEED=20261001;BATCH=12;STEPS_EPOCH=600;EPOCHS=20;TOTAL=12000;CONTEXT=384;TARGET=96;ANCHOR=144;DT=.01
def rc(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def wc(rows,path):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def r2(t,p):t=np.asarray(t,np.float64);p=np.asarray(p,np.float64);return float(1-np.sum((t-p)**2)/max(np.sum((t-t.mean())**2),1e-30))
def setup():
 qa=json.loads((LFC/'E8_LF_PREP_QA.json').read_text());assert qa['status']=='PASS' and qa['test_records_accessed']==0
 rows=rc(BASE/'E8_DATASET_INDEX.csv');train=[x for x in rows if x['split']=='train'];val=[x for x in rows if x['split']=='validation'];assert len(train)==7200 and len(val)==720 and not any(x['split']=='test' for x in train+val)
 norm=json.loads((BASE/'E8_NORMALIZATION.json').read_text());structures={x['structure_id']:x for x in rc(BASE/'E8_STRUCTURE_INDEX.csv')};return train,val,norm,structures
def model_and_optimizer(device):
 torch.manual_seed(SEED);base=ValidMaskG1R4A(model_config()).to(device);model=AnchoredQModel(base,ANCHOR).to(device);heads=[];back=[]
 for name,p in model.named_parameters():
  (heads if name.startswith(('q0_head.','delta_q_head.','base.response_head.','base.input_head.','base.observed_input_residual.')) else back).append(p)
 opt=torch.optim.AdamW([{'params':back,'lr':1e-4,'name':'backbone'},{'params':heads,'lr':2e-4,'name':'heads'}],weight_decay=1e-4)
 def lr_lambda(step):
  if step<600:return max((step+1)/600,1/600)
  return .5*(1+math.cos(math.pi*(step-600)/(TOTAL-600)))
 return model,opt,torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)
def epoch_batches(epoch,rows,ns_bank,smoke20=False):
 cells={}
 for i,r in enumerate(rows):cells.setdefault((r['family'],int(r['N']),int(ns_bank[epoch-1,i])),[]).append(i)
 batches=[]
 for fam in ('shear','frame','wall','braced','dual'):
  for n in (6,8,10,12):
   for ns,expected in ((4,216),(5,144)):
    ids=cells[(fam,n,ns)];assert len(ids)==expected
    rng=np.random.default_rng(SEED+epoch*100003+n*1009+ns*101+sum(map(ord,fam)));rng.shuffle(ids);batches.extend(ids[i:i+BATCH] for i in range(0,len(ids),BATCH))
 rng=np.random.default_rng(SEED+epoch*900001);rng.shuffle(batches)
 if smoke20:
  # Deterministic 20-cell coverage with both Ns values represented.
  out=[]
  for ci,(fam,n) in enumerate(( (f,n) for f in ('shear','frame','wall','braced','dual') for n in (6,8,10,12) )):
   ns=4 if ci%2==0 else 5;out.append(cells[(fam,n,ns)][:BATCH])
  return out
 assert len(batches)==600;return batches
def make_batch(ids,rows,epoch,sigs,sensors,norm,structures,device):
 ns=int(np.sum(sensors[epoch-1,ids[0]]>=0));p=ns*(ns-1);n=int(rows[ids[0]]['N']);sp=[];rp=[];gp=[];mk=[];coords=[];q0=[]
 for idx in ids:
  r=rows[idx];assert int(r['N'])==n and int(np.sum(sensors[epoch-1,idx]>=0))==ns
  with np.load(BASE/r['path']) as z:q=z['q'];v=z['v'];a=z['a_abs'];g=z['ground_motion']
  peak=int(np.argmax(np.abs(g)));lo=max(0,min(len(g)-TARGET,peak-TARGET//2));rng=np.random.default_rng(SEED+epoch*1000003+idx);target=int(np.clip(lo+rng.integers(-192,193),0,len(g)-TARGET));cs=target-ANCHOR;sens=sensors[epoch-1,idx,:ns].astype(int);absolute=physical_slice(a,cs,CONTEXT);response=np.stack((physical_slice(q,cs,CONTEXT),physical_slice(v,cs,CONTEXT),physical_slice(a,cs,CONTEXT)),-1);ground=physical_slice(g,cs,CONTEXT);mask=np.zeros(n,np.float32);mask[sens]=1;x=np.zeros((CONTEXT,n),np.float32);x[:,sens]=((absolute-norm['acceleration_mean'])/norm['acceleration_std'])[:,sens];sp.append(x);rp.append((response-np.asarray(norm['response_mean']))/np.asarray(norm['response_std']));gp.append((ground-norm['input_mean'])/norm['input_std']);mk.append(mask);coords.append(np.linspace(0,1,n,dtype=np.float32));q0.append((q[target]-norm['response_mean'][0])/norm['response_std'][0])
 return {'sparse':torch.tensor(np.stack(sp),device=device),'response':torch.tensor(np.stack(rp),device=device,dtype=torch.float32),'input':torch.tensor(np.stack(gp),device=device,dtype=torch.float32),'mask':torch.tensor(np.stack(mk),device=device),'coords':torch.tensor(np.stack(coords),device=device),'valid':torch.ones(len(ids),n,device=device),'lf':torch.tensor(np.asarray(sigs[epoch-1,ids,:p]),device=device),'q0':torch.tensor(np.stack(q0),device=device,dtype=torch.float32),'ns':ns,'N':n}
def losses(model,b,norm):
 o=model(b['sparse'],b['mask'],b['coords'],b['valid'],b['lf']);sl=slice(ANCHOR,ANCHOR+TARGET);pr=o['response'][:,sl];tr=b['response'][:,sl];dqtruth=tr[...,0]-b['q0'][:,None,:];qabs=(pr[...,0]-tr[...,0]).square().mean();dq=(o['delta_q'][:,sl]-dqtruth).square().mean();q0=(o['q0']-b['q0']).square().mean();qp=pr[...,0]*norm['response_std'][0]+norm['response_mean'][0];deriv=(qp[:,2:]-qp[:,:-2])/(2*DT);dn=(deriv-norm['response_mean'][1])/norm['response_std'][1];kin=(dn-pr[:,1:-1,:,1]).square().mean();v=(pr[...,1]-tr[...,1]).square().mean();a=(pr[...,2]-tr[...,2]).square().mean();response=(qabs+dq+q0+.1*kin+v+a)/5.1;inp=(o['input'][:,sl]-b['input'][:,sl]).square().mean();return o,response,inp,{'q_absolute':qabs,'delta_q':dq,'q0':q0,'dqdt_v':kin,'v':v,'a':a}
def gradnorm(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);return float(np.sqrt(sum(float((g.detach()**2).sum()) for g in gs if g is not None)))
def calibrate(response,inp,params):
 gr,gi=gradnorm(response,params),gradnorm(inp,params);t=np.sqrt(max(gr,1e-16)*max(gi,1e-16));return {'response':t/max(gr,1e-16),'input':t/max(gi,1e-16),'response_grad':gr,'input_grad':gi,'response_share':.5,'input_share':.5}
def smoke(steps):
 train,_,norm,structures=setup();device=training_device(allow_cpu=False);sigs=np.load(LFC/'train_signature.npy',mmap_mode='r');sensors=np.load(LFC/'train_sensors.npy',mmap_mode='r');nsbank=np.load(LFC/'train_Ns.npy',mmap_mode='r');model,opt,sched=model_and_optimizer(device);batches=epoch_batches(1,train,nsbank,smoke20=True);seen={'family':set(),'N':set(),'Ns':set()};audit=[];weights=None
 for step in range(1,steps+1):
  ids=batches[(step-1)%len(batches)];b=make_batch(ids,train,1,sigs,sensors,norm,structures,device);model.train();_,resp,inp,parts=losses(model,b,norm);recal=weights is None or step%150==0
  if recal:weights=calibrate(resp,inp,[p for p in model.parameters() if p.requires_grad])
  loss=weights['response']*resp+weights['input']*inp;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();r=train[ids[0]];seen['family'].add(r['family']);seen['N'].add(int(r['N']));seen['Ns'].add(b['ns']);audit.append({'step':step,'loss':float(loss.detach()),'recalibrated':recal,'response_weight':weights['response'],'input_weight':weights['input'],'lr_backbone':opt.param_groups[0]['lr'],'lr_heads':opt.param_groups[1]['lr']})
 passed=all(np.isfinite(x['loss']) for x in audit) and seen=={'family':set(('shear','frame','wall','braced','dual')),'N':set((6,8,10,12)),'Ns':set((4,5))} and (steps<150 or any(x['step']==150 and x['recalibrated'] for x in audit));result={'pass':passed,'steps':steps,'seen':{k:sorted(v) for k,v in seen.items()},'step150_recalibration':any(x['step']==150 and x['recalibrated'] for x in audit),'test_accessed':False,'persistent_checkpoint_saved':False,'last':audit[-1]};OUT.mkdir(exist_ok=True);(OUT/f'SMOKE_{steps}_STEPS.json').write_text(json.dumps(result,indent=2),encoding='utf-8');wc(audit,OUT/f'SMOKE_{steps}_AUDIT.csv');print(json.dumps(result,indent=2));
 if not passed:raise RuntimeError('smoke failed')
@torch.no_grad()
def validate(model,rows,norm,structures,device):
 sigs=np.load(LFC/'validation_signature.npy',mmap_mode='r');sensbank=np.load(LFC/'validation_sensors.npy',mmap_mode='r');model.eval();truth={k:[] for k in ('q','v','a','input','pfa','idr')};pred={k:[] for k in truth};byfam={f:({k:[] for k in truth},{k:[] for k in truth}) for f in ('shear','frame','wall','braced','dual')};starts=list(range(0,2001-TARGET+1,TARGET));
 if starts[-1]!=2001-TARGET:starts.append(2001-TARGET)
 for n in (6,8,10,12):
  ids=[i for i,r in enumerate(rows) if int(r['N'])==n]
  for off in range(0,len(ids),BATCH):
   chunk=ids[off:off+BATCH];raw=[]
   for idx in chunk:
    with np.load(BASE/rows[idx]['path']) as z:raw.append((z['q'].astype(np.float32),z['v'].astype(np.float32),z['a_abs'].astype(np.float32),z['ground_motion'].astype(np.float32)))
   prs=[np.zeros((2001,n,3),np.float32) for _ in chunk];pis=[np.zeros(2001,np.float32) for _ in chunk];cnt=np.zeros(2001,np.float32)
   for st in starts:
    sp=[];mk=[];cr=[]
    for j,idx in enumerate(chunk):
     q,v,a,g=raw[j];s=sensbank[0,idx,:4].astype(int);x=np.zeros((CONTEXT,n),np.float32);aa=physical_slice(a,st-ANCHOR,CONTEXT);x[:,s]=((aa-norm['acceleration_mean'])/norm['acceleration_std'])[:,s];m=np.zeros(n,np.float32);m[s]=1;sp.append(x);mk.append(m);cr.append(np.linspace(0,1,n,dtype=np.float32))
    o=model(torch.tensor(np.stack(sp),device=device),torch.tensor(np.stack(mk),device=device),torch.tensor(np.stack(cr),device=device),torch.ones(len(chunk),n,device=device),torch.tensor(np.asarray(sigs[0,chunk,:12]),device=device));rr=o['response'][:,ANCHOR:ANCHOR+TARGET].cpu().numpy()*np.asarray(norm['response_std'])+np.asarray(norm['response_mean']);ii=o['input'][:,ANCHOR:ANCHOR+TARGET].cpu().numpy()*norm['input_std']+norm['input_mean']
    for j in range(len(chunk)):prs[j][st:st+TARGET]+=rr[j];pis[j][st:st+TARGET]+=ii[j]
    cnt[st:st+TARGET]+=1
   for j,idx in enumerate(chunk):
    q,v,a,g=raw[j];pp=prs[j]/cnt[:,None,None];pi=pis[j]/cnt;h=float(structures[rows[idx]['structure_id']]['N']) # overwritten below from structure file
    with np.load(BASE/structures[rows[idx]['structure_id']]['path']) as z:heights=z['floor_height'].astype(float)
    story_h=np.diff(np.r_[0.,heights]);ti=np.max(np.abs(np.diff(np.c_[np.zeros(len(q)),q],axis=1))/story_h,axis=0);pqi=pp[...,0];pri=np.max(np.abs(np.diff(np.c_[np.zeros(len(pqi)),pqi],axis=1))/story_h,axis=0);vals_t={'q':q.ravel(),'v':v.ravel(),'a':a.ravel(),'input':g.ravel(),'pfa':np.max(np.abs(a),axis=0),'idr':ti};vals_p={'q':pp[...,0].ravel(),'v':pp[...,1].ravel(),'a':pp[...,2].ravel(),'input':pi.ravel(),'pfa':np.max(np.abs(pp[...,2]),axis=0),'idr':pri};fam=rows[idx]['family']
    for k in truth:truth[k].append(vals_t[k]);pred[k].append(vals_p[k]);byfam[fam][0][k].append(vals_t[k]);byfam[fam][1][k].append(vals_p[k])
 def pack(t,p):
  m={k+'_r2':r2(np.concatenate(t[k]),np.concatenate(p[k])) for k in t};rn=np.concatenate([(np.concatenate(t[k])-norm['response_mean'][i])/norm['response_std'][i] for i,k in enumerate(('q','v','a'))]);pn=np.concatenate([(np.concatenate(p[k])-norm['response_mean'][i])/norm['response_std'][i] for i,k in enumerate(('q','v','a'))]);m['response_r2']=r2(rn,pn);return m
 result=pack(truth,pred)
 for fam,(t,p) in byfam.items():
  for k,v in pack(t,p).items():result[f'{fam}_{k}']=v
 result['weakest_family_response_r2']=min(result[f'{f}_response_r2'] for f in byfam);result['weakest_family_input_r2']=min(result[f'{f}_input_r2'] for f in byfam);return result
def train():
 trainrows,valrows,norm,structures=setup();device=training_device(allow_cpu=False);sigs=np.load(LFC/'train_signature.npy',mmap_mode='r');sensors=np.load(LFC/'train_sensors.npy',mmap_mode='r');nsbank=np.load(LFC/'train_Ns.npy',mmap_mode='r');model,opt,sched=model_and_optimizer(device);OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);protocol={'architecture':'TC-GTO','context':CONTEXT,'target':TARGET,'status':'RUNNING'};protocol['status']='RUNNING';(OUT/'PROTOCOL_LOCK.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');trajectory=[];calibration=[];global_step=0;weights=None
 for epoch in range(1,EPOCHS+1):
  batches=epoch_batches(epoch,trainrows,nsbank);losses_epoch=[];started=time.perf_counter();seen_events=set()
  for ids in batches:
   global_step+=1;b=make_batch(ids,trainrows,epoch,sigs,sensors,norm,structures,device);model.train();_,resp,inp,parts=losses(model,b,norm);recal=weights is None or global_step%150==0
   if recal:
    weights=calibrate(resp,inp,[p for p in model.parameters() if p.requires_grad]);calibration.append({'global_step':global_step,'epoch':epoch,**weights})
   total=weights['response']*resp+weights['input']*inp;opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();losses_epoch.append(float(total.detach()));seen_events.update(trainrows[i]['earthquake_id'] for i in ids)
  assert len(seen_events)==72;metrics=validate(model,valrows,norm,structures,device);row={'epoch':epoch,'global_step':global_step,'train_loss':float(np.mean(losses_epoch)),'train_unique_earthquakes':len(seen_events),'Ns4_batches':360,'Ns5_batches':240,'epoch_seconds':time.perf_counter()-started,**metrics};trajectory.append(row);torch.save({'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'scheduler_state':sched.state_dict(),'loss_weights':weights,'epoch':epoch,'global_step':global_step,'architecture':'random AnchoredQModel TC-GTO','context':CONTEXT,'target':TARGET,'normalization':'E8 train-only','E6_ancestry':False},OUT/'checkpoints'/f'E{epoch:02d}.pt');wc(trajectory,OUT/'E8_SCRATCH_VALIDATION_TRAJECTORY.csv');wc(calibration,OUT/'E8_SCRATCH_GRADIENT_CALIBRATION.csv');print(json.dumps(row),flush=True)
 (OUT/'DONE.json').write_text(json.dumps({'status':'COMPLETE_AWAITING_MANUAL_CHECKPOINT_SELECTION','epochs':EPOCHS,'steps':global_step,'automatic_best_selected':False,'test_accessed':False,'complex_EQ_accessed':False},indent=2),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',type=int,choices=(20,160));p.add_argument('--train',action='store_true');a=p.parse_args();train() if a.train else smoke(a.smoke)
