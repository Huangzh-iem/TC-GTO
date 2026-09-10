"""E7 structure factory and condensed linear FE dynamics."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.linalg import eigh

from .fe_eigen import beam_element_global

FAMILIES=("shear","frame","wall","braced","dual")
REGIMES=("baseline_nonuniform","soft_lower_story","weak_mid_story","strong_taper","high_variation")

def chain_k(story):
 story=np.asarray(story,float);n=len(story);K=np.zeros((n,n))
 for i in range(n):K[i,i]=story[i]+(story[i+1] if i+1<n else 0)
 for i in range(n-1):K[i,i+1]=K[i+1,i]=-story[i+1]
 return K

def truss_global(x1,y1,x2,y2,E,A):
 L=math.hypot(x2-x1,y2-y1);c=(x2-x1)/L;s=(y2-y1)/L
 return E*A/L*np.array([[c*c,c*s,-c*c,-c*s],[c*s,s*s,-c*s,-s*s],[-c*c,-c*s,c*c,c*s],[-c*s,-s*s,c*s,s*s]])

def assemble_frame(n,h,bw,col_A,col_I,beam_A,beam_I,scale,braces=False,brace_A=.006):
 E=2e11;nc=4;nodes=[(fl,c,c*bw,fl*h) for fl in range(n+1) for c in range(nc)];nd=3*len(nodes);K=np.zeros((nd,nd));idx=lambda fl,c:fl*nc+c
 def add_beam(a,b,A,I):
  _,_,x1,y1=nodes[a];_,_,x2,y2=nodes[b];ke=beam_element_global(x1,y1,x2,y2,E,A,I);d=np.r_[3*a+np.arange(3),3*b+np.arange(3)];K[np.ix_(d,d)]+=ke
 for fl in range(n):
  for c in range(nc):add_beam(idx(fl,c),idx(fl+1,c),col_A*math.sqrt(scale[fl]),col_I*scale[fl])
 for fl in range(1,n+1):
  for c in range(nc-1):add_beam(idx(fl,c),idx(fl,c+1),beam_A*math.sqrt(scale[fl-1]),beam_I*scale[fl-1])
 if braces:
  for fl in range(n):
   for c in range(nc-1):
    a,b=(idx(fl,c),idx(fl+1,c+1)) if (fl+c)%2==0 else (idx(fl,c+1),idx(fl+1,c));_,_,x1,y1=nodes[a];_,_,x2,y2=nodes[b];ke=truss_global(x1,y1,x2,y2,E,brace_A*scale[fl]);d=np.array([3*a,3*a+1,3*b,3*b+1]);K[np.ix_(d,d)]+=ke
 return condense_with_diaphragm(K,n,nc)

def assemble_wall(n,h,A,I,scale,E=3e10):
 nd=3*(n+1);K=np.zeros((nd,nd))
 for fl in range(n):
  ke=beam_element_global(0,fl*h,0,(fl+1)*h,E,A*math.sqrt(scale[fl]),I*scale[fl]);d=np.r_[3*fl+np.arange(3),3*(fl+1)+np.arange(3)];K[np.ix_(d,d)]+=ke
 return condense_with_diaphragm(K,n,1)

def condense_with_diaphragm(K,n,ncols):
 base=np.arange(3*ncols);free=np.setdiff1d(np.arange(K.shape[0]),base);Kf=K[np.ix_(free,free)];internal=[];maps=[]
 for old in free:
  node=old//3;dof=old%3;fl=node//ncols
  if dof==0:maps.append(fl-1)
  else:internal.append(old);maps.append(None)
 nred=n+len(internal);T=np.zeros((len(free),nred));j=n
 for i,m in enumerate(maps):
  if m is not None:T[i,m]=1
  else:T[i,j]=1;j+=1
 Kr=T.T@Kf@T
 if nred==n:return (Kr+Kr.T)/2
 Krr=Kr[:n,:n];Kri=Kr[:n,n:];Kii=Kr[n:,n:];Kc=Krr-Kri@np.linalg.solve(Kii,Kri.T)
 return (Kc+Kc.T)/2

def rayleigh(M,K,zeta):
 w=np.sqrt(np.maximum(eigh(K,M,eigvals_only=True)[:3],1e-12));A=np.array([[1/(2*w[0]),w[0]/2],[1/(2*w[2]),w[2]/2]]);alpha,beta=np.linalg.solve(A,np.array([zeta,zeta]));return alpha*M+beta*K,float(alpha),float(beta)

def regime_scale(rng,n,regime):
 base=np.linspace(1,rng.uniform(.72,.96),n)*np.exp(rng.normal(0,.045,n))
 if regime=="soft_lower_story":base[0]*=.48
 elif regime=="weak_mid_story":base[n//2]*=.52
 elif regime=="strong_taper":base*=np.linspace(1,.55,n)
 elif regime=="high_variation":base*=np.exp(rng.normal(0,.20,n))
 return np.clip(base,.28,1.45)

def make_structure(family,n,index,split):
 seed=20260901+100000*FAMILIES.index(family)+1000*n+index;rng=np.random.default_rng(seed);regime=REGIMES[index%len(REGIMES)];h=float(rng.uniform(3.25,3.90));bw=float(rng.uniform(5.2,6.8));mass=2e5*rng.uniform(.82,1.18)*np.exp(rng.normal(0,.07,n))*np.linspace(1.05,.95,n);zeta=float(rng.uniform(.025,.05));sc=regime_scale(rng,n,regime);params={"story_height":h,"bay_width":bw,"regime":regime,"story_scale":sc.tolist()}
 if family=="shear":
  story=rng.uniform(2.3e8,3.9e8)*sc;K=chain_k(story);params["story_stiffness"]=story.tolist()
 elif family=="frame":
  vals=(rng.uniform(.024,.044),rng.uniform(.0007,.0018),rng.uniform(.019,.034),rng.uniform(.00036,.00082));K=assemble_frame(n,h,bw,*vals,sc);params.update(dict(zip(("column_A","column_I","beam_A","beam_I"),vals)))
 elif family=="wall":
  vals=(rng.uniform(1.4,2.4),rng.uniform(1.1,3.2));K=assemble_wall(n,h,*vals,sc);params.update({"wall_A":vals[0],"wall_I":vals[1],"wall_E":3e10})
 elif family=="braced":
  vals=(rng.uniform(.024,.042),rng.uniform(.00072,.00168),rng.uniform(.019,.033),rng.uniform(.00038,.00078));ba=float(rng.uniform(.0035,.0105));K=assemble_frame(n,h,bw,*vals,sc,True,ba);params.update(dict(zip(("column_A","column_I","beam_A","beam_I"),vals)));params["brace_area"]=ba
 else:
  vals=(rng.uniform(.022,.039),rng.uniform(.00062,.00142),rng.uniform(.018,.031),rng.uniform(.00034,.00072));wa=float(rng.uniform(1.3,2.3));wi=float(rng.uniform(.75,2.15));target=float(rng.uniform(.35,.70));Kf=assemble_frame(n,h,bw,*vals,sc);Kw=assemble_wall(n,h,wa,wi,sc);scale=target/(1-target)*np.trace(Kf)/max(np.trace(Kw),1e-30);K=Kf+scale*Kw;realized=float(np.trace(scale*Kw)/np.trace(K));params.update(dict(zip(("column_A","column_I","beam_A","beam_I"),vals)));params.update({"wall_A":wa,"wall_I":wi,"wall_E":3e10,"frame_wall_stiffness_control":target,"applied_wall_EI_scale":scale,"realized_wall_trace_share":realized})
 M=np.diag(mass);C,alpha,beta=rayleigh(M,K,zeta);w2,phi=eigh(K,M);w=np.sqrt(np.maximum(w2,1e-12));freq=w/(2*np.pi);phi=phi[:,:min(12,n)];gamma=phi.T@M@np.ones(n);off=K.copy();
 for i in range(n):
  for j in range(n):
   if abs(i-j)<=1:off[i,j]=0
 nonlocal_ratio=float(np.sum(np.abs(off))/max(np.sum(np.abs(K)),1e-30));sid=f"E7_{family.upper()}_N{n}_{split.upper()}_{index:03d}"
 return {"structure_id":sid,"family":family,"N":n,"split":split,"regime":regime,"story_height":h,"floor_height":np.arange(1,n+1)*h,"mass":mass,"M":M,"C":C,"K":K,"f":freq,"phi":phi,"gamma":gamma,"zeta":zeta,"rayleigh_alpha":alpha,"rayleigh_beta":beta,"nonlocal_coupling_ratio":nonlocal_ratio,"params":params}

def simulate(structure,ground,dt=.01):
 M=structure["M"];K=structure["K"];freq=structure["f"];phi=structure["phi"];gamma=structure["gamma"];zeta=structure["zeta"];steps=len(ground);modes=phi.shape[1];ym=np.zeros((steps,modes));vm=np.zeros_like(ym);am=np.zeros_like(ym)
 for r in range(modes):
  w=2*np.pi*freq[r];num,den,_=signal.cont2discrete(([-gamma[r]],[1,2*zeta*w,w*w]),dt,method="zoh");y=signal.lfilter(num.ravel(),den.ravel(),ground);v=np.gradient(y,dt,edge_order=2);acc=-2*zeta*w*v-w*w*y-gamma[r]*ground;ym[:,r]=y;vm[:,r]=v;am[:,r]=acc
 q=ym@phi.T;v=vm@phi.T;arel=am@phi.T;aabs=arel+ground[:,None]
 return q.astype(np.float32),v.astype(np.float32),aabs.astype(np.float32)

def save_structure(s,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,M=s["M"],C=s["C"],K=s["K"],frequencies_hz=s["f"],modal_shapes=s["phi"],participation=s["gamma"],floor_height=s["floor_height"],mass=s["mass"],metadata_json=np.asarray(json.dumps({k:s[k] for k in ("structure_id","family","N","split","regime","story_height","zeta","rayleigh_alpha","rayleigh_beta","nonlocal_coupling_ratio")}|{"params":s["params"]})))

def load_structure(path):
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z["metadata_json"]));return {**meta,"M":z["M"].astype(float),"C":z["C"].astype(float),"K":z["K"].astype(float),"f":z["frequencies_hz"].astype(float),"phi":z["modal_shapes"].astype(float),"gamma":z["participation"].astype(float),"floor_height":z["floor_height"].astype(float),"mass":z["mass"].astype(float)}
