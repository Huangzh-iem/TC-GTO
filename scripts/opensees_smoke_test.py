from __future__ import annotations
import copy,json,math,sys,time
from pathlib import Path
import numpy as np
import openseespy.opensees as ops
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcgto.structure_factory import make_structure
OUT=Path(__file__).resolve().parent; EFRAME=2e11; EWALL=3e10

def add_beam(tag,a,b,A,E,I):ops.element('elasticBeamColumn',tag,a,b,A,E,I,1);return tag+1
def build(s,dual_multiplier=1.0):
    fam=s['family'];n=s['N'];h=s['story_height'];p=s['params'];m=s['mass'];ops.wipe();ops.model('basic','-ndm',2,'-ndf',3);masters=[];tag=1
    if fam=='shear':
        # 1D shear chain represented in a 2D domain; floor masters are the only lateral DOFs.
        ops.wipe();ops.model('basic','-ndm',1,'-ndf',1);ops.node(1,0.);ops.fix(1,1)
        for fl in range(1,n+1):ops.node(fl+1,0.);ops.mass(fl+1,float(m[fl-1]));masters.append(fl+1)
        for fl,k in enumerate(p['story_stiffness'],1):ops.uniaxialMaterial('Elastic',fl,float(k));ops.element('zeroLength',fl,fl,fl+1,'-mat',fl,'-dir',1)
        return masters
    bw=p['bay_width'];node=lambda fl,c:fl*4+c+1
    if fam in ('frame','braced','dual'):
        for fl in range(n+1):
            for c in range(4):
                nd=node(fl,c);ops.node(nd,c*bw,fl*h)
                if fl==0:ops.fix(nd,1,1,1)
                else:ops.mass(nd,float(m[fl-1]) if c==0 else 1e-9,1e-9,1e-9)
        for fl in range(1,n+1):
            master=node(fl,0);masters.append(master)
            for c in range(1,4):ops.equalDOF(master,node(fl,c),1)
        ops.geomTransf('Linear',1)
        sc=np.asarray(p['story_scale'])
        for fl in range(n):
            for c in range(4):tag=add_beam(tag,node(fl,c),node(fl+1,c),p['column_A']*math.sqrt(sc[fl]),EFRAME,p['column_I']*sc[fl])
        for fl in range(1,n+1):
            for c in range(3):tag=add_beam(tag,node(fl,c),node(fl,c+1),p['beam_A']*math.sqrt(sc[fl-1]),EFRAME,p['beam_I']*sc[fl-1])
        if fam=='braced':
            brace_mat=900001;ops.uniaxialMaterial('Elastic',brace_mat,EFRAME)
            for fl in range(n):
                for c in range(3):
                    a,b=(node(fl,c),node(fl+1,c+1)) if (fl+c)%2==0 else (node(fl,c+1),node(fl+1,c));ops.element('truss',tag,a,b,p['brace_area']*sc[fl],brace_mat);tag+=1
        if fam=='dual':
            wall=lambda fl:10000+fl;ops.node(wall(0),2*bw,0.);ops.fix(wall(0),1,1,1)
            for fl in range(1,n+1):ops.node(wall(fl),2*bw,fl*h);ops.mass(wall(fl),1e-9,1e-9,1e-9);ops.equalDOF(masters[fl-1],wall(fl),1)
            scale=float(p['applied_wall_EI_scale'])*dual_multiplier
            for fl in range(n):tag=add_beam(tag,wall(fl),wall(fl+1),p['wall_A']*math.sqrt(sc[fl]),EWALL*scale,p['wall_I']*sc[fl])
        return masters
    # wall
    for fl in range(n+1):
        ops.node(fl+1,0.,fl*h)
        if fl==0:ops.fix(fl+1,1,1,1)
        else:ops.mass(fl+1,float(m[fl-1]),1e-9,1e-9);masters.append(fl+1)
    ops.geomTransf('Linear',1);sc=np.asarray(p['story_scale'])
    for fl in range(n):tag=add_beam(tag,fl+1,fl+2,p['wall_A']*math.sqrt(sc[fl]),EWALL,p['wall_I']*sc[fl])
    return masters

def eigen_and_participation(s,masters):
    vals=np.asarray(ops.eigen('-fullGenLapack',3),float);freq=np.sqrt(vals)/(2*np.pi);M=np.diag(s['mass']);phi=np.asarray([[ops.nodeEigenvector(nd,r+1,1) for r in range(3)] for nd in masters]);
    for r in range(3):phi[:,r]/=max(np.max(np.abs(phi[:,r])),1e-12)
    gamma=np.asarray([(phi[:,r]@M@np.ones(s['N']))/(phi[:,r]@M@phi[:,r]) for r in range(3)])
    return vals,freq,phi,gamma
def transient(s,masters):
    dt=.01;values=(.08*np.sin(2*np.pi*1.2*np.arange(201)*dt)).tolist();ops.timeSeries('Path',1,'-dt',dt,'-values',*values,'-factor',9.80665);ops.pattern('UniformExcitation',1,1,'-accel',1);ops.rayleigh(float(s['rayleigh_alpha']),0.,0.,float(s['rayleigh_beta']));ops.wipeAnalysis();ops.constraints('Transformation');ops.numberer('RCM');ops.system('BandGeneral');ops.test('NormDispIncr',1e-9,20);ops.algorithm('Linear');ops.integrator('Newmark',.5,.25);ops.analysis('Transient');last=None
    for _ in range(200):
        if ops.analyze(1,dt)!=0:return False,None
        last={'q_m':[ops.nodeDisp(x,1) for x in masters],'v_mps':[ops.nodeVel(x,1) for x in masters],'a_rel_mps2':[ops.nodeAccel(x,1) for x in masters]}
    return True,last
def qa_one(fam):
    s=make_structure(fam,8,0,'e8_smoke');masters=build(s);vals,freq,phi,gamma=eigen_and_participation(s,masters);ok,last=transient(s,masters);K=s['K'];M=s['M'];C=s['C'];units=last and all(np.isfinite(np.asarray(v)).all() for v in last.values())
    row={'family':fam,'pass':bool(ok and len(masters)==8 and K.shape==(8,8) and M.shape==(8,8) and C.shape==(8,8) and np.allclose(K,K.T) and np.allclose(M,M.T) and np.allclose(C,C.T) and np.diag(M).min()>0 and np.all(freq>0) and np.isfinite(gamma).all() and units),'analysis_converged':ok,'matrix_shape':list(K.shape),'K_symmetric':bool(np.allclose(K,K.T)),'C_symmetric':bool(np.allclose(C,C.T)),'positive_mass':bool(np.diag(M).min()>0),'frequencies_hz':freq.tolist(),'modal_participation':gamma.tolist(),'floor_master_nodes':masters,'units':{'q':'m','v':'m/s','a_rel':'m/s^2','a_abs':'a_rel+ground'},'ground_excitation':'UniformExcitation; exported absolute floor acceleration = nodeAccel(relative)+ground','last_state':last};ops.wipe();return row
def dual_check():
    s=make_structure('dual',8,0,'e8_smoke');out=[]
    for mult in (.5,1.,2.):
        masters=build(s,mult);_,f,_,_=eigen_and_participation(s,masters);out.append({'wall_scale_multiplier':mult,'frequencies_hz':f.tolist()});ops.wipe()
    base=np.asarray(out[1]['frequencies_hz']);delta=max(float(np.max(np.abs(np.asarray(x['frequencies_hz'])-base)/base)) for x in (out[0],out[2]));return {'pass':delta>1e-3,'frame_wall_stiffness_control':s['params']['frame_wall_stiffness_control'],'applied_wall_EI_scale':s['params']['applied_wall_EI_scale'],'sweep':out,'max_relative_frequency_change':delta,'verification':'same frame/wall properties; only wall stiffness multiplier changed inside OpenSees build'}
def main():
    started=time.perf_counter();rows=[qa_one(f) for f in ('shear','frame','wall','braced','dual')];dual=dual_check();result={'pass':all(x['pass'] for x in rows) and dual['pass'],'python_version':sys.version,'opensees_version':ops.version(),'systems':rows,'dual_system_fix_verification':dual,'elapsed_seconds':time.perf_counter()-started};(OUT/'E8_FIVE_SYSTEM_SMOKE_QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));
    if not result['pass']:raise RuntimeError('five-system QA failed')
if __name__=='__main__':main()
