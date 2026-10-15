import numpy as np
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.get('input_shape',[8])[-1];self.net=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,b):return {'prediction':self.net(b['x'].float())}
    def objective(self,b,p):return F.mse_loss(p['prediction'],b['y'].float())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,8)).astype('float32');return {'x':x,'y':(x.sum(1,keepdims=True)/8).astype('float32'),'group':np.arange(n)}


def simulate(problem,variant='full'):
    import math
    stages=problem['stages'];offered=problem['captures'];deadline=problem['lifetime'];selector=0;slots=[None,None];clock=0.;epoch=None;capture=0.;k=0;activation=0;committed=0;next_offer=0;valid=0;skipped=0;expired=0;writes=0.;energy=0.;replay=0.;executed=0.;expired_energy=0.;epoch_energy=0.;coverage=np.zeros(len(stages));target=None;attempts=0;interrupted=0;latencies=[];events=[]
    capacity=max(s['payload'] for s in stages);nvm=2*(capacity+60)+8
    if nvm>problem['nvm']:raise ValueError('Two complete checkpoint slots exceed writable NVM')
    for interval in problem['trace']:
        duration=interval['duration'];end=clock+duration
        if not interval['on']:
            clock=end;target=None
            if epoch is not None:
                slot=slots[selector]
                if slot is not None and slot['epoch']==epoch:k,activation,committed=slot['position'],slot['activation'],slot['position']
                else:k,activation,committed=0,epoch+1,0
            continue
        available=problem['usable_energy'];write_allowance=problem['write_allowance']
        while clock<end:
            if epoch is None:
                if next_offer>=len(offered):clock=end;break
                clock=max(clock,offered[next_offer])
                if clock>=end:break
                while next_offer+1<len(offered) and offered[next_offer+1]<=clock:skipped+=1;next_offer+=1
                epoch=next_offer;capture=offered[epoch];activation=epoch+1;k=0;committed=0;next_offer+=1;slots=[None,None];selector=0;coverage[:]=0;epoch_energy=0.;target=None
            remaining_time=sum(stage['exec_ms'] for stage in stages[k:])
            if variant!='no_lifetime' and clock-capture+remaining_time>deadline:
                expired+=1;expired_energy+=epoch_energy;events.append({'epoch':epoch,'event':'expired','time':clock});epoch=None;continue
            stage=stages[k]
            if stage['live']+stage['scratch']+problem.get('policy_memory',1536)>problem['sram']:raise ValueError('Live state, scratch, and policy exceed SRAM')
            if target is None:
                if variant=='baseline':target=len(stages)
                elif variant=='periodic':target=min(len(stages),k+problem.get('checkpoint_interval',1))
                else:
                    scored=[];debt=sum(v['energy'] for v in stages[committed:k]);normalizer=sum(v['energy'] for v in stages)
                    for frontier in range(k+1,min(len(stages),k+problem.get('candidate_count',16))+1):
                        compute=sum(v['energy'] for v in stages[k:frontier]);elapsed=sum(v['exec_ms'] for v in stages[k:frontier]);payload=0 if frontier==len(stages) else stages[frontier-1]['payload']+68;preserve=payload*problem['write_energy_per_byte'];duration=elapsed+payload/problem['write_bytes_per_ms']
                        if compute+preserve+problem['reserve']>available or payload>write_allowance:continue
                        probability=math.exp(-duration/problem.get('mean_on',40));fresh=float(variant=='no_lifetime' or clock-capture+duration+sum(v['exec_ms'] for v in stages[frontier:])<=deadline)
                        score=fresh*probability*frontier/len(stages)-.1*(compute+preserve+(1-probability)*(.5*compute+debt))/max(normalizer,1e-8)-.01*payload/max(problem['write_allowance'],1)
                        scored.append((score,frontier))
                    if not scored:clock=end;break
                    target=max(scored)[1]
            checkpoint=variant!='baseline' and k+1==target and k<len(stages)-1
            ck_bytes=stage['payload']+68;ck_energy=ck_bytes*problem['write_energy_per_byte'];ck_time=ck_bytes/problem['write_bytes_per_ms']
            if checkpoint and (stage['energy']+ck_energy+problem['reserve']>available or ck_bytes>write_allowance):checkpoint=False
            if stage['energy']+problem['reserve']>available:clock=end;break
            needed=stage['exec_ms']
            fraction=min(1.,(end-clock)/needed);spent=stage['energy']*fraction;energy+=spent;executed+=spent;available-=spent
            replay+=min(coverage[k],fraction)*stage['energy'];coverage[k]=max(coverage[k],fraction);epoch_energy+=spent
            clock+=needed*fraction
            if fraction<1:
                k=committed;target=None;break
            activation=(activation*stage['factor']+stage['offset'])%65521;k+=1
            if checkpoint:
                attempts+=1;fraction=min(1.,max(0,end-clock)/ck_time);issued=ck_bytes*fraction;writes+=issued;write_allowance-=issued;energy+=ck_energy*fraction;available-=ck_energy*fraction;clock+=ck_time*fraction
                if fraction<1:interrupted+=1;break
                inactive=1-selector;slots[inactive]={'epoch':epoch,'capture':capture,'position':k,'activation':activation,'model_version':problem.get('model_version',1)};selector=inactive;committed=k;target=None
            if k==len(stages):
                oracle=epoch+1
                for reference in stages:oracle=(oracle*reference['factor']+reference['offset'])%65521
                correct=activation==oracle;age=clock-capture
                if correct and age<=deadline:valid+=1;latencies.append(age)
                events.append({'epoch':epoch,'event':'complete','time':clock,'age':age,'semantic_equal':correct,'on_time':age<=deadline});epoch=None
    baseline_energy=len(offered)*sum(stage['energy'] for stage in stages)
    return {'valid_completion_fraction':valid/len(offered),'offered':len(offered),'valid':valid,'skipped':skipped,'expired':expired,'unfinished_or_unadmitted':len(offered)-valid-skipped-expired,
            'nvm_kib_per_offered_epoch':writes/1024/len(offered),'energy_normalized':energy/max(baseline_energy,1e-8),'replay_energy_fraction':replay/max(executed,1e-8),'expired_work_share':expired_energy/max(executed,1e-8),'interrupted_write_rate':interrupted/max(attempts,1),'p95_valid_latency_ms':float(np.quantile(latencies,.95)) if latencies else None,'two_slot_capacity':nvm,'events':events}


def execute(args):
    import json
    from pathlib import Path
    result=simulate(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'execution.json').write_text(json.dumps(result,indent=2))
