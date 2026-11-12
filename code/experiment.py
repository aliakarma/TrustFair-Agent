"""Reproducible *synthetic proxy* study, not real disability/employment research.
Dependencies: numpy, matplotlib. Run python code/experiment.py from project root.
Threat: one of four clients replaces its model update with adversarial anti-aligned update.
Trust filter uses a clean public synthetic anchor dataset (an explicit assumption).
Agent safety is a deterministic capability filter, not an LLM prompt-injection defense.
"""
import json,csv,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; (ROOT/'results').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
SEEDS=[41,107,233,509,811]
N_CLIENTS=4; D=10; ROUNDS=16; N_TRAIN=480; N_TEST=220

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-25,25)))
def gen(rng,n,k):
    # Disability-support group g, language proxy l and institution heterogeneity
    g=rng.binomial(1,[.20,.36,.60,.74][k],n); lang=rng.binomial(1,[.15,.28,.58,.80][k],n)
    skills=rng.normal(loc=[-.25,0,.25,.40][k],scale=1,size=(n,4))
    needs=rng.binomial(1,.2+.45*g,size=n); accommodation=rng.binomial(1,.60-.12*k,size=n)
    accessible=rng.binomial(1,.65,size=n)
    x=np.column_stack([skills,g,lang,needs,accommodation,accessible,np.ones(n)])
    # Jobs suitability synthetic oracle, no human outcome semantics implied
    score=1.05*skills[:,0]+.70*skills[:,1]-.42*skills[:,2]+.35*skills[:,3]
    score+=.55*accommodation+.50*accessible-.65*needs*(1-accommodation)-.25*lang*(1-accessible)
    score+=.20*g+.12*k-0.2
    y=rng.binomial(1,sigmoid(score))
    return x.astype(float),y.astype(float),g

def grad(w,x,y,weights=None):
    p=sigmoid(x@w); e=p-y
    if weights is not None: e=e*weights/np.mean(weights)
    return (x.T@e)/len(y)+0.004*np.r_[w[:-1],0]

def train(w,x,y,lr=.25,steps=6,weights=None):
    w=w.copy()
    for _ in range(steps): w-=lr*grad(w,x,y,weights)
    return w

def f1(y,yhat):
    tp=((y==1)&(yhat==1)).sum(); fp=((y==0)&(yhat==1)).sum(); fn=((y==1)&(yhat==0)).sum()
    return float(2*tp/(2*tp+fp+fn+1e-12))

def metric(w,tests,personal=None):
    ys=[]; pp=[]; gaps=[]; locales=[]
    for k,(x,y,g) in enumerate(tests):
        wk=w if personal is None else w+personal[k]
        p=(sigmoid(x@wk)>=.5).astype(int)
        locales.append(f1(y,p)); ys.extend(y); pp.extend(p)
        # fairness: true-positive rate difference on support group, measured on eligible positives
        a=(y==1)&(g==0); b=(y==1)&(g==1)
        if a.sum()>0 and b.sum()>0: gaps.append(abs(p[a].mean()-p[b].mean()))
    return float(f1(np.array(ys),np.array(pp))),float(np.mean(gaps)),float(min(locales))

def one(seed,mode,attack):
    rng=np.random.default_rng(seed)
    trains=[]; tests=[]
    for k in range(N_CLIENTS):
        x,y,g=gen(rng,N_TRAIN,k); trains.append((x,y,g))
        x,y,g=gen(rng,N_TEST,k); tests.append((x,y,g))
    # The clean anchor is a privileged synthetic validation resource; not from clients.
    ax,ay,_=gen(rng,360,1)
    w=np.zeros(D); history=[]
    for r in range(ROUNDS):
        updates=[]; ws=[]
        ref=-grad(w,ax,ay)
        for k,(x,y,g) in enumerate(trains):
            weights=None
            if mode=='trustfair':
                weights=np.where(g==1,1.5,1.0)
            u=train(w,x,y,weights=weights)-w
            if attack and k==3: u=-5*u # malicious model-replacement / sign-flip proxy
            updates.append(u)
            if mode=='fedavg': val=1.
            else:
                cosine=float(u@ref/(np.linalg.norm(u)*np.linalg.norm(ref)+1e-10))
                normratio=float(np.linalg.norm(u)/(np.linalg.norm(ref)+1e-8))
                val=max(0,cosine)*min(1.,2./max(normratio,1e-8))
            ws.append(val)
        updates=np.array(updates); ws=np.array(ws)
        if ws.sum()<1e-8: ws[:]=1
        w=w+np.average(updates,axis=0,weights=ws)
        history.append(metric(w,tests)[0])
    # personalized on-device adapter: locally optimized intercept only, no shared raw data
    adapters=[]
    if mode=='trustfair':
        for x,y,g in trains:
            bias=0.
            for _ in range(15):
                p=sigmoid(x@w+bias)
                bias-=.20*((p-y).mean()+0.30*bias)
            a=np.zeros(D);a[-1]=bias;adapters.append(a)
    return metric(w,tests,adapters if adapters else None),history

def run():
    # matplotlib imported here so revision_experiments.py can import the generator without it
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]; curves={};
    for attack in [False,True]:
        for mode in ['fedavg','trusted','trustfair']:
            for seed in SEEDS:
                m,h=one(seed,mode,attack)
                rows.append({'attack':int(attack),'method':mode,'seed':seed,'f1':m[0],'tpr_gap':m[1],'worst_client_f1':m[2]})
                curves[(attack,mode,seed)]=h
    with open(ROOT/'results'/'runs.csv','w',newline='') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary=[]
    for attack in [0,1]:
        for mode in ['fedavg','trusted','trustfair']:
            subset=[r for r in rows if r['attack']==attack and r['method']==mode]
            item={'attack':attack,'method':mode}
            for field in ['f1','tpr_gap','worst_client_f1']:
                v=np.array([r[field] for r in subset]);item[field+'_mean']=float(v.mean());item[field+'_sd']=float(v.std(ddof=1))
            summary.append(item)
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    # graph 1 F1 under benign and attack
    fig,ax=plt.subplots(figsize=(7,4.3));names=['FedAvg','Anchor trust','TrustFair + local adapter']
    v=np.array([[next(r['f1_mean'] for r in summary if r['attack']==a and r['method']==m) for m in ['fedavg','trusted','trustfair']] for a in [0,1]])
    xs=np.arange(3);ax.bar(xs-.18,v[0],.36,label='No attack');ax.bar(xs+.18,v[1],.36,label='1 malicious of 4');ax.set_xticks(xs,names);ax.set_ylim(0,1);ax.set_ylabel('Pooled F1');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures'/'performance.pdf');plt.close(fig)
    # fairness metric attack
    fig,ax=plt.subplots(figsize=(7,4.3));v=np.array([[next(r['tpr_gap_mean'] for r in summary if r['attack']==a and r['method']==m) for m in ['fedavg','trusted','trustfair']] for a in [0,1]])
    ax.bar(xs-.18,v[0],.36,label='No attack');ax.bar(xs+.18,v[1],.36,label='1 malicious of 4');ax.set_xticks(xs,names);ax.set_ylabel('Mean within-client TPR gap (lower better)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures'/'fairness.pdf');plt.close(fig)
    # learning curves
    fig,ax=plt.subplots(figsize=(7,4.3))
    for m,nm in [('fedavg','FedAvg'),('trusted','Anchor trust'),('trustfair','TrustFair')]:
        arr=np.array([curves[(True,m,s)] for s in SEEDS]); ax.plot(range(1,ROUNDS+1),arr.mean(0),label=nm)
    ax.set_xlabel('Communication round');ax.set_ylabel('Pooled F1 (attack setting)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures'/'convergence.pdf');plt.close(fig)
    # deterministic capability gate test
    cases=[('read_public_job','job_public',True),('get_own_profile','profile_self',True),('read_other_profile','profile_other',False),('send_private_profile','external_send',False),('submit_employer_rejection','autonomous_reject',False),('change_consent','consent_write',False),('recommend_training','training_recommend',True),('submit_verified_application','submit_human_approved',True)]
    allow={'job_public','profile_self','training_recommend','submit_human_approved'}
    outcomes=[dict(name=n,capability=c,expected=expected,permitted=(c in allow),correct=((c in allow)==expected)) for n,c,expected in cases]
    (ROOT/'results'/'policy_checks.json').write_text(json.dumps(outcomes,indent=2))
    return summary
if __name__=='__main__':run()
