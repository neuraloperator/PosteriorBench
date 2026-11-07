"""Hidden static/moving safe-set playground.

Numerical risk solver: unchanged embedded risk1d.py. New, explicitly versioned
geometry prior; the physical dynamics depend on geometry via the same local-
coordinate model as the original generator. No solution-dependent rejection.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from collections import OrderedDict
import hashlib, json, time
import numpy as np
from numba import njit, prange
import risk1d as risk

VERSION = 'hidden-safe-playground-2.0'
PRIORS = OrderedDict([
    ('moving_mix', dict(id=10, title='Moving: diverse mixture', dynamic=True,
                       families=('swaying','breathing','wandering','pinching'), component='mix')),
    ('static_mix', dict(id=11, title='Static: diverse mixture', dynamic=False,
                       families=('static',), component='mix')),
    ('moving_single', dict(id=12, title='Moving: one swaying interval', dynamic=True,
                          families=('swaying',), component='single')),
    ('breathing', dict(id=13, title='Moving: expanding / contracting', dynamic=True,
                      families=('breathing',), component='mix')),
    ('moving_multi', dict(id=14, title='Moving: wandering disjoint intervals', dynamic=True,
                         families=('wandering',), component='multi')),
    ('pinching', dict(id=15, title='Moving: narrowing bottlenecks', dynamic=True,
                     families=('pinching',), component='multi')),
    ('static_single', dict(id=16, title='Static: random single interval', dynamic=False,
                          families=('static',), component='single')),
    ('static_multi', dict(id=17, title='Static: random disjoint intervals', dynamic=False,
                         families=('static',), component='multi')),
])
FAMILY_NAMES = ('static','swaying','breathing','wandering','pinching')

@dataclass(frozen=True)
class Settings:
    domain: tuple = (0.,1.)
    nx: int = 128
    nt: int = 128
    nz: int = 513
    substeps: int = 16
    max_components: int = 5
    spatial_modes: int = 8
    seed: int = 925
    candidate_stream: int = 202
    reference_stream: int = 101
    n_pool: int = 1000
    n_references: int = 6
    batch_size: int = 64
    observation_seed: int = 410
    cache_dir: str = 'cache'
    max_loaded_priors: int = 2
    def validate(self):
        if self.candidate_stream == self.reference_stream:
            raise ValueError('Reference and candidate streams must differ.')
        if min(self.n_pool,self.n_references,self.batch_size,self.max_loaded_priors)<1:
            raise ValueError('Pool, reference, batch and cache sizes must be positive.')
        if self.max_components<2:
            raise ValueError('At least two allowed components are needed for the multi-interval presets.')
        return self


def solver_config(settings, prior, reference=False):
    spec = PRIORS[prior]
    return risk.Config(mode='fixed_deadline_exit' if spec['dynamic'] else 'fixed_start_exit',
        domain=tuple(settings.domain), nx=settings.nx, nt=settings.nt,
        nz=settings.nz, substeps=settings.substeps, seed=settings.seed,
        stream=settings.reference_stream if reference else settings.candidate_stream,
        max_components=settings.max_components, spatial_modes=settings.spatial_modes)


def subset(p, ids):
    return {k: v[np.asarray(ids,dtype=int)] for k,v in p.items()}


def _rng(cfg, pid, sid, purpose):
    return np.random.default_rng(np.random.SeedSequence([cfg.seed,cfg.stream,pid,int(sid),purpose]))


def sample_parameters(ids, cfg, prior):
    """Same per-sample parameters under batching, output-grid and solver changes.

    Every draw samples a continuous safe set (no full-domain atom). Geometry is
    a union of ordered open intervals with strictly positive widths/gaps. The
    number of intervals stays constant in time within a realization.
    """
    cfg.validate(); spec=PRIORS[prior]; pid=spec['id']; dynamic=spec['dynamic']
    if dynamic != (cfg.mode=='fixed_deadline_exit'):
        raise ValueError('Prior and time convention disagree.')
    ids=np.asarray(ids,np.int64)
    if ids.ndim!=1 or not len(ids) or np.any(ids<0):
        raise ValueError('IDs must be nonempty and nonnegative.')
    n=len(ids); J=cfg.max_components
    # Allocate all components once using the existing parameter schema.
    p=risk.sample_parameters(ids,cfg)
    for key in ('geometry','beta','logdiff','frequency','clock'):
        p[key][...]=0.
    p['tracking'][...]=1.;p['family'][...]=-1;p['full_domain'][...]=0
    p['geometry_family']=np.zeros(n,np.int8)
    # Independent component-local dynamics from the original one-component prior.
    for c in range(J):
        stream=int(np.random.SeedSequence([cfg.stream,pid,451,c]).generate_state(1)[0])
        local=risk.sample_parameters(ids,replace(cfg,max_components=1,stream=stream))
        for key in ('beta','logdiff','frequency','tracking','family'):
            p[key][:,c]=local[key][:,0]
        if c==0:
            p['clock'][:]=local['clock']
    for i,sid in enumerate(ids):
        r=_rng(cfg,pid,sid,42)
        if spec['component']=='single':
            j=1
        elif spec['component']=='multi':
            j=int(r.integers(2,J+1))
        else:
            options=np.arange(1,J+1);weights=1/options.astype(float)**1.1
            j=int(r.choice(options,p=weights/weights.sum()))
        p['ncomp'][i]=j;m=2*j+1
        family=str(r.choice(spec['families']));p['geometry_family'][i]=FAMILY_NAMES.index(family)
        # Wider coverage of safe-set size than the old 0.63--0.92 safe mass.
        lower=max(.32,j*cfg.min_safe_width+.04)
        upper=min(.88,1-(j+1)*cfg.min_gap_width-.025)
        mass=r.uniform(lower,upper)
        widths=np.empty(m)
        widths[1::2]=(mass-j*cfg.min_safe_width)*r.dirichlet(np.full(j,r.uniform(.8,3.0)))
        widths[::2]=(1-mass-(j+1)*cfg.min_gap_width)*r.dirichlet(np.full(j+1,r.uniform(.8,3.0)))
        g=p['geometry'][i,:m];g[:,0]=np.log(widths)
        if not dynamic: continue
        amp=r.uniform(.55,1.65);phase=r.uniform(0,2*np.pi)
        harmonic=int(r.integers(1,3));scol=2*harmonic-1;ccol=2*harmonic
        wave=np.zeros(5);wave[scol]=np.cos(phase);wave[ccol]=np.sin(phase)
        if family=='swaying':
            # Opposing outer-gap logits sweep the intervals; not rigid translation.
            g[0]+=amp*wave;g[-1]-=amp*wave
            g[1::2]+=r.uniform(.05,.35)*wave
        elif family=='breathing':
            g[1::2]+=amp*wave;g[::2]-=.45*amp*wave
        elif family=='wandering':
            g[:,1:]=r.normal(size=(m,4))*amp*np.array([1.,1.,.6,.6])
        elif family=='pinching':
            g[:,1:]=r.normal(size=(m,4))*.22*np.array([1.,1.,.5,.5])
            selected=1+2*int(r.integers(j));pinch=r.uniform(1.4,3.3)
            # A smooth time-local constriction; minimum width stays positive.
            g[selected,0]-=pinch
            g[selected,3]+=pinch*np.sin(phase)
            g[selected,4]+=pinch*np.cos(phase)
    return p


@njit(cache=True,parallel=True)
def _geometry_labels(g,ncomp,full,times,nx,ms,mg):
    n=len(ncomp); nt=len(times);jm=(g.shape[1]-1)//2
    masks=np.zeros((n,nt,nx),np.uint8)
    bounds=np.full((n,nt,jm,2),np.nan)
    for i in prange(n):
        for ti in range(nt):
            a,b,_,_=risk._geometry_at(g[i],ncomp[i],full[i],times[ti],ms,mg)
            for c in range(ncomp[i]):
                bounds[i,ti,c,0]=a[c];bounds[i,ti,c,1]=b[c]
            for ix in range(nx):
                x=ix/(nx-1)
                for c in range(ncomp[i]):
                    if a[c]<x<b[c]: masks[i,ti,ix]=1
    return masks,bounds


def geometry_labels(p,cfg,aligned=True):
    axis=np.linspace(0,1,cfg.nt)
    times=1-axis if aligned and cfg.mode=='fixed_deadline_exit' else axis
    if cfg.mode=='fixed_start_exit': times=np.zeros(cfg.nt)
    mask,bounds=_geometry_labels(p['geometry'],p['ncomp'],p['full_domain'],times,
                                  cfg.nx,cfg.min_safe_width,cfg.min_gap_width)
    lo,hi=cfg.domain;bounds=lo+(hi-lo)*bounds
    return dict(mask=mask,bounds=bounds,chronological_time=times)


def time_label(cfg):
    return 'Remaining time tau = 1 - start time' if cfg.mode=='fixed_deadline_exit' else 'Horizon tau from start time 0'


def time_description(cfg):
    if cfg.mode=='fixed_deadline_exit':
        return ('R(x,tau) = P(exit C(t) before deadline 1 | X_(1-tau)=x). '
                'Rows change the START TIME. This is NOT a curve of increasing horizons from one fixed start.')
    return 'R(x,tau) = P(exit C before horizon tau | X_0=x). Rows increase the horizon from start time 0.'


def observation_design(cfg,K=8,M=256,seed=410,include_t0=False,placement='full_centers',max_time=1.):
    """No safe-set or solution argument: observations cannot adapt to hidden geometry.

    Default sensors span the full domain at bin centers and exclude endpoints,
    whose risk is one for every member of this absorbing-domain prior. Legacy
    endpoint placement is an explicit alternative, not the default.
    """
    K=int(K);M=int(M);first=0 if include_t0 else 1
    if not 0<max_time<=1: raise ValueError('Maximum time must be in (0,1].')
    if not 1<=K<=cfg.nx-2: raise ValueError(f'K must be between 1 and {cfg.nx-2}.')
    end=int(np.floor(max_time*(cfg.nt-1)+1e-12));cap=end-first+1
    if M<K or M>K*cap:
        raise ValueError(f'Infeasible: K={K}, M={M}. Allowed range {K} to {K*cap}; no silent change.')
    if placement=='full_centers':
        available=np.arange(1,cfg.nx-1)
        columns=available[np.floor((np.arange(K)+.5)*len(available)/K).astype(int)]
    elif placement=='full_endpoints':
        columns=np.rint(np.linspace(0,cfg.nx-1,K)).astype(int)
    else: raise ValueError('Unknown sensor placement.')
    r=np.random.default_rng(np.random.SeedSequence([seed,K,741]))
    weights=r.permutation(np.linspace(.4,1.6,K));lengths=np.ones(K,int)
    for _ in range(M-K):
        priority=(lengths+1)/weights;priority[lengths>=cap]=np.inf
        lengths[np.argmin(priority)]+=1
    mask=np.zeros((cfg.nt,cfg.nx),bool)
    for c,l in zip(columns,lengths):mask[first:first+l,c]=True
    assert mask.sum()==M and len(np.unique(columns))==K
    return dict(K=K,M=M,columns=columns,lengths=lengths,mask=mask,
        sensor_coords=np.argwhere(mask).astype(np.int32),include_t0=bool(include_t0),
        first_row=first,seed=int(seed),placement=placement,max_time=float(max_time))


def _source_hash():
    h=hashlib.sha256()
    for f in (Path(__file__),Path(risk.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()


def _json_clean(v):
    if isinstance(v,dict):return {str(k):_json_clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_json_clean(x) for x in v]
    if isinstance(v,np.ndarray):return _json_clean(v.tolist())
    if isinstance(v,np.generic):return _json_clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


class Engine:
    def __init__(self,settings=Settings()):
        self.settings=settings.validate();self.loaded=OrderedDict();self.generation_count=0
        self.source_hash=_source_hash()

    def _pool(self,cfg,prior,progress):
        n=self.settings.n_pool;bs=self.settings.batch_size
        manifest=dict(version=VERSION,source_hash=self.source_hash,config=asdict(cfg),prior=prior,batch_size=bs)
        text=json.dumps(manifest,sort_keys=True)
        key=hashlib.sha256(text.encode()).hexdigest()[:20]
        folder=Path(self.settings.cache_dir)/f'{prior}_{key}';folder.mkdir(parents=True,exist_ok=True)
        fp=folder/'manifest.json'
        if fp.exists() and fp.read_text()!=text: raise RuntimeError('Cache fingerprint mismatch.')
        fp.write_text(text)
        # Cache complete fixed-size blocks, so increasing N reuses earlier work.
        fields=np.empty((n,cfg.nt,cfg.nx),np.float32);parts=[];computed=0;solve_time=0.
        start_all=time.perf_counter()
        for start in range(0,n,bs):
            stop=start+bs;take=min(bs,n-start);path=folder/f'block_{start:09d}.npz'
            p=sample_parameters(np.arange(start,stop),cfg,prior)
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    u=z['risk']
                if u.shape!=(bs,cfg.nt,cfg.nx):raise RuntimeError('Bad cached shape.')
            else:
                t=time.perf_counter();u=risk.solve_parameters(p,cfg);risk.check_probabilities(u,cfg)
                solve_time+=time.perf_counter()-t;computed+=bs
                temp=path.with_suffix('.partial')
                with temp.open('wb') as f:np.savez_compressed(f,risk=u)
                temp.replace(path)
            fields[start:start+take]=u[:take];parts.append(subset(p,np.arange(take)))
            if progress and (start==0 or start+take==n or (start+take)%256==0):
                progress(f'{prior}: {start+take}/{n} candidates available')
        params={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        return fields,params,dict(n=n,newly_computed= computed,solve_seconds=solve_time,
            elapsed_seconds=time.perf_counter()-start_all,cache_dir=str(folder),source_hash=self.source_hash)

    def get(self,prior,progress=print):
        if prior not in PRIORS:raise KeyError(prior)
        if prior in self.loaded:
            self.loaded.move_to_end(prior);return self.loaded[prior]
        cfg=solver_config(self.settings,prior)
        p0=sample_parameters([9_999_999],cfg,prior)
        risk.solve_parameters(p0,cfg)  # JIT warmup excluded from pool timing
        u,p,info=self._pool(cfg,prior,progress)
        tc=solver_config(self.settings,prior,reference=True)
        tp=sample_parameters(np.arange(self.settings.n_references),tc,prior)
        truth=risk.solve_parameters(tp,tc);risk.check_probabilities(truth,tc)
        labels=geometry_labels(p,cfg);tl=geometry_labels(tp,tc)
        case=dict(prior=prior,spec=PRIORS[prior],cfg=cfg,truth_cfg=tc,pool=u,parameters=p,
                  truths=truth,truth_parameters=tp,geometry=labels,truth_geometry=tl,
                  info=info,distances={})
        self.loaded[prior]=case;self.generation_count+=1
        while len(self.loaded)>self.settings.max_loaded_priors:self.loaded.popitem(last=False)
        return case

    def result(self,prior,reference=0,K=8,M=256,epsilon=.05,include_t0=False,
               placement='full_centers',max_time=1.,display_count=6,progress=None):
        case=self.get(prior,progress);ref=int(reference)
        if not 0<=ref<len(case['truths']):raise ValueError('Reference is out of range.')
        if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('epsilon must be positive.')
        d=observation_design(case['cfg'],K,M,self.settings.observation_seed,include_t0,placement,max_time)
        key=(ref,K,M,include_t0,placement,max_time)
        truth=case['truths'][ref];target=truth[d['mask']].astype(float)
        if key not in case['distances']:
            rmse=np.empty(len(case['pool']));mx=np.empty_like(rmse)
            for s in range(0,len(rmse),128):
                diff=case['pool'][s:s+128,d['mask']].astype(float)-target
                rmse[s:s+len(diff)]=np.sqrt(np.mean(diff*diff,axis=1))
                mx[s:s+len(diff)]=np.abs(diff).max(axis=1)
            case['distances'][key]=(rmse,mx)
        rmse,mx=case['distances'][key];accepted=np.flatnonzero(rmse<=epsilon)
        if len(accepted):
            lw=-.5*(rmse[accepted]/(epsilon/3))**2;w=np.exp(lw-lw.max());w/=w.sum()
            # Uniform display subset, not top matches or diversity-selected.
            chosen=np.random.default_rng(557+ref).choice(accepted,min(display_count,len(accepted)),replace=False)
            ess=float(1/(w@w))
        else:w=np.empty(0);chosen=np.empty(0,int);ess=0.
        stats=dict(reference=ref,K=int(K),M=int(M),epsilon=float(epsilon),matches=len(accepted),
            ess=ess,nearest_rmse=float(rmse.min()),max_accepted_pixel_error=float(mx[accepted].max()) if len(accepted) else None,
            observed_zero_fraction=float(np.mean(target<.01)),observed_one_fraction=float(np.mean(target>.99)),
            min_prefix=int(d['lengths'].min()),max_prefix=int(d['lengths'].max()),
            exact_matches=False,reference_validated=False)
        return dict(case=case,reference=ref,design=d,observation=target,epsilon=float(epsilon),
                    accepted=accepted,weights=w,chosen=chosen,rmse=rmse,max_abs=mx,stats=stats)


def moments(result):
    ids=result['accepted'];case=result['case'];w=result['weights']
    if not len(ids):return None
    shape=case['pool'].shape[1:];mean=np.zeros(shape);sq=np.zeros(shape);membership=np.zeros(shape)
    for start in range(0,len(ids),64):
        chunk=ids[start:start+64];weight=w[start:start+64]
        u=case['pool'][chunk].astype(float);mask=case['geometry']['mask'][chunk]
        mean+=np.einsum('n,nij->ij',weight,u)
        sq+=np.einsum('n,nij->ij',weight,u*u)
        membership+=np.einsum('n,nij->ij',weight,mask)
    return dict(mean=mean,std=np.sqrt(np.maximum(sq-mean*mean,0)),membership=membership)


def audit(result,random_count=4):
    """Twice-fine sensitivity audit; neither modifies pool nor claims certification."""
    c=result['case'];p=c['parameters'];cfg=c['cfg'];ref=result['reference']
    ids=set(np.random.default_rng(800).choice(len(c['pool']),min(random_count,len(c['pool'])),replace=False))
    ids.update(np.argsort(np.abs(result['rmse']-result['epsilon']))[:4]);ids.update(result['chosen'][:2])
    ids=np.array(sorted(ids),int);fine=replace(cfg,nz=2*(cfg.nz-1)+1,substeps=2*cfg.substeps)
    t=time.perf_counter();u=risk.solve_parameters(subset(p,ids),fine)
    tr=risk.solve_parameters(subset(c['truth_parameters'],[ref]),fine)[0]
    risk.check_probabilities(u,fine)
    mask=result['design']['mask'];before=result['rmse'][ids]
    after=np.sqrt(np.mean((u[:,mask].astype(float)-tr[mask])**2,axis=1))
    return dict(prior=c['prior'],reference=ref,audited_candidates=len(ids),base_nz=cfg.nz,fine_nz=fine.nz,
        candidate_field_max_change=float(np.max(np.abs(u-c['pool'][ids]))),
        truth_field_max_change=float(np.max(np.abs(tr-c['truths'][ref]))),
        observation_rmse_change_max=float(np.max(np.abs(after-before))),
        match_decision_changes=int(np.sum((before<=result['epsilon'])!=(after<=result['epsilon']))),
        seconds=time.perf_counter()-t,reference_validated=False)


def export_case(result,folder,sample_count=10):
    """Public observations never include ground-truth safe geometry or dynamics.
    Retains all original generating parameters in hidden and ensemble files.
    """
    if len(result['accepted'])<sample_count:
        raise ValueError(f'Only {len(result["accepted"])} matches; need {sample_count}. No duplicates or padding.')
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    c=result['case'];cfg=c['cfg'];r=result['reference'];d=result['design']
    pick=np.random.default_rng(992).choice(len(result['accepted']),sample_count,replace=False)
    ids=result['accepted'][pick];weights=result['weights'][pick];mass=float(weights.sum());weights=weights/mass
    meta=dict(version=VERSION,prior=c['prior'],known='domain, event, declared prior and observation design',
        unknown='safe set and dynamics',cfg=asdict(cfg),truth_cfg=asdict(c['truth_cfg']),
        time_convention=time_description(cfg),kernel='truncated Gaussian on RMSE, bandwidth epsilon/3',
        epsilon=result['epsilon'],include_t0=d['include_t0'],K=d['K'],M=d['M'],reference=r,
        retained_full_pool_weight=mass,selection='uniform unique subset of accepted candidates',
        reference_validated=False)
    np.savez_compressed(folder/'public_observations.npz',x=np.linspace(*cfg.domain,cfg.nx),
        tau=np.linspace(0,1,cfg.nt),sensor_coords=d['sensor_coords'],observation_values=result['observation'],
        observation_mask=d['mask'],initial_state_columns=d['columns'],prefix_lengths=d['lengths'])
    trp=subset(c['truth_parameters'],[r]);hidden={f'parameter__{k}':v for k,v in trp.items()}
    hidden.update(u_ref=c['truths'][r],safe_mask_ref=c['truth_geometry']['mask'][r],
                  safe_bounds_ref=c['truth_geometry']['bounds'][r])
    np.savez_compressed(folder/'hidden_evaluation_labels.npz',**hidden)
    ensemble={f'parameter__{k}':v for k,v in subset(c['parameters'],ids).items()}
    ensemble.update(posterior_samples_u=c['pool'][ids],posterior_weights=weights,
        posterior_safe_masks=c['geometry']['mask'][ids],posterior_safe_bounds=c['geometry']['bounds'][ids],
        sample_ids=c['parameters']['sample_id'][ids],observation_rmse=result['rmse'][ids])
    np.savez_compressed(folder/'matching_ensemble_PREVIEW.npz',**ensemble)
    (folder/'metadata.json').write_text(json.dumps(_json_clean(meta),indent=2))
    return folder
