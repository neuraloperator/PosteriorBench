"""Moving-safe-set risk completion: fixed observations, independent prior rejection.

Public observations never contain geometry, dynamics, reference seeds, or full
reference fields. Private parameters and evaluator-only targets live separately.
No case is swapped, no threshold is relaxed, and no observations are overwritten.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import hashlib, json, math, os, platform, time, uuid
import numpy as np
import risk1d as risk
import safe_explorer_core as prior
import screen_multi as screen

VERSION = 'risk-quantification-1.0'
PROFILES = {'standard': (513,16), 'fine': (1025,32), 'very-fine': (2049,64)}
DEFAULTS = dict(version=VERSION, seed=925, candidate_stream=202, reference_stream=101,
    observation_seed=410, cases=20, k_values=list(range(6,13)), M=256, epsilon=0.05,
    nx=128, nt=128, domain=[0.,1.], max_components=5, spatial_modes=8,
    nz=513, substeps=16, profile='standard', block_size=1000)


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,Path): return str(v)
    return v


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


@contextmanager
def exclusive(path):
    """Fail clearly rather than letting two workers overwrite the same block."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try: fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError(f'Lock exists: {path}. Check for a running worker; remove only a stale lock.') from None
    try:
        with os.fdopen(fd,'w') as f: f.write(f'pid={os.getpid()} host={platform.node()}\n')
        yield
    finally: path.unlink(missing_ok=True)


def source_hash():
    h=hashlib.sha256()
    for name in ('risk1d.py','safe_explorer_core.py','exact_screen.py','screen_multi.py'):
        h.update((Path(__file__).parent/name).read_bytes().rstrip(b'\n'))
    return h.hexdigest()


def solver_config(spec,reference=False):
    return risk.Config(mode='fixed_deadline_exit',domain=tuple(spec['domain']),
        nx=spec['nx'],nt=spec['nt'],nz=spec['nz'],substeps=spec['substeps'],
        seed=spec['seed'],stream=spec['reference_stream'] if reference else spec['candidate_stream'],
        max_components=spec['max_components'],spatial_modes=spec['spatial_modes'])


def validate_spec(spec):
    if spec['cases']<1: raise ValueError('cases must be positive')
    if spec['M']!=256 or spec['epsilon']!=0.05: raise ValueError('This observation design requires M=256 and epsilon=0.05.')
    if not spec['k_values'] or any(k<6 or k>12 for k in spec['k_values']):
        raise ValueError('K values must be between 6 and 12.')
    if len(set(spec['k_values']))!=len(spec['k_values']): raise ValueError('Duplicate K values.')
    if spec['reference_stream']==spec['candidate_stream']: raise ValueError('Independent streams required.')
    if spec['block_size']<1: raise ValueError('block_size must be positive')
    solver_config(spec).validate()


def init_run(root,spec=None):
    """Fix cases and K assignments BEFORE searching. A rerun never replaces cases."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfgspec=dict(DEFAULTS if spec is None else spec);validate_spec(cfgspec)
    request=dict(**cfgspec,source_sha256=source_hash(),prior='moving_mix',
        case_selection=cfgspec.get('reference_selection', 'consecutive independent reference draws; no match-count selection'),
        time_axis='remaining_time_to_deadline_1',candidate_selection='independent prior rejection; RMSE <= 0.05')
    manifest=root/'internal'/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=request:
            raise RuntimeError('Existing run has a different configuration or numerical source. Use a new run directory.')
    else: atomic_json(manifest,request)
    cfg=solver_config(request,True);n=request['cases']
    ks=np.array([request['k_values'][i%len(request['k_values'])] for i in range(n)],np.int32)
    masks=np.stack([prior.observation_design(cfg,int(k),256,request['observation_seed'])['mask'] for k in ks])
    refpath=root/'internal'/'references.npz'
    if not refpath.exists():
        with exclusive(root/'internal'/'references.lock'):
            reference_ids=np.asarray(request.get('reference_ids',list(range(n))),np.int64)
            if reference_ids.shape!=(n,) or len(np.unique(reference_ids))!=n:
                raise ValueError('Reference IDs must be unique with one ID per case.')
            p=prior.sample_parameters(reference_ids,cfg,'moving_mix')
            u=risk.solve_parameters(p,cfg);risk.check_probabilities(u,cfg)
            atomic_npz(refpath,risk=u,K=ks,masks=masks,**{'parameter__'+k:v for k,v in p.items()})
    refs=load_references(root)
    if not np.array_equal(refs['K'],ks) or not np.array_equal(refs['masks'],masks):
        raise RuntimeError('Reference/mask metadata inconsistent with run manifest.')
    for i,k in enumerate(ks):
        design=prior.observation_design(cfg,int(k),256,request['observation_seed'])
        path=root/'public'/f'case_{i:04d}.npz'
        if not path.exists():
            atomic_npz(path,case_id=np.asarray(f'case_{i:04d}'),
                x=np.linspace(*cfg.domain,cfg.nx),tau=np.linspace(0.,1.,cfg.nt),
                chronological_start_time=1-np.linspace(0.,1.,cfg.nt),
                sensor_coords=design['sensor_coords'],observation_mask=design['mask'],
                observation_values=refs['risk'][i,design['mask']],
                initial_state_columns=design['columns'],prefix_lengths=design['lengths'],K=np.asarray(k))
    atomic_json(root/'public'/'README.json',dict(
        quantity='first-exit risk; safety = 1 - risk',
        time_convention='R(x,tau) = P(exit C(t) by deadline 1 | X_(1-tau)=x)',
        meaning='tau is remaining time; it is not a fixed-start horizon',
        M=256,epsilon=0.05,include_tau_zero=False,
        inputs='Only sensor coordinates, values, mask and global grids. No geometry or dynamics supplied.',
        likelihood='Exploratory truncated RMSE kernel exp(-0.5*(RMSE/(0.05/3))**2).'))
    return request


def load_spec(root):
    spec=json.loads((Path(root)/'internal'/'manifest.json').read_text())
    if spec['source_sha256']!=source_hash():
        raise RuntimeError('Numerical source changed. Start a new run instead of mixing solvers.')
    return spec


def load_references(root):
    with np.load(Path(root)/'internal'/'references.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def reference_parameters(refs):
    return {k[len('parameter__'):]:v for k,v in refs.items() if k.startswith('parameter__')}


def distances(fields,truths,masks):
    out=np.empty((len(fields),len(truths)),np.float64)
    for r,mask in enumerate(masks):
        out[:,r]=np.sqrt(np.mean((fields[:,mask].astype(np.float64)-truths[r,mask].astype(np.float64))**2,axis=1))
    return out


def compute_block(spec,refs,start,stop):
    """One sampled setting can match more than one reference. Solve it only once."""
    cfg=solver_config(spec);ids=np.arange(start,stop,dtype=np.int64)
    t=time.perf_counter();p=prior.sample_parameters(ids,cfg,'moving_mix');parameter_s=time.perf_counter()-t
    t=time.perf_counter()
    bound=screen.bounds_for_cases(p,cfg,refs['masks'],refs['risk'],refs['K'])
    limit=256*.05**2;eligible=np.flatnonzero((bound<=limit+1e-10).any(axis=1))
    geometry_s=time.perf_counter()-t
    t=time.perf_counter()
    if len(eligible):
        err,stops=screen.prefix(prior.subset(p,eligible),cfg,refs['masks'],refs['risk'],bound[eligible],.05)
        keep=eligible[np.flatnonzero((err<=limit+1e-10).any(axis=1))]
    else: keep=np.empty(0,int);stops=np.empty(0,int)
    prefix_s=time.perf_counter()-t;t=time.perf_counter()
    if len(keep):
        fields=risk.solve_parameters(prior.subset(p,keep),cfg);risk.check_probabilities(fields,cfg)
        dist=distances(fields,refs['risk'],refs['masks'])
        good=(dist<=.05).any(axis=1);fields=fields[good];dist=dist[good];keep=keep[good]
    else: fields=np.empty((0,cfg.nt,cfg.nx),np.float32);dist=np.empty((0,len(refs['risk'])))
    full_s=time.perf_counter()-t
    diag=dict(start=start,stop=stop,candidate_settings=stop-start,geometry_passes=len(eligible),
        prefix_solves=len(eligible),complete_matching_fields=len(keep),
        mean_prefix_rows=float(stops.mean()) if len(stops) else 0,
        parameter_seconds=parameter_s,geometry_seconds=geometry_s,prefix_seconds=prefix_s,
        complete_solve_seconds=full_s,compute_seconds=parameter_s+geometry_s+prefix_s+full_s)
    return dict(ids=ids[keep],risk=fields,distances=dist,diagnostics=np.asarray(json.dumps(clean(diag))))


def warmup(spec,refs):
    t=time.perf_counter();cfg=solver_config(spec)
    p=prior.sample_parameters([987654321],cfg,'moving_mix')
    # Force the prefix signature to compile, even when a candidate is bounded out.
    bounds=screen.bounds_for_cases(p,cfg,refs['masks'],refs['risk'],refs['K'])
    screen.prefix(p,cfg,refs['masks'],refs['risk'],np.zeros_like(bounds),.05)
    risk.solve_parameters(p,cfg)
    return time.perf_counter()-t


def block_paths(root):
    return sorted((Path(root)/'internal'/'blocks').glob('block_*.npz'))


def status(root,matches=10,write=True):
    spec=load_spec(root);refs=load_references(root);counts=np.zeros(spec['cases'],np.int64)
    searched=0;contiguous=0;compute=0.;complete_fields=0;blocks=0;timings=[]
    for path in block_paths(root):
        with np.load(path,allow_pickle=False) as z:
            d=z['distances'];diag=json.loads(str(z['diagnostics']))
            if d.shape[1]!=spec['cases']: raise RuntimeError(f'Incompatible block {path}')
            counts+=(d<=.05).sum(axis=0)
            searched+=diag['candidate_settings'];complete_fields+=diag['complete_matching_fields']
            compute+=diag['compute_seconds'];timings.append(diag);blocks+=1
            if diag['start']==contiguous: contiguous=diag['stop']
    rows=[]
    for i,k in enumerate(refs['K']):
        y=refs['risk'][i,refs['masks'][i]];c=int(counts[i])
        # m / p is a crude point projection, not a promise; zero counts stay unknown.
        estimate=int(math.ceil(matches*searched/c)) if c and searched else None
        rows.append(dict(case_id=f'case_{i:04d}',reference_id=i,K=int(k),matches=c,
            target=int(matches),ready=c>=matches,observed_below_001=float(np.mean(y<.01)),
            observed_above_099=float(np.mean(y>.99)),
            estimated_candidate_settings_for_target=estimate))
    result=dict(target_matches=matches,cases=spec['cases'],ready_cases=int((counts>=matches).sum()),
        all_cases_ready=bool((counts>=matches).all()),candidate_settings_processed=searched,
        contiguous_candidate_prefix=contiguous,completed_blocks=blocks,
        complete_matching_fields=complete_fields,cumulative_compute_seconds=compute,rows=rows,
        caution='Rates depend on the observation case. Zero matches do not justify a finite completion-time estimate.')
    if write: atomic_json(Path(root)/'status.json',result)
    return result


def search(root,max_candidates=100000,matches=10,shard_index=0,num_shards=1,max_blocks=None):
    """Resume numbered blocks. Do not concurrently run this and init/export.
    SLURM arrays assign disjoint block numbers using block_index % num_shards.
    Sequential mode stops once every fixed case has the requested count.
    """
    spec=load_spec(root);refs=load_references(root);bs=spec['block_size']
    if min(max_candidates,matches,num_shards)<1 or not 0<=shard_index<num_shards:
        raise ValueError('Invalid search budget or shard assignment.')
    if max_candidates%bs: raise ValueError('max_candidates must be a multiple of block_size.')
    ready=status(root,matches)
    if ready['all_cases_ready'] and num_shards==1:
        print('All fixed cases already meet the requested target.',flush=True);return ready
    nnew=0;t_all=time.perf_counter()
    print(f'CPU warmup: {warmup(spec,refs):.2f}s (excluded from throughput)',flush=True)
    for bi in range(shard_index,max_candidates//bs,num_shards):
        start=bi*bs;stop=start+bs;path=Path(root)/'internal'/'blocks'/f'block_{start:010d}_{stop:010d}.npz'
        if path.exists(): continue
        if max_blocks is not None and nnew>=max_blocks: break
        with exclusive(path.with_suffix('.lock')):
            t=time.perf_counter();out=compute_block(spec,refs,start,stop)
            atomic_npz(path,**out);elapsed=time.perf_counter()-t
        nnew+=1;st=status(root,matches)
        print(f'IDs [{start},{stop}) {elapsed:.2f}s | ready {st["ready_cases"]}/{spec["cases"]} | counts '+
              str([r['matches'] for r in st['rows']]),flush=True)
        if st['all_cases_ready'] and num_shards==1: break
    result=status(root,matches)
    result['this_invocation_wall_seconds']=time.perf_counter()-t_all
    atomic_json(Path(root)/f'last_search_worker_{shard_index}.json',result)
    return result


def weights_from_rmse(rmse):
    logw=-.5*(np.asarray(rmse,dtype=float)/(.05/3))**2
    w=np.exp(logw-logw.max());return w/w.sum()


def collect_matches(root,case_index,limit=None):
    spec=load_spec(root);arrays=[];ids=[];rmse=[]
    for path in block_paths(root):
        with np.load(path,allow_pickle=False) as z:
            d=z['distances'][:,case_index];ok=d<=.05
            arrays.append(z['risk'][ok]);ids.append(z['ids'][ok]);rmse.append(d[ok])
    if not arrays:return np.empty((0,spec['nt'],spec['nx']),np.float32),np.empty(0,np.int64),np.empty(0)
    u=np.concatenate(arrays);ii=np.concatenate(ids);dd=np.concatenate(rmse)
    order=np.argsort(ii)
    if limit is not None:order=order[:limit]
    return u[order],ii[order],dd[order]


def export_dataset(root,matches=10,allow_partial=False):
    """First m unique prior-accepted candidates, never the closest m.
    Inference inputs, evaluation targets and optional private generator labels
    are separate directories. No geometry/dynamics enter the benchmark package.
    """
    root=Path(root);st=status(root,matches);spec=load_spec(root);refs=load_references(root)
    if st['contiguous_candidate_prefix']!=st['candidate_settings_processed']:
        raise RuntimeError('Candidate blocks have gaps. Complete the missing blocks before exporting first accepted samples.')
    if not st['all_cases_ready'] and not allow_partial:
        raise RuntimeError(f'Only {st["ready_cases"]}/{spec["cases"]} cases have {matches} matches. '
                           'Increase the search budget; no incomplete case is silently substituted.')
    folder=root/f'export_{matches:03d}';folder.mkdir(exist_ok=True)
    manifest_rows=[]
    for i,row in enumerate(st['rows']):
        if not row['ready']: continue
        u,ids,dd=collect_matches(root,i,matches);w=weights_from_rmse(dd)
        mask=refs['masks'][i];true=refs['risk'][i]
        errmax=np.abs(u[:,mask].astype(float)-true[mask]).max(axis=1)
        public=root/'public'/f'case_{i:04d}.npz'
        with np.load(public,allow_pickle=False) as z: pub={k:z[k] for k in z.files}
        if not np.array_equal(pub['observation_values'],true[mask]):
            raise RuntimeError('Public observations do not match the saved reference field.')
        atomic_npz(folder/'inputs'/public.name,**pub)
        atomic_npz(folder/'evaluation'/public.name,u_ref=true,posterior_samples_u=u,posterior_weights=w,
            observation_rmse=dd,maximum_observed_pixel_error=errmax)
        # Stored outside the inputs/evaluation directories to prevent seed/geometry leakage.
        pp=prior.sample_parameters(ids,solver_config(spec),'moving_mix')
        atomic_npz(root/'internal'/'export_provenance'/public.name,sample_ids=ids,
            **{'parameter__'+k:v for k,v in pp.items()})
        manifest_rows.append(dict(case_id=f'case_{i:04d}',K=int(refs['K'][i]),sample_count=len(u),
            accepted_pool_count=row['matches'],ess=float(1/(w@w)),max_rmse=float(dd.max()),
            max_observed_pixel_error=float(errmax.max()),
            mean_posterior_pointwise_std=float(np.sqrt(np.einsum('n,nij->ij',w,(u.astype(float)-np.einsum('n,nij->ij',w,u.astype(float)))**2)).mean())))
    output=dict(quantity='risk probability',time_axis='remaining_time_to_deadline_1',
        M=256,epsilon=.05,requested_cases=spec['cases'],exported_cases=len(manifest_rows),
        complete=len(manifest_rows)==spec['cases'],matches_per_case=matches,
        selection='First m accepted prior draws in candidate-ID order; no nearest/diversity selection',
        weighting='exp(-0.5*(observed_RMSE/(0.05/3))**2), normalized within exported ensemble',
        metadata_excludes='No safe masks, interval boundaries, dynamics or random seeds in benchmark inputs/evaluation.',
        validation_status='Exploratory finite-sample posterior. Match count is not a convergence certificate.',cases=manifest_rows)
    atomic_json(folder/'dataset_manifest.json',output)
    return folder,output


def numerical_audit(root,n_random=16,n_matches=16,include_very_fine=False):
    spec=load_spec(root);refs=load_references(root);cfg=solver_config(spec)
    refp=reference_parameters(refs)
    ids=set(range(8000000,8000000+n_random))
    for i in range(spec['cases']):
        _,ii,_=collect_matches(root,i,limit=max(1,n_matches//spec['cases']))
        ids.update(ii.tolist())
    # Add some candidates immediately inside the threshold where a flip matters most.
    for path in block_paths(root):
        with np.load(path,allow_pickle=False) as z:
            d=z['distances'];near=np.min(np.abs(d-.05),axis=1)
            ids.update(z['ids'][np.argsort(near)[:2]].tolist())
        if len(ids)>=n_random+2*n_matches:break
    ids=np.array(sorted(ids),np.int64);p=prior.sample_parameters(ids,cfg,'moving_mix')
    profiles=[('base',cfg),('fine',replace(cfg,nz=2*(cfg.nz-1)+1,substeps=cfg.substeps*2))]
    if include_very_fine:profiles.append(('very_fine',replace(cfg,nz=4*(cfg.nz-1)+1,substeps=cfg.substeps*4)))
    fields=[];truths=[];times=[];comparisons=[]
    # Warm identical signatures before measuring.
    risk.solve_parameters(prior.subset(p,[0]),cfg)
    for label,cc in profiles:
        t=time.perf_counter();u=risk.solve_parameters(p,cc);tr=risk.solve_parameters(refp,cc)
        elapsed=time.perf_counter()-t;risk.check_probabilities(u,cc);risk.check_probabilities(tr,cc)
        times.append(dict(profile=label,nz=cc.nz,substeps=cc.substeps,
            candidate_fields=len(u),reference_fields=len(tr),seconds=elapsed))
        fields.append(u);truths.append(tr)
    for j in range(1,len(profiles)):
        u0,u1=fields[j-1:j+1];t0,t1=truths[j-1:j+1]
        d0=distances(u0,t0,refs['masks']);d1=distances(u1,t1,refs['masks'])
        per_field=np.abs(u1-u0).max(axis=(1,2));per_ref=np.abs(t1-t0).max(axis=(1,2))
        per_obs=np.array([np.sqrt(np.mean((t1[i,m].astype(float)-t0[i,m])**2)) for i,m in enumerate(refs['masks'])])
        rows=[]
        for i in range(spec['cases']):
            w0=weights_from_rmse(d0[:,i]);w1=weights_from_rmse(d1[:,i])
            rows.append(dict(reference_id=i,K=int(refs['K'][i]),
                gt_field_max_change=float(per_ref[i]),gt_observed_rmse_change=float(per_obs[i]),
                candidate_rmse_max_change=float(np.max(np.abs(d1[:,i]-d0[:,i]))),
                acceptance_flips=int(np.sum((d0[:,i]<=.05)!=(d1[:,i]<=.05)))))
        comparisons.append(dict(from_profile=profiles[j-1][0],to_profile=profiles[j][0],
            candidate_field_max_change=float(per_field.max()),candidate_field_max_p95=float(np.quantile(per_field,.95)),
            gt_field_max_change=float(per_ref.max()),gt_observed_rmse_max_change=float(per_obs.max()),
            matching_rmse_max_change=float(np.max(np.abs(d1-d0))),
            acceptance_flips=int(np.sum((d0<=.05)!=(d1<=.05))),
            max_RMSE_kernel_logweight_change=float(np.max(np.abs(d1*d1-d0*d0))/(2*(.05/3)**2)),
            time_ratio=times[j]['seconds']/times[j-1]['seconds'],per_case=rows))
    report=dict(candidate_ids=ids,timings=times,comparisons=comparisons,
        note='Paired full solves including references, accepted, near-threshold and independent prior cases. '
             'Audit does not alter the search/observation model. No continuum error certificate.')
    atomic_json(Path(root)/'numerical_audit.json',report)
    return clean(report)


def benchmark(root,n=1000,start=7000000):
    spec=load_spec(root);refs=load_references(root)
    compile_s=warmup(spec,refs);t=time.perf_counter()
    out=compute_block(spec,refs,start,start+n)
    compute_wall=time.perf_counter()-t
    path=Path(root)/'benchmark_tmp.npz';t=time.perf_counter();atomic_npz(path,**out)
    io=time.perf_counter()-t;size=path.stat().st_size;path.unlink()
    diag=json.loads(str(out['diagnostics']));rate=n/(compute_wall+io)
    hardware=dict(platform=platform.platform(),processor=platform.processor(),visible_cpu_count=os.cpu_count())
    try:hardware['cpu_quota']=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    except OSError:pass
    try:
        import numba;hardware['numba_threads']=numba.get_num_threads()
    except ImportError:pass
    result=dict(hardware=hardware,profile=spec['profile'],cases=spec['cases'],K_values=spec['k_values'],
        candidates=n,compile_warmup_seconds=compile_s,wall_compute_seconds=compute_wall,
        write_seconds=io,benchmark_file_bytes=size,settings_per_second=rate,
        projection_100k_hours=100000/rate/3600,projection_1M_hours=1000000/rate/3600,
        matches_per_case=(out['distances']<=.05).sum(axis=0),details=diag,
        note='Small measured workload; extrapolations are time for a candidate budget, not time to complete all cases.')
    atomic_json(Path(root)/'benchmark.json',result)
    return clean(result)
