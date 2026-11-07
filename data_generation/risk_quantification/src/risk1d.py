"""Reproducible 1D first-exit probability PDE dataset generator.

No trained networks, trajectories, probability clipping, or image warping.
The two modes have different, explicitly recorded time semantics.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import os
import time
from typing import Optional

import numpy as np
from numba import njit, prange

VERSION = "risk1d-0.1.0"
FAMILY_NAMES = ("diffusive", "restoring", "repelling", "bistable", "random_field")

@dataclass(frozen=True)
class Config:
    mode: str = "fixed_start_exit"  # or "fixed_deadline_exit"
    domain: tuple[float, float] = (0.0, 1.0)
    nx: int = 64
    nt: int = 64
    nz: int = 129  # internal points PER safe component, including endpoints
    substeps: int = 8  # internal substeps per exported time interval
    integrator: str = "rannacher_cn"  # or "implicit_euler"
    max_components: int = 5
    spatial_modes: int = 8
    min_safe_width: float = 0.045  # normalized to total physical domain width
    min_gap_width: float = 0.008
    seed: int = 925
    stream: int = 0  # use different streams for train, test, reference prior

    def validate(self):
        if self.integrator not in ("rannacher_cn", "implicit_euler"):
            raise ValueError("Unknown integrator")
        if self.mode not in ("fixed_start_exit", "fixed_deadline_exit"):
            raise ValueError("Unrecognized mode")
        if len(self.domain) != 2 or not self.domain[0] < self.domain[1]:
            raise ValueError("domain must be an increasing pair")
        if min(self.nx, self.nt) < 3 or self.nz < 9 or self.substeps < 1:
            raise ValueError("Invalid resolution")
        if not 1 <= self.max_components <= 8 or self.spatial_modes < 1:
            raise ValueError("Invalid prior dimensions")
        if self.min_safe_width <= 0 or self.min_gap_width <= 0:
            raise ValueError("Minimum widths must be positive")
        floor = self.max_components*self.min_safe_width + (self.max_components+1)*self.min_gap_width
        if floor >= 0.55:
            raise ValueError("Minimum interval widths leave too little prior flexibility")
        if self.seed < 0 or self.stream < 0:
            raise ValueError("seed and stream must be nonnegative")
        return self

    @property
    def time_semantics(self):
        if self.mode == "fixed_start_exit":
            return "tau=horizon; initial chronological time=0"
        return "tau=1-start_time; terminal chronological time=1"


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _random_fourier(rng, k, amplitude):
    """Finite-bandwidth random field; hyperparameters randomized per draw."""
    active = int(rng.integers(2, k+1)) if k > 1 else 1
    smoothness = rng.uniform(0.5, 2.4)
    w = np.arange(1, k+1, dtype=float)**(-smoothness)
    w[active:] = 0
    w /= np.sqrt(np.sum(w*w))
    return amplitude * rng.normal(size=(2, k)) * w[None, :]


def sample_parameters(ids, cfg: Config):
    """Parameters depend only on (version, config prior, seed, stream, sample id).

    Solver resolution, batching, worker count, and restart boundaries do not
    change a sample's parameters. No solution-dependent rejection is performed.
    """
    cfg.validate()
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0):
        raise ValueError("ids must be a nonempty 1D array of nonnegative integers")
    n, jmax, k = len(ids), cfg.max_components, cfg.spatial_modes
    f = 3 + 2*k
    p = {
        "sample_id": ids.copy(),
        "ncomp": np.zeros(n, np.int16),
        "full_domain": np.zeros(n, np.uint8),
        "geometry": np.zeros((n, 2*jmax+1, 5), np.float64),
        "beta": np.zeros((n, jmax, 3, f), np.float64),
        "logdiff": np.zeros((n, jmax, 3, f), np.float64),
        "frequency": np.zeros((n, jmax), np.float64),
        "clock": np.zeros((n, 5), np.float64),
        "tracking": np.ones((n, jmax), np.float64),
        "family": np.full((n, jmax), -1, np.int8),
    }
    for i, sid in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, cfg.stream, int(sid)]))
        # A mixture over topology, not a function of the generated solution.
        weights = 1 / np.sqrt(np.arange(1, jmax+1, dtype=float))
        j = int(rng.choice(np.arange(1, jmax+1), p=weights/weights.sum()))
        p["ncomp"][i] = j
        full = j == 1 and rng.random() < 0.40
        p["full_domain"][i] = full
        m = 2*j+1
        safe_mass = rng.uniform(0.63, 0.92)
        excess = np.empty(m)
        excess[1::2] = (safe_mass-j*cfg.min_safe_width)*rng.dirichlet(np.full(j, 2.5))
        excess[0::2] = (1-safe_mass-(j+1)*cfg.min_gap_width)*rng.dirichlet(np.full(j+1, 2.5))
        p["geometry"][i, :m, 0] = np.log(excess)
        if cfg.mode == "fixed_deadline_exit" and not full:
            amp = rng.uniform(0.15, 1.0)
            p["geometry"][i, :m, 1:] = rng.normal(size=(m,4))*amp*np.array([1,1,.35,.35])
        # Physical time change is common to all generator coefficients.
        p["clock"][i,0] = np.log(_loguniform(rng, .25, 2.5))
        p["clock"][i,1:] = rng.normal(size=4)*rng.uniform(0,.7)*np.array([1,1,.35,.35])
        for c in range(j):
            family = int(rng.choice(5, p=[.15,.25,.12,.18,.30]))
            p["family"][i,c] = family
            b = p["beta"][i,c,0]
            mid = rng.uniform(.30,.70)
            if family == 0:
                b[0] = rng.uniform(-.10,.10)
            elif family == 1:
                strength = _loguniform(rng,.3,3.)
                b[0], b[1] = strength*(mid-.5), -strength
            elif family == 2:
                strength = _loguniform(rng,.2,1.5)
                b[0], b[1] = strength*(.5-mid), strength
            elif family == 3:
                # Cubic bistable drift around z=1/2, with a bias.
                strength = _loguniform(rng,2.,12.)
                radius = rng.uniform(.15,.35)
                b[0], b[1], b[2] = rng.uniform(-.08,.08), strength*radius**2, -strength
            else:
                b[0] = rng.uniform(-.4,.4)
            amp = _loguniform(rng,.05,1.0) if family == 4 else _loguniform(rng,.005,.12)
            b[3:] = _random_fourier(rng,k,amp).reshape(-1)
            d = p["logdiff"][i,c,0]
            d[0] = np.log(_loguniform(rng,.002,.16))
            d[3:] = _random_fourier(rng,k,rng.uniform(.10,.75)).reshape(-1)
            p["frequency"][i,c] = rng.uniform(.4,2.5)
            if cfg.mode == "fixed_deadline_exit":
                for rank in (1,2):
                    p["beta"][i,c,rank,0] = rng.normal()*rng.uniform(.05,.45)
                    p["beta"][i,c,rank,3:] = _random_fourier(rng,k,rng.uniform(.03,.30)).reshape(-1)
                    p["logdiff"][i,c,rank,0] = rng.normal()*.25
                    p["logdiff"][i,c,rank,3:] = _random_fourier(rng,k,rng.uniform(.02,.25)).reshape(-1)
                # Tracking is a physical modeling choice, recorded as a label.
                u = rng.random()
                p["tracking"][i,c] = 1.0 if u < .60 else (rng.uniform(.85,1.0) if u < .88 else rng.uniform(0,.75))
    return p


@njit(cache=True)
def _time_basis(t):
    v = np.empty(5)
    dv = np.empty(5)
    v[0], dv[0] = 1.,0.
    for k in range(1,3):
        w = math.pi*k
        v[2*k-1],v[2*k] = math.sin(w*t),math.cos(w*t)
        dv[2*k-1],dv[2*k] = w*math.cos(w*t),-w*math.sin(w*t)
    return v,dv


@njit(cache=True)
def _geometry_at(g, ncomp, full, t, min_safe, min_gap):
    """Normalized endpoints AND chronological-time derivatives."""
    a,b,ad,bd = np.empty(ncomp),np.empty(ncomp),np.empty(ncomp),np.empty(ncomp)
    if full:
        a[0],b[0],ad[0],bd[0] = 0.,1.,0.,0.
        return a,b,ad,bd
    m = 2*ncomp+1
    v,dv = _time_basis(t)
    logits,ld = np.zeros(m),np.zeros(m)
    for r in range(m):
        for k in range(5):
            logits[r] += g[r,k]*v[k]
            ld[r] += g[r,k]*dv[k]
    w = np.exp(logits-np.max(logits))
    w /= w.sum()
    mean_ld = np.dot(w,ld)
    free = 1 - ncomp*min_safe - (ncomp+1)*min_gap
    pos,vel,c = 0.,0.,0
    for r in range(m):
        width = (min_safe if r%2 else min_gap) + free*w[r]
        wd = free*w[r]*(ld[r]-mean_ld)
        if r%2:
            a[c],b[c],ad[c],bd[c] = pos,pos+width,vel,vel+wd
            c += 1
        pos += width
        vel += wd
    return a,b,ad,bd


@njit(cache=True)
def _basis(z, k):
    f = np.empty(3+2*k)
    f[0],f[1],f[2] = 1.,z-.5,(z-.5)**3
    for j in range(k):
        f[3+j] = math.sin(math.pi*(j+1)*z)
        f[3+k+j] = math.cos(math.pi*(j+1)*z)
    return f


@njit(cache=True)
def _rates(b, d, h):
    """Exponential-fitted positive rates for b*d/dz + d*d2/dz2.

    This is the backward NONDIVERGENCE operator. No derivative of d is added.
    """
    p = b*h/d
    scale = d/(h*h)
    if abs(p) < 1e-5:
        plus = scale*(1+p/2+p*p/12)
        minus = scale*(1-p/2+p*p/12)
    elif p > 50:
        plus,minus = b/h,scale*p*math.exp(-p)
    elif p < -50:
        plus,minus = -scale*p*math.exp(p),-b/h
    else:
        plus = scale*(-p)/math.expm1(-p)
        minus = scale*p/math.expm1(p)
    return plus,minus


@njit(cache=True)
def _linear_step(old,plus,minus,oldplus,oldminus,dt,theta):
    n=len(old)
    cp,rhs=np.empty(n-2),np.empty(n-2)
    new=np.zeros(n)
    for ii in range(n-2):
        i=ii+1
        explicit=old[i]+(1-theta)*dt*(oldplus[i]*(old[i+1]-old[i])+oldminus[i]*(old[i-1]-old[i]))
        lo,diag,up=-theta*dt*minus[i],1+theta*dt*(plus[i]+minus[i]),-theta*dt*plus[i]
        denom=diag if ii==0 else diag-lo*cp[ii-1]
        cp[ii]=up/denom
        rhs[ii]=(explicit if ii==0 else explicit-lo*rhs[ii-1])/denom
    new[n-2]=rhs[n-3]
    for ii in range(n-4,-1,-1):
        new[ii+1]=rhs[ii]-cp[ii]*new[ii+2]
    return new


@njit(cache=True)
def _component_rates(sb,sd,freq,tracking,dynamic,t,speed,a,b,ad,bd,nz):
    plus,minus=np.zeros(nz),np.zeros(nz)
    h=1/(nz-1)
    ts=math.sin(math.pi*freq*t) if dynamic else 0.
    tc=math.cos(math.pi*freq*t) if dynamic else 0.
    for i in range(1,nz-1):
        drift=speed*(sb[0,i]+ts*sb[1,i]+tc*sb[2,i])
        if dynamic:
            drift+=(tracking-1)*(ad+(bd-ad)*i*h)/(b-a)
        diffusion=speed*math.exp(sd[0,i]+ts*sd[1,i]+tc*sd[2,i])
        plus[i],minus[i]=_rates(drift,diffusion,h)
    return plus,minus


@njit(cache=True)
def _solve_one(ncomp, full, geometry, beta, logdiff, frequency, clock,
               tracking, dynamic, nx, nt, nz, substeps, min_safe, min_gap, use_cn):
    """Solve survival on exact component-local moving coordinates.

    Positive-rate birth/death approximation with Rannacher/CN or Euler.
    CN steps that violate bounds or required monotonicity use two implicit
    Euler half-steps instead. No post-hoc probability clipping.
    """
    k = (beta.shape[2]-3)//2
    h,dt = 1/(nz-1),1/((nt-1)*substeps)
    survival = np.ones((ncomp,nz))
    survival[:,0] = 0.
    survival[:,-1] = 0.
    spatial_b,spatial_d = np.empty((ncomp,3,nz)),np.empty((ncomp,3,nz))
    for c in range(ncomp):
        for i in range(nz):
            bas = _basis(i*h,k)
            for rank in range(3):
                spatial_b[c,rank,i] = np.dot(beta[c,rank],bas)
                spatial_d[c,rank,i] = np.dot(logdiff[c,rank],bas)
    out = np.ones((nt,nx),np.float32)
    tgeom = 1. if dynamic else 0.
    aa,bb,ad,bd = _geometry_at(geometry,ncomp,full,tgeom,min_safe,min_gap)
    for ix in range(nx):
        x = ix/(nx-1)
        for c in range(ncomp):
            if aa[c] < x < bb[c]:
                out[0,ix] = 0.
    t_initial = 1. if dynamic else 0.
    tb,_ = _time_basis(t_initial)
    initial_speed=math.exp(np.dot(clock,tb))
    oldplus,oldminus=np.empty((ncomp,nz)),np.empty((ncomp,nz))
    for c in range(ncomp):
        oldplus[c],oldminus[c]=_component_rates(spatial_b[c],spatial_d[c],frequency[c],
            tracking[c],dynamic,t_initial,initial_speed,aa[c],bb[c],ad[c],bd[c],nz)
    for it in range(1,nt):
        for step in range(substeps):
            istep=(it-1)*substeps+step+1
            tau = istep*dt
            t = 1-tau if dynamic else tau
            aa,bb,ad,bd = _geometry_at(geometry,ncomp,full,t if dynamic else 0.,min_safe,min_gap)
            tb,_ = _time_basis(t)
            speed = math.exp(np.dot(clock,tb))
            for c in range(ncomp):
                plus,minus=_component_rates(spatial_b[c],spatial_d[c],frequency[c],
                    tracking[c],dynamic,t,speed,aa[c],bb[c],ad[c],bd[c],nz)
                if not use_cn:
                    new=_linear_step(survival[c],plus,minus,oldplus[c],oldminus[c],dt,1.)
                else:
                    fallback=istep==1
                    if not fallback:
                        new=_linear_step(survival[c],plus,minus,oldplus[c],oldminus[c],dt,.5)
                        for iz in range(1,nz-1):
                            if new[iz]<-2e-13 or new[iz]>1+2e-13:
                                fallback=True
                            if not dynamic and new[iz]>survival[c,iz]+2e-13:
                                fallback=True
                    if fallback:
                        pm,mm=.5*(plus+oldplus[c]),.5*(minus+oldminus[c])
                        mid=_linear_step(survival[c],pm,mm,pm,mm,dt/2,1.)
                        new=_linear_step(mid,plus,minus,plus,minus,dt/2,1.)
                survival[c]=new
                oldplus[c],oldminus[c]=plus,minus
        # Exact physical boundaries, convex fitted probability interpolation.
        for ix in range(nx):
            x = ix/(nx-1)
            for c in range(ncomp):
                if aa[c] < x < bb[c]:
                    pos = (x-aa[c])/(bb[c]-aa[c])*(nz-1)
                    lo = min(int(pos),nz-2)
                    frac = pos-lo
                    # Exponential-fitted, convex interpolation resolves thin
                    # absorbing boundary layers better than linear interpolation.
                    ts = math.sin(math.pi*frequency[c]*t) if dynamic else 0.
                    tc = math.cos(math.pi*frequency[c]*t) if dynamic else 0.
                    bmid = speed*(.5*(spatial_b[c,0,lo]+spatial_b[c,0,lo+1])
                           +ts*.5*(spatial_b[c,1,lo]+spatial_b[c,1,lo+1])
                           +tc*.5*(spatial_b[c,2,lo]+spatial_b[c,2,lo+1]))
                    if dynamic:
                        bmid += (tracking[c]-1)*(ad[c]+(bd[c]-ad[c])*(lo+.5)*h)/(bb[c]-aa[c])
                    dmid = speed*math.exp(.5*(spatial_d[c,0,lo]+spatial_d[c,0,lo+1])
                           +ts*.5*(spatial_d[c,1,lo]+spatial_d[c,1,lo+1])
                           +tc*.5*(spatial_d[c,2,lo]+spatial_d[c,2,lo+1]))
                    peclet = bmid*h/dmid
                    if abs(peclet) < 1e-5:
                        weight = frac + .5*peclet*frac*(1-frac)
                    elif peclet > 0:
                        weight = math.expm1(-peclet*frac)/math.expm1(-peclet)
                    else:
                        weight = math.exp(peclet*(1-frac))*math.expm1(peclet*frac)/math.expm1(peclet)
                    s = (1-weight)*survival[c,lo]+weight*survival[c,lo+1]
                    out[it,ix] = 1-s
    return out


@njit(cache=True, parallel=True)
def _solve_batch(ncomp,full,geometry,beta,logdiff,frequency,clock,tracking,
                 dynamic,nx,nt,nz,substeps,min_safe,min_gap,use_cn):
    n = len(ncomp)
    out = np.empty((n,nt,nx),np.float32)
    for i in prange(n):
        out[i] = _solve_one(ncomp[i],full[i],geometry[i],beta[i],logdiff[i],
                            frequency[i],clock[i],tracking[i],dynamic,nx,nt,nz,
                            substeps,min_safe,min_gap,use_cn)
    return out


def solve_parameters(p, cfg: Config):
    cfg.validate()
    return _solve_batch(p["ncomp"],p["full_domain"],p["geometry"],p["beta"],
        p["logdiff"],p["frequency"],p["clock"],p["tracking"],
        cfg.mode == "fixed_deadline_exit",cfg.nx,cfg.nt,cfg.nz,cfg.substeps,
        cfg.min_safe_width,cfg.min_gap_width,cfg.integrator=="rannacher_cn")


def generate(ids, cfg: Config):
    p = sample_parameters(ids,cfg)
    u = solve_parameters(p,cfg)
    check_probabilities(u,cfg)
    return u,p


def generate_refined(ids,cfg,tolerance=.005,max_refinements=4):
    """Refine EVERY sample until successive-grid max difference is small.

    Keeps the finer result. This is a numerical convergence check, not a
    rigorous error bound or a rare-event relative-accuracy guarantee. Unresolved
    samples cause an exception, rather than rejection/resampling of the prior.
    """
    if tolerance <= 0 or max_refinements < 1:
        raise ValueError("Invalid refinement settings")
    p=sample_parameters(ids,cfg)
    n=len(p["sample_id"])
    result=np.empty((n,cfg.nt,cfg.nx),np.float32)
    p["solver_nz"]=np.zeros(n,np.int32)
    p["solver_substeps"]=np.zeros(n,np.int32)
    p["refinement_difference"]=np.full(n,np.nan,np.float64)
    pending=np.arange(n)
    previous=solve_parameters(p,cfg)
    for level in range(1,max_refinements+1):
        fine=replace(cfg,nz=(cfg.nz-1)*2**level+1,substeps=cfg.substeps*2**level)
        subset={key:value[pending] for key,value in p.items()}
        current=solve_parameters(subset,fine)
        delta=np.abs(current-previous).max(axis=(1,2))
        good=delta<=tolerance
        accepted=pending[good]
        result[accepted]=current[good]
        p["solver_nz"][accepted]=fine.nz
        p["solver_substeps"][accepted]=fine.substeps
        p["refinement_difference"][accepted]=delta[good]
        if good.all():
            check_probabilities(result,cfg)
            return result,p
        pending=pending[~good]
        previous=current[~good]
    raise RuntimeError(f"{len(pending)} samples did not pass refinement: ids "
                       f"{p['sample_id'][pending][:20].tolist()}. Increase max_refinements "
                       "or the base resolution. No sample has been replaced.")


def check_probabilities(u,cfg,tolerance=2e-6):
    if not np.isfinite(u).all():
        raise RuntimeError("Nonfinite PDE output")
    if float(u.min()) < -tolerance or float(u.max()) > 1+tolerance:
        raise RuntimeError("Probability bound violation; results were NOT clipped")
    violation = max(0.,float(-np.min(np.diff(u,axis=1))))
    if cfg.mode == "fixed_start_exit" and violation > tolerance:
        raise RuntimeError(f"Horizon monotonicity violated by {violation}")
    return {"minimum":float(u.min()),"maximum":float(u.max()),
            "largest_decrease_along_tau":violation,
            "monotonicity_required":cfg.mode == "fixed_start_exit"}


def evaluate_settings(p,cfg,index=0,times=None,x=None):
    """Dense human/ML-readable labels at CHRONOLOGICAL times.

    b and D are NaN outside the safe set: dynamics there are irrelevant for
    the killed process. Always use safe_mask when consuming these labels.
    beta/logdiff/geometry coefficients preserve their analytic definitions.
    """
    if times is None:
        times = np.linspace(0,1,cfg.nt)
    if x is None:
        x = np.linspace(*cfg.domain,cfg.nx)
    times,x = np.asarray(times,float),np.asarray(x,float)
    lo,hi = cfg.domain
    xn = (x-lo)/(hi-lo)
    j = int(p["ncomp"][index])
    bounds = np.empty((len(times),j,2))
    b = np.full((len(times),len(x)),np.nan)
    d = np.full_like(b,np.nan)
    safe = np.zeros_like(b,dtype=bool)
    for it,t in enumerate(times):
        a,zend,ad,bd = _geometry_at(p["geometry"][index],j,p["full_domain"][index],
             t if cfg.mode=="fixed_deadline_exit" else 0.,cfg.min_safe_width,cfg.min_gap_width)
        bounds[it,:,0],bounds[it,:,1] = lo+(hi-lo)*a,lo+(hi-lo)*zend
        tb,_ = _time_basis(t)
        speed = np.exp(p["clock"][index]@tb)
        for c in range(j):
            mask = (xn>a[c])&(xn<zend[c])
            zz = (xn[mask]-a[c])/(zend[c]-a[c])
            f = np.array([_basis(float(v),cfg.spatial_modes) for v in zz])
            if not len(f):
                continue
            dynamic = cfg.mode == "fixed_deadline_exit"
            angle = np.pi*p["frequency"][index,c]*t
            tr = np.array([1,np.sin(angle) if dynamic else 0.,np.cos(angle) if dynamic else 0.])
            beta = f @ (tr@p["beta"][index,c])
            diff = np.exp(f @ (tr@p["logdiff"][index,c]))
            width = zend[c]-a[c]
            vel = ad[c]+(bd[c]-ad[c])*zz if dynamic else np.zeros_like(zz)
            b[it,mask] = (hi-lo)*(p["tracking"][index,c]*vel+width*speed*beta)
            d[it,mask] = (hi-lo)**2*width**2*speed*diff
            safe[it,mask] = True
    return {"x":x,"chronological_time":times,"safe_intervals":bounds,
            "safe_mask":safe,"drift":b,"diffusivity":d,"noise_std":np.sqrt(2*d)}


def convergence_report(ids,cfg,levels=3):
    """Paired refinements with the SAME continuous parameters and output grid.

    Differences are diagnostics, not rigorous a posteriori error bounds.
    """
    if levels < 2:
        raise ValueError("Need at least two resolutions")
    p = sample_parameters(ids,cfg)
    previous = solve_parameters(p,cfg)
    rows=[]
    for level in range(1,levels):
        fine = replace(cfg,nz=(cfg.nz-1)*2**level+1,substeps=cfg.substeps*2**level)
        t0=time.perf_counter()
        current=solve_parameters(p,fine)
        error=np.abs(current-previous)
        per_max=error.max(axis=(1,2))
        per_rmse=np.sqrt(np.mean(error.astype(float)**2,axis=(1,2)))
        rows.append({"level":level,"nz":fine.nz,"substeps":fine.substeps,
                     "max_abs":float(per_max.max()),"median_sample_max":float(np.median(per_max)),
                     "p95_sample_max":float(np.quantile(per_max,.95)),
                     "mean_rmse":float(per_rmse.mean()),"seconds":time.perf_counter()-t0})
        previous=current
    return rows


def diversity_summary(u,seed=123):
    """Descriptive, not a certificate of PosteriorBench-equivalent difficulty.

    Each sample keeps its physical probability scale; no per-image rescaling.
    PCA uses a centered empirical Gram matrix, at most 512 samples.
    """
    v=np.asarray(u[:512],dtype=float)
    a=v.reshape(len(v),-1)
    mid=((v>.05)&(v<.95)).mean(axis=(1,2))
    mean=a.mean(axis=1)
    centered=a-a.mean(axis=0,keepdims=True)
    gram=centered@centered.T/max(1,len(a)-1)
    eigenvalues=np.maximum(np.linalg.eigvalsh(gram)[::-1],0)
    total=eigenvalues.sum()
    if total<=1e-20:
        n95,pr=0,0.
    else:
        frac=eigenvalues/total
        n95=int(np.searchsorted(np.cumsum(frac),.95)+1)
        pr=float(1/np.sum(frac**2))
    rng=np.random.default_rng(seed)
    ia=rng.integers(0,len(a),2000)
    ib=rng.integers(0,len(a),2000)
    keep=ia!=ib
    distances=np.sqrt(np.mean((a[ia[keep]]-a[ib[keep]])**2,axis=1))
    summary={"n_samples":len(a),"mean_risk_quantiles":np.quantile(mean,[.05,.5,.95]).tolist(),
             "intermediate_fraction_quantiles":np.quantile(mid,[.05,.5,.95]).tolist(),
             "pca_components_95pct":n95,"pca_participation_ratio":pr,
             "pairwise_rmse_quantiles":np.quantile(distances,[.05,.5,.95]).tolist() if len(distances) else [],
             "samples_mean_above_0.98":float(np.mean(mean>.98)),
             "samples_mean_below_0.02":float(np.mean(mean<.02))}
    return summary,{"sample_mean":mean,"intermediate_fraction":mid,"eigenvalues":eigenvalues}


def save_shard(path,u,p,cfg):
    """Write a closed HDF5 shard under a temp name, then atomically rename.

    Use a local filesystem for atomicity. Network/cloud mounts have different
    guarantees; on Colab, copy completed local shards to durable storage.
    """
    import h5py
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+".partial")
    with h5py.File(tmp,"w") as f:
        f.attrs["version"]=VERSION
        f.attrs["config_json"]=json.dumps(dataclasses.asdict(cfg),sort_keys=True)
        f.attrs["time_semantics"]=cfg.time_semantics
        f.attrs["event"]="first exit from the (possibly time-dependent) open safe set"
        f.attrs["diffusion_convention"]="dX = b dt + sqrt(2 D) dW"
        f.attrs["integrator"]=cfg.integrator+"; exponential-fitted positive spatial rates and reconstruction"
        f.create_dataset("risk",data=u,chunks=(1,cfg.nt,cfg.nx),compression="lzf",shuffle=True)
        f.create_dataset("x",data=np.linspace(*cfg.domain,cfg.nx))
        f.create_dataset("tau",data=np.linspace(0,1,cfg.nt))
        if cfg.mode=="fixed_deadline_exit":
            f.create_dataset("start_time",data=1-np.linspace(0,1,cfg.nt))
        labels=f.create_group("parameters")
        for key,value in p.items():
            labels.create_dataset(key,data=value,compression="lzf",shuffle=True)
        f.attrs["complete"]=True
        f.flush()
    os.replace(tmp,path)


def estimate_storage(cfg,n=1_000_000):
    p=sample_parameters([0],cfg)
    label_bytes=sum(a.nbytes for a in p.values())
    return {"uncompressed_risk_GB":n*cfg.nt*cfg.nx*4/1e9,
            "uncompressed_labels_GB":n*label_bytes/1e9,
            "note":"Decimal GB; metadata overhead excluded; compression is data-dependent."}


def throughput(cfg,n=64):
    # The warm-up is excluded; generation includes parameter sampling and checks.
    generate([9_999_999],cfg)
    t0=time.perf_counter()
    generate(np.arange(n)+8_000_000,cfg)
    seconds=time.perf_counter()-t0
    rate=n/seconds
    return {"samples":n,"seconds":seconds,"samples_per_second":rate,
            "million_compute_hours_excluding_IO":1_000_000/rate/3600}


def write_dataset(root,cfg,total=1_000_000,shard_size=2048,batch_size=64,
                  first_shard=0,stop_shard=None,max_shards=None,
                  sentinel_count=2,sentinel_tolerance=.01,
                  adaptive_tolerance=None,max_refinements=4,
                  backup_root=None,keep_local=True):
    """Resumable deterministic shards; no one-million-sample allocation.

    Run disjoint shard ranges on separate machines. Never write the same shard
    concurrently. Existing complete shards are checked, not silently replaced.
    Sentinel refinement failure STOPS the run. No samples are discarded to
    make the prior prettier. This is a sampled check, not an all-sample proof.
    """
    import h5py
    import shutil
    cfg.validate()
    if total<1 or shard_size<1 or batch_size<1:
        raise ValueError("total, shard_size, batch_size must be positive")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest={"version":VERSION,"config":dataclasses.asdict(cfg),"total":total,
              "shard_size":shard_size,"dtype":"float32","time_semantics":cfg.time_semantics,
              "adaptive_tolerance":adaptive_tolerance,
              "family_names":FAMILY_NAMES,"event":"first_exit",
              "parameterization":"See bundled risk1d.py and notebook mathematical definitions.",
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # JSON canonicalization also converts tuples to lists for restart comparison.
    manifest=json.loads(json.dumps(manifest,sort_keys=True))
    mp=root/"manifest.json"
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:
            raise ValueError("Manifest mismatch. Use another directory for a changed prior, solver, or sample count.")
    else:
        mp.write_text(json.dumps(manifest,indent=2))
    backup=Path(backup_root) if backup_root is not None else None
    if backup is not None:
        backup.mkdir(parents=True,exist_ok=True)
        bmp=backup/"manifest.json"
        if bmp.exists() and json.loads(bmp.read_text())!=manifest:
            raise ValueError("Backup manifest mismatch")
        if not bmp.exists():
            shutil.copy2(mp,bmp)
    # Preserve the exact code used to define the prior and generator.
    for directory in ([root] if backup is None else [root,backup]):
        dest=directory/"risk1d_source.py"
        if not dest.exists():
            shutil.copy2(Path(__file__),dest)
    n_shards=(total+shard_size-1)//shard_size
    stop=n_shards if stop_shard is None else min(stop_shard,n_shards)
    if max_shards is not None:
        stop=min(stop,first_shard+max_shards)
    for shard in range(first_shard,stop):
        start,end=shard*shard_size,min((shard+1)*shard_size,total)
        path=root/f"risk_{shard:06d}.h5"
        existing=path if path.exists() else (backup/path.name if backup is not None and (backup/path.name).exists() else None)
        if existing is not None:
            with h5py.File(existing,"r") as f:
                good=bool(f.attrs.get("complete",False)) and f["risk"].shape==(end-start,cfg.nt,cfg.nx)
                good=good and int(f["parameters/sample_id"][0])==start and int(f["parameters/sample_id"][-1])==end-1
                good=good and f.attrs.get("config_json","")==json.dumps(dataclasses.asdict(cfg),sort_keys=True)
            if not good:
                raise RuntimeError(f"Incomplete/incompatible shard: {path}")
            if backup is not None and existing==path and not (backup/path.name).exists():
                remote_tmp=backup/(path.name+".partial")
                shutil.copy2(path,remote_tmp)
                os.replace(remote_tmp,backup/path.name)
                if not keep_local:
                    path.unlink()
            print(f"skip {path.name}: complete",flush=True)
            continue
        tick=time.perf_counter()
        # Memory is bounded by one shard, not total dataset size.
        us,ps=[],[]
        for a in range(start,end,batch_size):
            ids=np.arange(a,min(a+batch_size,end))
            if adaptive_tolerance is None:
                u,p=generate(ids,cfg)
            else:
                u,p=generate_refined(ids,cfg,adaptive_tolerance,max_refinements)
            us.append(u);ps.append(p)
        u=np.concatenate(us,axis=0)
        p={key:np.concatenate([v[key] for v in ps],axis=0) for key in ps[0]}
        del us,ps
        diag=check_probabilities(u,cfg)
        if adaptive_tolerance is not None:
            diag["all_sample_refinement_max"]=float(p["refinement_difference"].max())
            diag["solver_nz_range"]=[int(p["solver_nz"].min()),int(p["solver_nz"].max())]
        if sentinel_count and adaptive_tolerance is None:
            choose=np.unique(np.linspace(0,end-start-1,min(sentinel_count,end-start),dtype=int))
            small={key:value[choose] for key,value in p.items()}
            fine=replace(cfg,nz=2*(cfg.nz-1)+1,substeps=2*cfg.substeps)
            refined=solve_parameters(small,fine)
            error=np.abs(refined-u[choose])
            diag["sentinel_sample_ids"]=p["sample_id"][choose].tolist()
            diag["sentinel_max_abs_refinement_difference"]=float(error.max())
            if error.max()>sentinel_tolerance:
                raise RuntimeError(f"Shard {shard}: sentinel refinement difference {error.max():.4g} > {sentinel_tolerance}. "
                    "Increase resolution and use a new directory; this sample has NOT been rejected/replaced.")
        save_shard(path,u,p,cfg)
        diag["elapsed_seconds"]=time.perf_counter()-tick
        path.with_suffix(".diagnostics.json").write_text(json.dumps(diag,indent=2))
        print(f"{path.name}: {end-start} samples, {diag['elapsed_seconds']:.1f}s, {path.stat().st_size/1e6:.1f} MB",flush=True)
        if backup is not None:
            remote_tmp=backup/(path.name+".partial")
            shutil.copy2(path,remote_tmp)
            if remote_tmp.stat().st_size!=path.stat().st_size:
                raise IOError("Backup size mismatch; local shard kept")
            os.replace(remote_tmp,backup/path.name)
            shutil.copy2(path.with_suffix(".diagnostics.json"),backup/path.with_suffix(".diagnostics.json").name)
            if not keep_local:
                path.unlink()
    return root


def read_shard(path):
    import h5py
    with h5py.File(path,"r") as f:
        cfg=Config(**json.loads(f.attrs["config_json"]))
        u=f["risk"][:]
        p={name:value[:] for name,value in f["parameters"].items()}
    return u,p,cfg


def run_validation_tests():
    """Analytical checks plus reproducibility/storage tests (small, CPU-only)."""
    import tempfile
    from scipy.integrate import quad
    report={}

    def brownian_risk(z,operational_time,d=.05):
        z=np.asarray(z)
        risk=np.ones_like(z,dtype=float)
        mask=(z>0)&(z<1)
        if operational_time == 0:
            risk[mask]=0.
        else:
            modes=np.arange(1,602,2,dtype=float)
            co=4/(np.pi*modes)*np.exp(-d*(np.pi*modes)**2*operational_time)
            risk[mask]=1-co@np.sin(np.pi*modes[:,None]*z[mask][None,:])
        return risk

    cfg=Config(nx=65,nt=33,nz=257,substeps=16)
    p=sample_parameters([123],cfg)
    p["ncomp"][:]=1;p["full_domain"][:]=1
    p["beta"][:]=0;p["logdiff"][:]=0;p["logdiff"][0,0,0,0]=np.log(.05)
    p["clock"][:]=0;p["tracking"][:]=1
    u=solve_parameters(p,cfg)[0]
    x=np.linspace(0,1,cfg.nx);tau=np.linspace(0,1,cfg.nt)
    exact=np.array([brownian_risk(x,t) for t in tau])
    error=float(np.max(np.abs(u-exact)))
    assert error<.004,error
    report["brownian_static_max_abs_error"]=error

    # Tests chronological time reversal, independently using quadrature.
    dynamic=replace(cfg,mode="fixed_deadline_exit")
    p["clock"][0]=np.array([0.,.6,-.4,.15,.2])
    speed=lambda t: float(np.exp(p["clock"][0]@_time_basis(t)[0]))
    for c in (cfg,dynamic):
        u=solve_parameters(p,c)[0]
        operational=[quad(speed,1-t,1,epsabs=1e-11)[0] if c.mode=="fixed_deadline_exit"
                     else quad(speed,0,t,epsabs=1e-11)[0] for t in tau]
        exact=np.array([brownian_risk(x,ot) for ot in operational])
        error=float(np.max(np.abs(u-exact)))
        assert error<.005,error
        report[c.mode+"_clock_max_abs_error"]=error

    # Moving boundaries + tracking drift reduce analytically to killed Brownian
    # motion on z in (0,1). This checks the coordinate-velocity correction.
    p=sample_parameters([3],dynamic)
    p["ncomp"][:]=1;p["full_domain"][:]=0
    p["geometry"][:]=0
    p["geometry"][0,:3,0]=np.log([.12,.70,.18])
    p["geometry"][0,:3,1]=[.6,-.3,.1]
    p["geometry"][0,:3,2]=[-.2,.4,-.5]
    p["clock"][:]=0;p["beta"][:]=0;p["logdiff"][:]=0
    p["logdiff"][0,0,0,0]=np.log(.05);p["tracking"][:]=1
    u=solve_parameters(p,dynamic)[0]
    exact=[]
    for t in tau:
        a,b,_,_=_geometry_at(p["geometry"][0],1,False,1-t,dynamic.min_safe_width,dynamic.min_gap_width)
        exact.append(brownian_risk((x-a[0])/(b[0]-a[0]),t))
    error=float(np.max(np.abs(u-np.array(exact))))
    assert error<.005,error
    report["moving_safe_interval_max_abs_error"]=error

    # Moving-boundary derivatives are compared with a central difference.
    g=sample_parameters([0],dynamic)
    t=.37;eps=1e-6;j=int(g["ncomp"][0])
    a,b,ad,bd=_geometry_at(g["geometry"][0],j,False,t,dynamic.min_safe_width,dynamic.min_gap_width)
    am,bm,_,_=_geometry_at(g["geometry"][0],j,False,t-eps,dynamic.min_safe_width,dynamic.min_gap_width)
    ap,bp,_,_=_geometry_at(g["geometry"][0],j,False,t+eps,dynamic.min_safe_width,dynamic.min_gap_width)
    error=max(float(np.max(np.abs((ap-am)/(2*eps)-ad))),float(np.max(np.abs((bp-bm)/(2*eps)-bd))))
    assert error<1e-7,error
    report["geometry_derivative_max_abs_error"]=error

    # Sampling is independent of batch boundaries and solver resolution.
    p1=sample_parameters([7,8,9],cfg)
    p2=[sample_parameters([i],replace(cfg,nz=513,substeps=32,nx=128,nt=128)) for i in [7,8,9]]
    assert all(np.array_equal(p1[k],np.concatenate([p[k] for p in p2])) for k in p1)
    report["batch_and_resolution_independent_parameters"]=True
    labels=evaluate_settings(p1,cfg,times=[0,.5,1])
    assert np.all(labels["diffusivity"][labels["safe_mask"]]>0)
    report["dense_label_evaluation"]=True
    u=solve_parameters(p1,cfg)
    report["probability_checks"]=check_probabilities(u,cfg)
    # Re-labeling the physical domain must scale b by L and D by L^2, not u.
    other=replace(cfg,domain=(-1.,1.))
    assert np.array_equal(u,solve_parameters(p1,other))
    labs2=evaluate_settings(p1,other,times=[0,.5,1])
    assert np.allclose(labs2["drift"],2*labels["drift"],equal_nan=True)
    assert np.allclose(labs2["diffusivity"],4*labels["diffusivity"],equal_nan=True)
    report["physical_domain_scaling"]=True
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"test.h5"
        save_shard(path,u,p1,cfg)
        loaded,params,c=read_shard(path)
        assert np.array_equal(u,loaded)
        assert all(np.array_equal(p1[k],params[k]) for k in p1)
    report["hdf5_roundtrip"]=True
    return report
