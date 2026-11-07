"""Lossless screening for different prefix masks per observation case.
Based on the attached exact_screen.py. Numerical time stepping is unchanged;
only each reference's own mask is used when accumulating squared discrepancies.
"""
import math
import numpy as np
from numba import njit, prange
from risk1d import _basis, _geometry_at, _time_basis, _component_rates, _linear_step
from exact_screen import geometry_bound

@njit(cache=True)
def _prefix_multi(ncomp, full, geometry, beta, logdiff, frequency, clock,
               tracking, dynamic, nx, nt, nz, substeps, min_safe, min_gap, use_cn, obs_mask, masks, targets, initial_bounds, threshold_sum):
    """Solve survival on exact component-local moving coordinates.

    Positive-rate birth/death approximation with Rannacher/CN or Euler.
    CN steps that violate bounds or required monotonicity use two implicit
    Euler half-steps instead. No post-hoc probability clipping.
    """
    # Exact sequential rejection: replace each known outside-set contribution
    # by its computed squared discrepancy as the SAME solver emits that row.
    accum=initial_bounds.copy()
    nref=targets.shape[0]
    last_row=0
    for ri in range(nt):
        for ci in range(nx):
            if obs_mask[ri,ci]: last_row=ri
    if np.min(accum)>threshold_sum+1e-10:
        return accum, 0
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
    for it in range(1,last_row+1):
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
        # At this point out[it] exactly matches the unmodified full solver.
        for ix in range(nx):
            if obs_mask[it,ix]:
                x=ix/(nx-1);inside=False
                for ci in range(ncomp):
                    if aa[ci]<x<bb[ci]: inside=True
                for ri in range(nref):
                    err=(float(out[it,ix])-targets[ri,it,ix])**2
                    floor=0. if inside else (1.-targets[ri,it,ix])**2
                    if masks[ri,it,ix]:
                        accum[ri]+=err-floor
        if np.min(accum)>threshold_sum+1e-10:
            return accum, it
    return accum, last_row


@njit(cache=True,parallel=True)
def prefix_multi_batch(ncomp,full,geometry,beta,logdiff,frequency,clock,tracking,
                       dynamic,nx,nt,nz,substeps,min_safe,min_gap,use_cn,
                       union,masks,targets,bounds,limit):
    errors=np.empty_like(bounds);stops=np.zeros(len(ncomp),np.int32)
    for i in prange(len(ncomp)):
        errors[i],stops[i]=_prefix_multi(ncomp[i],full[i],geometry[i],beta[i],logdiff[i],frequency[i],clock[i],tracking[i],
            dynamic,nx,nt,nz,substeps,min_safe,min_gap,use_cn,union,masks,targets,bounds[i],limit)
    return errors,stops


def bounds_for_cases(p,cfg,masks,truths,ks):
    out=np.zeros((len(p['sample_id']),len(truths)))
    for k in np.unique(ks):
        group=np.flatnonzero(ks==k);mask=masks[group[0]]
        if not np.all(masks[group]==mask):
            raise ValueError('Cases with the same K must share the observation design.')
        coords=np.argwhere(mask)
        target=np.ascontiguousarray(truths[group][:,mask],dtype=np.float64)
        out[:,group]=geometry_bound(p['geometry'],p['ncomp'],p['full_domain'],
            coords[:,0],coords[:,1],target,cfg.nt,cfg.nx,True,cfg.min_safe_width,cfg.min_gap_width)
    return out


def prefix(p,cfg,masks,truths,bounds,epsilon):
    m=masks.sum(axis=(1,2))
    if not np.all(m==m[0]): raise ValueError('This version requires the same observation budget for each case.')
    return prefix_multi_batch(p['ncomp'],p['full_domain'],p['geometry'],p['beta'],p['logdiff'],p['frequency'],p['clock'],p['tracking'],
        True,cfg.nx,cfg.nt,cfg.nz,cfg.substeps,cfg.min_safe_width,cfg.min_gap_width,cfg.integrator=='rannacher_cn',
        np.any(masks,axis=0),masks,truths.astype(np.float64),bounds,float(epsilon)**2*int(m[0]))
