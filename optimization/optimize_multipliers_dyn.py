import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import foftools as fof
import iterativecombination as ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from group_purity import get_metrics_by_group, get_metrics_by_halo

def dynmass(galra,galdec,galcz,galgrpid,Aval=9.9,h=0.7):
    Rproj=fof.get_grprproj_e17(galra, galdec, galcz, galgrpid, h)
    sigma_squared = np.zeros_like(Rproj)
    for ii,gg in enumerate(np.unique(galgrpid)):
        sel = np.where(galgrpid==gg)
        nn=len(sel[0])
        if nn>2:
            ii=np.arange(1,nn+1)
            weight=ii*(nn-ii)
            tmp = np.sqrt(np.pi)/(nn*nn-nn) * np.sum(np.diff(np.sort(galcz[sel]))*weight[:-1])
            sigma_squared[sel]=tmp*tmp
        else:
           sigma_squared[sel]=-99.
    GG = 4.3e-9 # Mpc km^2 s^-2 M_sun^-1
    mdyn = Aval*sigma_squared*Rproj/GG
    sel=np.where(galgrpid==881.)
    mdyn[np.where(mdyn<=0)]=1.
    return np.log10(mdyn)


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

def fast_grid_search(multsets,objective_fn):
    assert callable(objective_fn)
    pool = Pool(30)
    results = pool.map(objective_fn, multsets)
    return results

def get_score(radeg,dedeg,cz,absrmag,divide,truegroupID,trueloghalomass,ecovolume,rproj_fit_multiplier,\
              vproj_fit_multiplier, gd_rproj_fit_multiplier, gd_vproj_fit_multiplier):
    gfparams = dict({'volume':ecovolume,'rproj_fit_multiplier':rproj_fit_multiplier,'vproj_fit_multiplier':vproj_fit_multiplier,\
           'gd_rproj_fit_multiplier':rproj_fit_multiplier, 'gd_vproj_fit_multiplier':gd_vproj_fit_multiplier,\
           'gd_fit_bins':np.arange(-24,-19,0.25),\
           'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':100.,\
           'iterative_giant_only_groups':True, 'ic_decision_mode':'centers'})
    grpid = g3gf(radeg,dedeg,cz,absrmag,divide,**gfparams)[0]
    haloid,halomasses,_,_ = ic.HAMwrapper(radeg, dedeg, cz, absrmag, grpid, ecovolume)
    g3logmh=np.zeros_like(grpid)
    for ii,hh in enumerate(haloid):
        sel = np.where(grpid==hh)
        g3logmh[sel]=halomasses[ii]
    weights = 1./fof.multiplicity_function(grpid,return_by_galaxy=True)#np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    P_G, C_G = get_metrics_by_group(grpid, truegroupID, absrmag) 
    P_H, C_H = get_metrics_by_halo(grpid, truegroupID, absrmag) 
    dynEi = np.abs(loghalom - dynmass(radeg,dedeg,cz,grpid,Aval=9.9,h=1.))
    sel = (g3logmh>13)
    muHMEdyn = weighted_percentile(dynEi[sel],weights[sel],0.5)
    score = 1 - (np.mean(P_G[sel])*np.mean(C_G[sel])*np.mean(P_H[sel])*np.mean(C_H[sel]) - 2*muHMEdyn)
    print(muHMEdyn, np.mean(P_G[sel]),np.mean(C_G[sel]),np.mean(P_H[sel]),np.mean(C_H[sel]))
    return score

if __name__=='__main__':
    mock = pd.read_hdf("../halobiasmocks/fiducial/ECO_cat_0_Planck_memb_cat.hdf5")
    mock = mock[mock.M_r<-17.33]
    radeg = mock.ra.to_numpy()
    dedeg = mock.dec.to_numpy()
    cz = mock.cz.to_numpy()
    absrmag = mock.M_r.to_numpy()
    haloid = mock.haloid.to_numpy()
    loghalom = mock.loghalom.to_numpy()
    def objective(mult):
        return get_score(radeg,dedeg,cz,absrmag,-19.5,haloid,loghalom,192351.,mult[0],mult[1],mult[2],mult[3])
   
    if False: 
        # do grid serch
        ncandidates=10
        #rproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
        #vproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
        #gd_rproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
        #gd_vproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
        #candid = np.array([rproj_fit__candid,vproj_fit__candid,gd_rproj_fit__candid,gd_vproj_fit__candid]).T
        rproj_fit__candid = [1,2,3]#[2.5]#[1,3,5,7]
        vproj_fit__candid =  [4,5,6,7]#[7.]#[1,3,5,7]
        gd_rproj_fit__candid =  [1,2,3]#[1,2,3]#[1,3,5,7]
        gd_vproj_fit__candid = [1,2,3]#[2.5,4,4.5]#[1,3,5,7]
        candid=[]
        for R1 in rproj_fit__candid:
            for V1 in vproj_fit__candid:
                for R2 in gd_rproj_fit__candid:
                    for V2 in gd_vproj_fit__candid:
                        candid.append((R1,V1,R2,V2))
        candid=np.array(candid)

        print(candid)
        import time
        ti = time.time()
        scores = fast_grid_search(candid,objective)
        print("best score from grid search", np.min(scores))
        bestgridmult = candid[np.argmin(scores)]
        print('best mult from grid search', bestgridmult) 
        print('done in {} seconds'.format(time.time()-ti))
    print(objective((1,7,1,1)))
    print(objective((2.5,7,3,2.5))) 
