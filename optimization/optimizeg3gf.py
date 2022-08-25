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
    pool = Pool(20)
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
    weights = fof.multiplicity_function(grpid,return_by_galaxy=True)#np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    muHME = weighted_percentile(np.abs(np.array(g3logmh - loghalom)), weights, 0.5)
    P_G, C_G = get_metrics_by_group(grpid, truegroupID, absrmag) 
    P_H, C_H = get_metrics_by_halo(grpid, truegroupID, absrmag) 
    score = 1 - (np.mean(P_G)*np.mean(C_G)*np.mean(P_H)*np.mean(C_H) - muHME)
    print(muHME, (np.mean(P_G)*np.mean(C_G)*np.mean(P_H)*np.mean(C_H)))
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
    
    # do grid serch
    ncandidates=10
    #rproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
    #vproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
    #gd_rproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
    #gd_vproj_fit__candid = np.random.uniform(0,10,ncandidates)#np.array([2,5,8])
    #candid = np.array([rproj_fit__candid,vproj_fit__candid,gd_rproj_fit__candid,gd_vproj_fit__candid]).T
    rproj_fit__candid = [2, 2.5, 3, 3.5, 4.]#[1,3,5,7]
    vproj_fit__candid =  [6, 6.5, 7, 7.5, 8., 9.]#[1,3,5,7]
    gd_rproj_fit__candid =  [0.5, 1, 1.5, 2]#[1,3,5,7]
    gd_vproj_fit__candid = [4, 4.5, 5, 5.5, 6]#[1,3,5,7]
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
  
