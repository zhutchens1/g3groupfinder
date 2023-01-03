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

def get_score(radeg,dedeg,cz,absrmag,divide,truegroupID,trueloghalomass,halo_ngal,ecovolume,rproj_fit_multiplier,\
              vproj_fit_multiplier,vproj_fit_offset,gd_rproj_fit_multiplier, gd_vproj_fit_multiplier, gd_vproj_fit_offset):
    gfparams = dict({'volume':ecovolume,'rproj_fit_multiplier':rproj_fit_multiplier,'vproj_fit_multiplier':vproj_fit_multiplier,\
           'vproj_fit_offset':vproj_fit_offset,'gd_rproj_fit_multiplier':gd_rproj_fit_multiplier, 'gd_vproj_fit_multiplier':gd_vproj_fit_multiplier,\
           'gd_vproj_fit_offset':gd_vproj_fit_offset,'gd_fit_bins':np.arange(-24,-19,0.25),\
           'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':100.,\
           'iterative_giant_only_groups':True, 'ic_decision_mode':'centers'})
    grpid = g3gf(radeg,dedeg,cz,absrmag,divide,**gfparams)[0]
    haloid,halomasses,_,_ = ic.HAMwrapper(radeg, dedeg, cz, absrmag, grpid, ecovolume)
    g3logmh=np.zeros_like(grpid)
    for ii,hh in enumerate(haloid):
        sel = np.where(grpid==hh)
        g3logmh[sel]=halomasses[ii]
    g3logmhdyn = dynmass(radeg,dedeg,cz,grpid,Aval=9.9,h=1)
    grpn = fof.multiplicity_function(grpid,return_by_galaxy=True) 
    weights = 1./grpn#np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
     
    my_halo_ngal = fof.multiplicity_function(truegroupID,True)
    computesel = (my_halo_ngal==halo_ngal)
    muHME = weighted_percentile(np.abs(np.array(g3logmh[computesel] - loghalom[computesel])), weights[computesel], 0.5)
    P_G, C_G = get_metrics_by_group(grpid[computesel], truegroupID[computesel], absrmag[computesel]) 
    P_H, C_H = get_metrics_by_halo(grpid[computesel], truegroupID[computesel], absrmag[computesel])
    dynsel = computesel & (weights<1/7.)
    muHMEdyn = weighted_percentile(np.abs(np.array(g3logmhdyn[dynsel] - loghalom[dynsel])), weights[dynsel], 0.5)
    muHME_HAMhighN = weighted_percentile(np.abs(np.array(g3logmh[dynsel] - loghalom[dynsel])), weights[dynsel], 0.5)

    # output statistics
    bygalaxydf = pd.DataFrame(np.array([grpid[computesel], P_G, C_G, P_H, C_H, g3logmh[computesel], grpn[computesel], absrmag[computesel], trueloghalomass[computesel], truegroupID[computesel]]).T,columns=['grpid','P_G','C_G','P_H','C_H','g3logmh','grpn', 'absrmag', 'loghalom', 'haloid'])
    bygroupdf = bygalaxydf.groupby('grpid').first()
    meanPg = np.mean(bygroupdf.P_G.to_numpy())
    meanCg = np.mean(bygroupdf.C_G.to_numpy())
    byhalodf = bygalaxydf.groupby('haloid').first()
    meanPh = np.mean(byhalodf.P_H.to_numpy())
    meanCh = np.mean(byhalodf.C_H.to_numpy())
    dwarfgroups_by_gal = bygalaxydf.groupby('grpid').filter(lambda grp: (grp.absrmag>-19.5).all())
    dwarfgroups_by_grp = dwarfgroups_by_gal.groupby('grpid').first()
    meanPgdw = np.mean(dwarfgroups_by_grp.P_G.to_numpy())
    meanCgdw = np.mean(dwarfgroups_by_grp.C_G.to_numpy())
    dwarfhalos_by_halo = dwarfgroups_by_gal.groupby('haloid').first()
    meanPhdw = np.mean(dwarfhalos_by_halo.P_H.to_numpy())
    meanChdw = np.mean(dwarfhalos_by_halo.C_H.to_numpy())
    muHMEdw = weighted_percentile(np.abs(np.array(dwarfgroups_by_gal.g3logmh.to_numpy()-dwarfgroups_by_gal.loghalom.to_numpy())), 1/dwarfgroups_by_gal.grpn.to_numpy(), 0.5)
    ngt1_dwgr_bygal = dwarfgroups_by_gal[dwarfgroups_by_gal.grpn>1]
    ngt1_dwgr_bygrp = ngt1_dwgr_bygal.groupby('grpid').first()
    ngt1_dwgr_byhalo = ngt1_dwgr_bygal.groupby('haloid').first()
    print("# N>1 dwarf-only groups: ", len(ngt1_dwgr_bygrp))
    meanPgdwgt1 = np.mean(ngt1_dwgr_bygrp.P_G.to_numpy())
    meanCgdwgt1 = np.mean(ngt1_dwgr_bygrp.C_G.to_numpy())
    meanPhdwgt1 = np.mean(ngt1_dwgr_byhalo.P_H.to_numpy())
    meanChdwgt1 = np.mean(ngt1_dwgr_byhalo.C_H.to_numpy())
    muHMEdwgt1 = weighted_percentile(np.abs(np.array(ngt1_dwgr_bygal.g3logmh.to_numpy()-ngt1_dwgr_bygal.loghalom.to_numpy())), 1/ngt1_dwgr_bygal.grpn.to_numpy(), 0.5)
    Ndwarfgroups=len(dwarfgroups_by_grp)
    Ngt1dwarfgroups=len(ngt1_dwgr_bygrp)
    return (muHME, muHMEdyn, muHME_HAMhighN, meanPg, meanCg, meanPh, meanCh, muHMEdw, meanPgdw, meanCgdw, meanPhdw, meanChdw, meanPgdwgt1, meanCgdwgt1, meanPhdwgt1, meanChdwgt1, muHMEdwgt1,Ndwarfgroups,Ngt1dwarfgroups) 
    #return (muHME,muHMEdyn, np.mean(P_G),np.mean(C_G),np.mean(P_H),np.mean(C_H))

if __name__=='__main__':
    mock = pd.read_hdf("../halobiasmocks/fiducial/ECO_cat_0_Planck_memb_cat.hdf5")
    mock = mock[mock.M_r<-17.33]
    radeg = mock.ra.to_numpy()
    dedeg = mock.dec.to_numpy()
    cz = mock.cz.to_numpy()
    absrmag = mock.M_r.to_numpy()
    haloid = mock.haloid.to_numpy()
    loghalom = mock.loghalom.to_numpy()
    halo_ngal = mock.halo_ngal.to_numpy()
    def objective(mult):
        return get_score(radeg,dedeg,cz,absrmag,-19.5,haloid,loghalom,halo_ngal,192351.,mult[0],mult[1],mult[2],mult[3],mult[4],mult[5])
   
    if True: 
        # do grid serch
        rproj_fit__candid = [3]#[1,2,3,4,5,6]#np.random.uniform(0.5,10,3)#[1,2.5,4]# [2.5,3,3.5]#[2,3,4]#[1,3,5,7]
        vproj_fit__candid = [4]#[1,2,3,4,5,6]#np.random.uniform(0.5,10,3)# [2.5,3,3.5]#[2,3,4]#[1,3,5,7]
        vproj_off__candid = [200]#[100,200,300]#[150,200,250]#[100,200,300,400]
        gd_rproj_fit__candid = [10]#np.random.uniform(0.5,10,3)#[1.5,2,2.5]#[2,3,4]#[1,3,5,7]
        gd_vproj_fit__candid = [10]#np.random.uniform(0.5,10,3)#[3.5,4,4.5]#[4,5,6]#[1,3,5,7]
        gd_vproj_off__candid = [0]
        candid=[]
        for R1 in rproj_fit__candid:
            for V1 in vproj_fit__candid:
                for Voffset in vproj_off__candid:
                    for R2 in gd_rproj_fit__candid:
                        for V2 in gd_vproj_fit__candid:
                            for gdVoffset in gd_vproj_off__candid:
                                candid.append((R1,V1,Voffset,R2,V2,gdVoffset))
        candid=np.array(candid)

        print(candid)
        import time
        ti = time.time()
        scores = fast_grid_search(candid,objective)
        print(scores)
        #print("best score from grid search", np.min(scores))
        #bestgridmult = candid[np.argmin(scores)]
        #print('best mult from grid search', bestgridmult) 
        print('done in {} seconds'.format(time.time()-ti))

        outdf = pd.DataFrame(candid,columns=['rproj_fit_mult','vproj_fit_mult','vproj_fit_offset','gd_rproj_fit_mult', 'gd_vproj_fit_mult', 'gd_vproj_fit_offset'])
        scoredf = pd.DataFrame(scores,columns=['mu_HME','mu_HME_dyn','mu_HME_HAMngt7','P_G','C_G','P_H','C_H','mu_HME_dw','P_G_dw','C_G_dw','P_H_dw','C_H_dw',\
            'P_G_dwgt1','C_G_dwgt1','P_H_dwgt1','C_H_dwgt1','mu_HME_dwgt1','n_dwgroups','n_Ngt1dwgroups'])
        outdf = outdf.join(scoredf,how='outer')
        outdf.to_csv("table_group_params_dw.csv",index=False)
        print(outdf)

    #print(objective((1,3,200,1,3,0)))
    #print(objective((1,3,200,3,3,0)))
    # note: when giant group finding params. are fixed, changing the dwarf params tend to have little effect: vast, vast majority
    # of dwarf-only groups are N=1 and thus if most of them are identified as N=1, mean is not affected
