import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import foftools as fof
import iterativecombination as ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from group_purity import get_metrics_by_group, get_metrics_by_halo
from optimize_multipliers import weighted_percentile

def get_score(radeg,dedeg,cz,absrmag,divide,truegroupID,trueloghalomass,ecovolume,centermethod):
    gfparams = dict({'volume':ecovolume,'rproj_fit_multiplier':3,'vproj_fit_multiplier':7,\
           'gd_rproj_fit_multiplier':1, 'gd_vproj_fit_multiplier':4,\
           'gd_fit_bins':np.arange(-24,-19,0.25),\
           'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':100.,\
           'iterative_giant_only_groups':True, 'ic_decision_mode':'centers','center_mode':centermethod})
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
    print(muHME, np.mean(P_G),np.mean(C_G),np.mean(P_H),np.mean(C_H))
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
    def objective(method):
        return get_score(radeg,dedeg,cz,absrmag,-19.5,haloid,loghalom,192351.,method)

    score=objective('average')
    print("GF score for average group center: ", score)
    score=objective('BCG')
    print("GF score for BCG center: ", score)
    score=objective((0.1,8))
    print("GF score for logistic center: ", score)

