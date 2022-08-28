import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import foftools as fof
import iterativecombination as ic
from group_purity import get_metrics_by_group, get_metrics_by_halo
import os

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

def make_mock_group_cat(filename):
    mock = pd.read_hdf(filename)
    mock = mock[mock.M_r < -19.5]
    radeg = mock.ra.to_numpy()
    dedeg = mock.dec.to_numpy()
    cz = mock.cz.to_numpy()
    absrmag = mock.M_r.to_numpy()
    ecovolume = 192351.
    gfparams = dict({'volume':ecovolume,'rproj_fit_multiplier':2.5,'vproj_fit_multiplier':7,\
       'gd_rproj_fit_multiplier':3, 'gd_vproj_fit_multiplier':2.5,\
       'gd_fit_bins':np.arange(-24,-19,0.25),\
       'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':100.,\
       'iterative_giant_only_groups':True, 'ic_decision_mode':'centers','center_mode':'average'})
    grpid = g3gf(radeg,dedeg,cz,absrmag,-19.5,**gfparams)[0]
    haloid,halomasses,_,_ = ic.HAMwrapper(radeg, dedeg, cz, absrmag, grpid, ecovolume)
    g3logmh=np.zeros_like(grpid)
    for ii,hh in enumerate(haloid):
        sel = np.where(grpid==hh)
        g3logmh[sel]=halomasses[ii]
    grpn = fof.multiplicity_function(grpid,return_by_galaxy=True)
    muHME = weighted_percentile(np.abs(np.array(g3logmh - mock.loghalom.to_numpy())), 1./grpn, 0.5)
    P_G, C_G = get_metrics_by_group(grpid, mock.haloid.to_numpy(), absrmag) 
    P_H, C_H = get_metrics_by_halo(grpid, mock.haloid.to_numpy(), absrmag) 
    mock.loc[:,'g3grp_l']=grpid
    mock.loc[:,'g3grpn_l']=grpn
    mock.loc[:,'g3logmh_l']=g3logmh
    mock.loc[:,'mu_HME']=np.full(len(mock),muHME)
    mock.loc[:,'group_purity']=P_G
    mock.loc[:,'group_comp']=C_G
    mock.loc[:,'halo_purity']=P_H
    mock.loc[:,'halo_comp']=C_H
    x = filename.split('/')
    outfile=x[0]+'/halobiasgroupcats/'+x[2]+'/'+x[3][:-5]+".csv"
    mock.to_csv(outfile)
    return None

if __name__=='__main__':
    mockdir = '../halobiasmocks/'
    subdirs = ['fiducial/','dv0_8/','dv1_2/','central/']
    fidfiles = [mockdir+subdirs[0]+filename for filename in os.listdir(mockdir+subdirs[0])]
    dv08files = [mockdir+subdirs[1]+filename for filename in os.listdir(mockdir+subdirs[1])]
    dv12files = [mockdir+subdirs[2]+filename for filename in os.listdir(mockdir+subdirs[2])]
    cenfiles = [mockdir+subdirs[3]+filename for filename in os.listdir(mockdir+subdirs[3])]
    files_to_read = fidfiles + dv08files + dv12files + cenfiles

    from multiprocessing import Pool
    import time
    ti = time.time()
    pool = Pool(20)
    _=pool.map(make_mock_group_cat, files_to_read)
    print('done in ', time.time()-ti, ' seconds.')
