import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
from splitfalsepairs import split_false_pairs 
import foftools as fof
import iterativecombination as ic
from group_purity import get_metrics_by_group, get_metrics_by_halo
import os

def make_eco_group_cat(param):
    eco = pd.read_csv("ECOdata_080822.csv")
    eco = eco[eco.absrmag < -17.33]
    radeg = eco.radeg.to_numpy()
    dedeg = eco.dedeg.to_numpy()
    cz = eco.cz.to_numpy()
    absrmag = eco.absrmag.to_numpy()
    ecovolume = 192351.
    gfparams = dict({'volume':ecovolume,'rproj_fit_multiplier':param[0],'vproj_fit_multiplier':param[1],\
       'vproj_fit_offset':param[2],'gd_rproj_fit_multiplier':1.5, 'gd_vproj_fit_multiplier':3.5,\
       'gd_fit_bins':np.arange(-24,-19,0.25),\
       'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':70.,\
       'iterative_giant_only_groups':True, 'ic_decision_mode':'centers','center_mode':'average'})
    grpid = g3gf(radeg,dedeg,cz,absrmag,-19.5,**gfparams)[0]
    haloid,halomasses,_,_ = ic.HAMwrapper(radeg, dedeg, cz, absrmag, grpid, ecovolume)
    g3logmh=np.zeros_like(grpid)
    halomasses=halomasses-np.log10(gfparams['H0']/100.)
    for ii,hh in enumerate(haloid):
        sel = np.where(grpid==hh)
        g3logmh[sel]=halomasses[ii]
    eco.loc[:,'g3grp_l']=grpid
    eco.loc[:,'g3logmh_l']=g3logmh
    eco.loc[:,'g3grpmhi_l']=np.log10(10**ic.get_int_mass(eco.logmgas,eco.g3grp_l)/1.4)

    outfile = 'ad_prefac_eco_catls/'+'ECO_adgroupcatl__'+str(param[0])+'__'+str(param[1])+'__'+str(param[2])+'__.csv'
    eco.to_csv(outfile,index=False)
    return None

if __name__=='__main__':
    write_catls=False
    if write_catls:
        params = [(1,1,100), (2.5,3.5,200), (3,4,300), (5,7,300), (10,10,400)]
        from multiprocessing import Pool
        import time
        ti = time.time()
        pool = Pool(10)
        _=pool.map(make_eco_group_cat, params)
        print('done in ', time.time()-ti, ' seconds.')
        pool.close()
    else:
        make_plot('ad_prefac_eco_catls/')
