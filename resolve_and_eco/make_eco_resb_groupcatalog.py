import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

if __name__=='__main__':
    ########################
    # Group Finding: ECO
    ########################
    eco = pd.read_csv('ECOdata_080822.csv')
    ecogroupsel = (eco.absrmag<-17.33)
    ecog3grpid = np.zeros(len(eco))-99.
    _tmp = g3gf(eco[ecogroupsel].radeg,eco[ecogroupsel].dedeg,eco[ecogroupsel].cz,eco[ecogroupsel].absrmag,-19.4,volume=192351.,rproj_fit_multiplier=3.5,vproj_fit_multiplier=6.5,\
            gd_rproj_fit_multiplier=1., gd_vproj_fit_multiplier=2.5, gd_fit_bins=np.arange(-24,-19,0.25), gd_rproj_fit_guess=[1e-5, 0.4], gd_vproj_fit_guess=[3e-5,4e-1])[0]
    ecog3grpid[ecogroupsel] = _tmp
    eco.loc[:,'g3grp_l'] = ecog3grpid
    eco.to_csv("ECOdata_G3catalog_luminosity.csv",index=False)

    ##########################
    # Group Finding: RESOLVE
    ##########################
    resolve = pd.read_csv('RESOLVEdata_080822.csv')
    resolve = resolve.set_index('name')

    # get RESOLVE-A from ECO
    resa_in_eco = eco[eco.resname!='notinresolve'].sort_values(by='resname').set_index('resname')['g3grp_l']
    resolve = pd.concat([resolve,resa_in_eco])
    print(resolve.g3grp_l)
