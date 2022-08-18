import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import iterativecombination as ic
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

if __name__=='__main__':
    hubble_const = 70.
    ecovolume = 191958.08 / (hubble_const/100.)**3.
    ########################
    # Group Finding: ECO
    ########################
    eco = pd.read_csv('ECOdata_080822.csv')
    ecogroupsel = (eco.absrmag<-17.33)
    ecog3grpid = np.zeros(len(eco))-99.
    output = g3gf(eco[ecogroupsel].radeg,eco[ecogroupsel].dedeg,eco[ecogroupsel].cz,eco[ecogroupsel].absrmag,-19.4,volume=ecovolume,rproj_fit_multiplier=3.5,vproj_fit_multiplier=6.5,\
            gd_rproj_fit_multiplier=1., gd_vproj_fit_multiplier=2.5, gd_fit_bins=np.arange(-24,-19,0.25), gd_rproj_fit_guess=[1e-5, 0.4], gd_vproj_fit_guess=[3e-5,4e-1], H0=hubble_const)
    
    tmpid = output[0]
    ecog3grpid[ecogroupsel] = tmpid
    eco.loc[:,'g3grp_l'] = ecog3grpid
    
    ecog3logmh = np.zeros_like(ecog3grpid)-99.
    ecohamsel = (ecog3grpid!=-99.)
    haloid, halomass, junk, junk = ic.HAMwrapper(eco.radeg[ecohamsel].to_numpy(), eco.dedeg[ecohamsel].to_numpy(), eco.cz[ecohamsel].to_numpy(),\
         eco.absrmag[ecohamsel].to_numpy(), eco.g3grp_l[ecohamsel].to_numpy(), ecovolume*(hubble_const/100.)**3., inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grpid[ecohamsel], return_index=True)
    halomass = halomass-np.log10(hubble_const/100.)
    for i,idv in enumerate(haloid):
        sel = np.where(ecog3grpid==idv)
        ecog3logmh[sel] = halomass[i] # m337b
    eco.loc[:,'g3logmh_l'] = ecog3logmh

    eco.to_csv("ECOdata_G3catalog_luminosity.csv",index=False)

    ##########################
    # Group Finding: RESOLVE
    ##########################
    #resolve = pd.read_csv('RESOLVEdata_080822.csv')
    #resolve = resolve.set_index('name')

    # get RESOLVE-A from ECO
    #resa_in_eco = eco[eco.resname!='notinresolve'].sort_values(by='resname').set_index('resname')['g3grp_l']
    #resolve = pd.merge([resolve,resa_in_eco])
    #print(resolve.g3grp_l)
