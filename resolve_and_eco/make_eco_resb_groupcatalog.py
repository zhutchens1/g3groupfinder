import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import iterativecombination as ic
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

if __name__=='__main__':
    hubble_const = 100.
    ecovolume = 191958.08 / (hubble_const/100.)**3.
    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':3.5,'vproj_fit_multiplier':6.5,\
           'gd_rproj_fit_multiplier':1., 'gd_vproj_fit_multiplier':2.5, 'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4], 'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const,\
            'iterative_giant_only_groups':True, 'ic_decision_mode':'centers'})
    ########################
    # Group Finding: ECO
    ########################
    eco = pd.read_csv('ECOdata_080822.csv')
    ecogroupsel = (eco.absrmag<-17.33)
    ecog3grpid = np.zeros(len(eco))-99.
    output = g3gf(eco[ecogroupsel].radeg,eco[ecogroupsel].dedeg,eco[ecogroupsel].cz,eco[ecogroupsel].absrmag,-19.4,**gfargseco)
    
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
    resolve = pd.read_csv('RESOLVEdata_080822.csv')
    resolveg3grpid=np.zeros(len(resolve))-99.
    resolveg3logmh=np.zeros(len(resolve))-99.
    # RESOLVE-B analogue group finding + HAM
    resbana = pd.read_csv("ECOdata_080822.csv")
    resbana=resbana[(resbana.absrmag<-17.0)]
    resbanag3grpid, _, resbana_fofsep, resbana_rproj_fit_params, _, resbana_vproj_fit_params, _, resbana_gd_rproj_fit_params, _, resbana_gd_vproj_fit_params, _ = g3gf(resbana.radeg,resbana.dedeg, resbana.cz, resbana.absrmag,-19.4, **gfargseco)
    print(resbana_fofsep)
    resbana.loc[:,'g3grp_l'] = resbanag3grpid
    resbanag3logmh = np.zeros_like(resbanag3grpid)-99.
    resbanahamsel = (resbanag3grpid>0)
    haloid, halomass, junk, junk = ic.HAMwrapper(resbana.radeg[resbanahamsel].to_numpy(), resbana.dedeg[resbanahamsel].to_numpy(), resbana.cz[resbanahamsel].to_numpy(),\
         resbana.absrmag[resbanahamsel].to_numpy(), resbana.g3grp_l[resbanahamsel].to_numpy(), ecovolume*(hubble_const/100.)**3., inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(resbanag3grpid[resbanahamsel], return_index=True)
    halomass = halomass-np.log10(hubble_const/100.)
    for i,idv in enumerate(haloid):
        sel = np.where(resbanag3grpid==idv)
        resbanag3logmh[sel] = halomass[i] # m337b
    resbana.loc[:,'g3logmh_l'] = resbanag3logmh
    resbanaintmag = ic.get_int_mag(resbana.absrmag.to_numpy(),resbana.g3grp_l.to_numpy())
    mhspline = interp1d(resbanaintmag,resbanag3logmh)

    # RESOLVE-B group finding
    resbgroupsel = (resolve.fl_insample==1.)&(resolve.f_b==1)
    tmpid = g3gf(resolve[resbgroupsel].radeg, resolve[resbgroupsel].dedeg, resolve[resbgroupsel].cz, resolve[resbgroupsel].absrmag, -19.4, fof_sep=resbana_fofsep,\
         rproj_fit_params=resbana_rproj_fit_params,vproj_fit_params=resbana_vproj_fit_params, gd_rproj_fit_params = resbana_gd_rproj_fit_params,\
         gd_vproj_fit_params = resbana_gd_vproj_fit_params, rproj_fit_multiplier=3.5,vproj_fit_multiplier = 6.5, gd_rproj_fit_multiplier=1., gd_vproj_fit_multiplier=2.5, H0=hubble_const,\
         iterative_giant_only_groups=True)[0]

    resolveg3grpid[resbgroupsel] = tmpid
    resolveg3logmh[resbgroupsel] = mhspline(ic.get_int_mag(resolve[resbgroupsel].absrmag.to_numpy(), tmpid))

    # get RESOLVE-A from ECO
    resolvename = resolve.name.to_numpy()
    resname_in_eco = eco.resname.to_numpy()

    for ii,rn in enumerate(resolvename):
        if rn.startswith('rs'):
            ecosel = (resname_in_eco==rn)
            resolveg3grpid[ii]=eco.g3grp_l[ecosel]
            resolveg3logmh[ii]=eco.g3logmh_l[ecosel]

    resolve.loc[:,'g3grp_l']=resolveg3grpid
    resolve.loc[:,'g3logmh_l']=resolveg3logmh
    resolve.to_csv("RESOLVEdata_G3catalog_luminosity.csv")
