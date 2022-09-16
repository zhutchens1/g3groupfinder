import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import foftools as fof
import iterativecombination as ic
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from lss_dens import lss_dens_by_galaxy
import virtools as vz

if __name__=='__main__':
    hubble_const = 70.
    ecovolume = 191958.08 / (hubble_const/100.)**3.
    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':2.5,'vproj_fit_multiplier':3.5,'vproj_fit_offset':200,\
           'gd_rproj_fit_multiplier':1.5, 'gd_vproj_fit_multiplier':3.5, 'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'iterative_giant_only_groups':True, 'ic_decision_mode':'centers'})
    ########################
    # Group Finding: ECO
    ########################
    eco = pd.read_csv('ECOdata_080822.csv')
    ecogroupsel = (eco.absrmag<-17.33)
    eco.loc[:,'logmbary'] = np.log10(10**eco.logmstar+10**eco.logmgas)
    ecog3grpid = np.zeros(len(eco))-99.
    output = g3gf(eco[ecogroupsel].radeg,eco[ecogroupsel].dedeg,eco[ecogroupsel].cz,eco[ecogroupsel].absrmag,-19.5,**gfargseco)
    
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
    eco.loc[:,'g3grplogG_l'] = ic.get_int_mass(eco.logmgas.to_numpy(),eco.g3grp_l.to_numpy())
    eco.loc[:,'g3grplogS_l'] = ic.get_int_mass(eco.logmstar.to_numpy(),eco.g3grp_l.to_numpy())
    eco.loc[:,'g3grplogB_l'] = ic.get_int_mass(eco.logmbary.to_numpy(),eco.g3grp_l.to_numpy())
    eco.loc[:,'g3grpmhi_l'] = np.log10(10**eco.g3grplogG_l / 1.4)
    ecogrpradeg,ecogrpdedeg,ecogrpcz = fof.group_skycoords(eco.radeg.to_numpy(),eco.dedeg.to_numpy(),eco.cz.to_numpy(),eco.g3grp_l.to_numpy())
    eco.loc[:,'g3grpradeg_l'] = ecogrpradeg
    eco.loc[:,'g3grpdedeg_l'] = ecogrpdedeg
    eco.loc[:,'g3grpcz_l'] = ecogrpcz
    eco.loc[:,'g3fc_l'] = fof.get_central_flag(eco.absrmag.to_numpy(), eco.g3grp_l.to_numpy())
    eco.loc[:,'g3satmhi_l'] = fof.get_satint_mass(np.log10(10**eco.logmgas.to_numpy()/1.4), eco.g3grp_l.to_numpy(), eco.g3fc_l.to_numpy())
    eco.loc[:,'g3cenmhi_l'] = fof.get_central_mass(np.log10(10**eco.logmgas.to_numpy()/1.4), eco.g3grp_l.to_numpy(), eco.g3fc_l.to_numpy())
    eco.loc[:,'g3logmhdyn_l']=fof.dynmass(eco.radeg.to_numpy(),eco.dedeg.to_numpy(),eco.cz.to_numpy(),eco.g3grp_l.to_numpy(),9.9,hubble_const/100.)
    ecog3grpngi=np.zeros_like(ecogrpradeg)
    ecog3grpndw=np.zeros_like(ecogrpradeg)
    for uid in np.unique(ecog3grpid):
        grpsel = np.where(ecog3grpid==uid)
        gisel = np.where(np.logical_and((ecog3grpid==uid),(eco.absrmag.to_numpy()<=-19.5)))
        dwsel = np.where(np.logical_and((ecog3grpid==uid),(eco.absrmag.to_numpy()>-19.5)))
        if len(gisel[0])>0:
            ecog3grpngi[grpsel]=len(gisel[0])
        if len(dwsel[0])>0:
            ecog3grpndw[grpsel]=len(dwsel[0])
    eco.loc[:,'g3grpngi_l']=ecog3grpngi
    eco.loc[:,'g3grpndw_l']=ecog3grpndw
    densoutput = lss_dens_by_galaxy(eco.g3grp_l.to_numpy(),eco.radeg.to_numpy(), eco.dedeg.to_numpy(), eco.cz.to_numpy(), eco.g3logmh_l.to_numpy(),\
         Nnn=3, rarange=(130.05,237.45), decrange=(-1,50), czrange=(2530,7470))
    eco.loc[:,'g3grpnndens_l']=densoutput[0]
    eco.loc[:,'g3grpedgeflag_l']=densoutput[1]
    eco.loc[:,'g3grpnndens2d_l']=densoutput[2]
    eco.loc[:,'g3grpedgeflag2d_l']=densoutput[3]
    eco.loc[:,'g3grpedgescale2d_l']=densoutput[4]
    eco.loc[:,'g3grptcross_l']=vz.group_crossing_time(eco.radeg.to_numpy(), eco.dedeg.to_numpy(), eco.cz.to_numpy(),eco.g3grp_l.to_numpy(),H0=hubble_const)

    nansel = (eco.g3grp_l<0)
    eco.loc[nansel,'g3grplogG_l']=-99.
    eco.loc[nansel,'g3grplogS_l']=-99.
    eco.loc[nansel,'g3grplogB_l']=-99.
    eco.loc[nansel,'g3grpmhi_l']=-99.
    eco.loc[nansel,'g3satmhi_l']=-99.
    eco.loc[nansel,'g3cenmhi_l']=-99.
    eco.loc[nansel,'g3logmh_l']=-99.
    eco.loc[nansel,'g3grpradeg_l']=-99.
    eco.loc[nansel,'g3grpdedeg_l']=-99.
    eco.loc[nansel,'g3grpcz_l']=-99.
    eco.loc[nansel,'g3fc_l']=-99.
    eco.loc[nansel,'g3grpngi_l']=-99.
    eco.loc[nansel,'g3grpndw_l']=-99.
    eco.loc[nansel,'g3grpnndens_l']=-99.
    eco.loc[nansel,'g3grpedgeflag_l']=-99.
    eco.loc[nansel,'g3grpnndens2d_l']=-99.
    eco.loc[nansel,'g3grpedgeflag2d_l']=-99.
    eco.loc[nansel,'g3grpedgescale2d_l']=-99.
    eco.loc[nansel,'g3grptcross_l']=-99.
    eco.loc[nansel,'g3logmhdyn_l']=-99.
    eco.to_csv("ECOdata_G3catalog_luminosity.csv",index=False)
    ##########################
    # Group Finding: RESOLVE
    ##########################
    resolve = pd.read_csv('RESOLVEdata_080822.csv')
    resolve.loc[:,'logmbary']=np.log10(10**resolve.logmstar+10**resolve.logmgas)
    resolveg3grpid=np.zeros(len(resolve))-99.
    resolveg3logmh=np.zeros(len(resolve))-99.
    # RESOLVE-B analogue group finding + HAM
    resbana = pd.read_csv("ECOdata_080822.csv")
    resbana=resbana[(resbana.absrmag<-17.0)]
    resbanag3grpid, _, resbana_fofsep, resbana_rproj_fit_params, _, resbana_vproj_fit_params, _, resbana_gd_rproj_fit_params, _, resbana_gd_vproj_fit_params, _ = g3gf(resbana.radeg,resbana.dedeg, resbana.cz, resbana.absrmag,-19.5, **gfargseco)
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
    resbana.loc[:,'g3grpabsrmag_l'] = resbanaintmag
    resbana.to_csv("resbanalogue.csv",index=False)

    # RESOLVE-B group finding
    resbgroupsel = (resolve.fl_insample==1.)&(resolve.f_b==1)
    tmpid = g3gf(resolve[resbgroupsel].radeg, resolve[resbgroupsel].dedeg, resolve[resbgroupsel].cz, resolve[resbgroupsel].absrmag, -19.5, fof_sep=resbana_fofsep,\
         rproj_fit_params=resbana_rproj_fit_params,vproj_fit_params=resbana_vproj_fit_params, gd_rproj_fit_params = resbana_gd_rproj_fit_params,\
         gd_vproj_fit_params = resbana_gd_vproj_fit_params, rproj_fit_multiplier=2.5,vproj_fit_multiplier = 3.5, vproj_fit_offset=200, gd_rproj_fit_multiplier=1.5, gd_vproj_fit_multiplier=3.5,\
         H0=hubble_const, iterative_giant_only_groups=True)[0]

    resolveg3grpid[resbgroupsel] = tmpid
    resolveg3logmh[resbgroupsel] = mhspline(ic.get_int_mag(resolve[resbgroupsel].absrmag.to_numpy(), tmpid))
    resolveg3grpradeg=np.zeros_like(resolveg3grpid)
    resolveg3grpdedeg=np.zeros_like(resolveg3grpid)
    resolveg3grpcz=np.zeros_like(resolveg3grpid)
    resolveg3grplogG_l = np.zeros_like(resolveg3grpid)
    resolveg3grplogS_l = np.zeros_like(resolveg3grpid)
    resolveg3grplogB_l = np.zeros_like(resolveg3grpid)
    resolveg3fc = np.zeros_like(resolveg3grpid)
    resolveg3grplogG_l[resbgroupsel] = ic.get_int_mass(resolve[resbgroupsel].logmgas.to_numpy(), tmpid)
    resolveg3grplogS_l[resbgroupsel] = ic.get_int_mass(resolve[resbgroupsel].logmstar.to_numpy(), tmpid)
    resolveg3grplogB_l[resbgroupsel] = ic.get_int_mass(resolve[resbgroupsel].logmbary.to_numpy(), tmpid)
    resolveg3grpradeg[resbgroupsel],resolveg3grpdedeg[resbgroupsel],resolveg3grpcz[resbgroupsel]=fof.group_skycoords(resolve[resbgroupsel].radeg.to_numpy(),\
        resolve[resbgroupsel].dedeg.to_numpy(), resolve[resbgroupsel].cz.to_numpy(), tmpid)
    resolveg3fc[resbgroupsel] = fof.get_central_flag(resolve[resbgroupsel].absrmag.to_numpy(), tmpid)

    resolveg3satmhi=np.zeros_like(resolveg3grpid)
    resolveg3cenmhi=np.zeros_like(resolveg3grpid)
    resolveg3cenmhi[resbgroupsel] = fof.get_central_mass(np.log10(10**resolve[resbgroupsel].logmgas.to_numpy()/1.4), tmpid, resolveg3fc[resbgroupsel])
    resolveg3satmhi[resbgroupsel] = fof.get_satint_mass(np.log10(10**resolve[resbgroupsel].logmgas.to_numpy()/1.4), tmpid, resolveg3fc[resbgroupsel])

    resolveg3grpngi = np.zeros_like(resolveg3grpid)
    resolveg3grpndw = np.zeros_like(resolveg3grpid)
    resbg3grpngi = np.zeros_like(tmpid)
    resbg3grpndw = np.zeros_like(tmpid)
    for uid in np.unique(tmpid):
        grpsel = np.where(tmpid==uid)
        gisel = np.where(np.logical_and((tmpid==uid),(resolve[resbgroupsel].absrmag.to_numpy()<=-19.5)))
        dwsel = np.where(np.logical_and((tmpid==uid),(resolve[resbgroupsel].absrmag.to_numpy()>-19.5)))
        if len(gisel[0])>0.:
            resbg3grpngi[grpsel]=len(gisel[0])
        if len(dwsel[0])>0.:
            resbg3grpndw[grpsel]=len(dwsel[0])
    resolveg3grpngi[resbgroupsel]=resbg3grpngi
    resolveg3grpndw[resbgroupsel]=resbg3grpndw

    resolveg3nndens=np.zeros_like(resolveg3grpid)
    resolveg3edgeflag=np.zeros_like(resolveg3grpid)
    resolveg3nndens2d=np.zeros_like(resolveg3grpid)
    resolveg3edgeflag2d=np.zeros_like(resolveg3grpid)
    resolveg3edgescale2d=np.zeros_like(resolveg3grpid)
    RESB_RADEG_REMAPPED = np.copy(resolve[resbgroupsel].radeg.to_numpy())
    REMAPSEL = np.where(RESB_RADEG_REMAPPED>18*15.)
    RESB_RADEG_REMAPPED[REMAPSEL] = RESB_RADEG_REMAPPED[REMAPSEL]-360.
    densoutput = lss_dens_by_galaxy(tmpid, RESB_RADEG_REMAPPED, resolve[resbgroupsel].dedeg.to_numpy(), resolve[resbgroupsel].cz.to_numpy(), resolveg3logmh[resbgroupsel],\
        Nnn=3, rarange=(-2*15.,3*15.), decrange=(-1.25,1.25), czrange=(4250,7250))
    resolveg3nndens[resbgroupsel]=densoutput[0] 
    resolveg3edgeflag[resbgroupsel]=densoutput[1] 
    resolveg3nndens2d[resbgroupsel]=densoutput[2] 
    resolveg3edgeflag2d[resbgroupsel]=densoutput[3] 
    resolveg3edgescale2d[resbgroupsel]=densoutput[4] 
    resolveg3grptcross=np.zeros_like(resolveg3grpid)
    resolveg3grptcross[resbgroupsel]=vz.group_crossing_time(resolve[resbgroupsel].radeg.to_numpy(), resolve[resbgroupsel].dedeg.to_numpy(), resolve[resbgroupsel].cz.to_numpy(),\
        tmpid, H0=hubble_const)    

    resolveg3dynmass=np.zeros_like(resolveg3grpid)-99.
    resbdynmass = fof.dynmass(resolve[resbgroupsel].radeg.to_numpy(), resolve[resbgroupsel].dedeg.to_numpy(), resolve[resbgroupsel].cz.to_numpy(),\
        tmpid,9.9,hubble_const/100.)
    resolveg3dynmass[resbgroupsel]=resbdynmass

    # get RESOLVE-A from ECO
    resolvename = resolve.name.to_numpy()
    resname_in_eco = eco.resname.to_numpy()

    for ii,rn in enumerate(resolvename):
        if rn.startswith('rs'):
            ecosel = (resname_in_eco==rn)
            resolveg3grpid[ii]=eco.g3grp_l[ecosel]
            resolveg3logmh[ii]=eco.g3logmh_l[ecosel]
            resolveg3grpradeg[ii]=eco.g3grpradeg_l[ecosel]
            resolveg3grpdedeg[ii]=eco.g3grpdedeg_l[ecosel]
            resolveg3grpcz[ii]=eco.g3grpcz_l[ecosel]
            resolveg3grplogG_l[ii]=eco.g3grplogG_l[ecosel]
            resolveg3grplogS_l[ii]=eco.g3grplogS_l[ecosel]
            resolveg3grplogB_l[ii]=eco.g3grplogB_l[ecosel]
            resolveg3fc[ii]=eco.g3fc_l[ecosel]
            resolveg3grpngi[ii]=eco.g3grpngi_l[ecosel]
            resolveg3grpndw[ii]=eco.g3grpndw_l[ecosel]
            resolveg3nndens[ii]=eco.g3grpnndens_l[ecosel]
            resolveg3edgeflag[ii]=eco.g3grpedgeflag_l[ecosel]
            resolveg3nndens2d[ii]=eco.g3grpnndens2d_l[ecosel]
            resolveg3edgeflag2d[ii]=eco.g3grpedgeflag2d_l[ecosel]
            resolveg3edgescale2d[ii]=eco.g3grpedgescale2d_l[ecosel]
            resolveg3grptcross[ii]=eco.g3grptcross_l[ecosel]
            resolveg3cenmhi[ii]=eco.g3cenmhi_l[ecosel]
            resolveg3satmhi[ii]=eco.g3satmhi_l[ecosel]
            resolveg3dynmass[ii]=eco.g3logmhdyn_l[ecosel]
    resolve.loc[:,'g3grp_l']=resolveg3grpid
    resolve.loc[:,'g3logmh_l']=resolveg3logmh
    resolve.loc[:,'g3grplogG_l']=resolveg3grplogG_l
    resolve.loc[:,'g3grplogS_l']=resolveg3grplogS_l
    resolve.loc[:,'g3grplogB_l']=resolveg3grplogB_l
    resolve.loc[:,'g3grpmhi_l']=np.log10(10**resolveg3grplogG_l/1.4)
    resolve.loc[(resolve.g3grp_l==-99.),'g3grpmhi_l']=-99.
    resolve.loc[:,'g3cenmhi']=resolveg3cenmhi
    resolve.loc[:,'g3satmhi']=resolveg3satmhi
    resolve.loc[:,'g3grpradeg_l']=resolveg3grpradeg
    resolve.loc[:,'g3grpdedeg_l']=resolveg3grpdedeg
    resolve.loc[:,'g3grpcz_l']=resolveg3grpcz
    resolve.loc[:,'g3fc_l']=resolveg3fc
    resolve.loc[:,'g3grpngi_l']=resolveg3grpngi
    resolve.loc[:,'g3grpndw_l']=resolveg3grpndw
    resolve.loc[:,'g3grpnndens_l']=resolveg3nndens
    resolve.loc[:,'g3grpedgeflag_l']=resolveg3edgeflag
    resolve.loc[:,'g3grpnndens2d_l']=resolveg3nndens2d
    resolve.loc[:,'g3grpedgeflag2d_l']=resolveg3edgeflag2d
    resolve.loc[:,'g3grpedgescale2d_l']=resolveg3edgescale2d
    resolve.loc[:,'g3grptcross_l']=resolveg3grptcross
    resolve.loc[:,'g3logmhdyn_l']=resolveg3dynmass
    resolve.to_csv("RESOLVEdata_G3catalog_luminosity.csv")
