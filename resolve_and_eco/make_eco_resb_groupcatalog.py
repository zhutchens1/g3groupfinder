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
from scipy.interpolate import interp1d, UnivariateSpline
from lss_dens import lss_dens_by_galaxy
import virtools as vz
from replicate_fof_groups_katie import do_katie_HAM

if __name__=='__main__':
    hubble_const = 70.
    ecovolume = 191958.08 / (hubble_const/100.)**3.
    gfargseco = dict({'volume':ecovolume,'rproj_fit_multiplier':3,'vproj_fit_multiplier':4,'vproj_fit_offset':200,\
           'gd_rproj_fit_multiplier':2, 'gd_vproj_fit_multiplier':4, 'gd_vproj_fit_offset':100,\
           'gd_fit_bins':np.arange(-24,-19,0.25), 'gd_rproj_fit_guess':[1e-5, 0.4],\
           'gd_vproj_fit_guess':[3e-5,4e-1], 'H0':hubble_const, 'iterative_giant_only_groups':True, 'ic_decision_mode':'centers'})
    ########################
    # Group Finding: ECO
    ########################
    eco = pd.read_csv('ECOdata_051123.csv')
    eco = eco[eco.resname!='rs1492'] # removing rs1492
    ecogroupsel = (eco.absrmag<=-17.33)
    eco.loc[:,'logmbary'] = np.log10(10**eco.logmstar+10**eco.logmgas)
    ecog3grpid = np.zeros(len(eco))-999.

    # do g3 group finding with only galx. above floor to get best-fit params.
    _, _, fof_sep_eco, rproj_best_fit_eco, _, vproj_best_fit_eco, _, gd_rproj_best_fit_eco, _, gd_vproj_best_fit_eco, _ = g3gf(eco[ecogroupsel].radeg, \
        eco[ecogroupsel].dedeg,eco[ecogroupsel].cz,eco[ecogroupsel].absrmag,-19.5,**gfargseco)
    # now re-do group finding using best-fit parameters for full ECO, so we can get group ID for all galaxies including those below floor.
    output = g3gf(eco.radeg,eco.dedeg,eco.cz,eco.absrmag,-19.5,rproj_fit_params=rproj_best_fit_eco, vproj_fit_params=vproj_best_fit_eco,gd_rproj_fit_params=gd_rproj_best_fit_eco,\
        gd_vproj_fit_params=gd_vproj_best_fit_eco, **gfargseco)

    #tmpid = output[0] # commented these out on Feb 24 23; reworking code to include sub-Lr-floor galxs.
    #ecog3grpid[ecogroupsel] = tmpid
    eco.loc[:,'g3grp_l'] = output[0]
    ecog3grpid = output[0]
    
    ecog3logmh = np.zeros_like(ecog3grpid)-999.
    ecog3logmh200 = np.zeros_like(ecog3grpid)-999.
    ecohamsel = (eco.absrmag.to_numpy()<=-17.33)#(ecog3grpid!=-999.) # this no longer works because g3grpid goes below floor.
    haloid, halomass, junk, junk = ic.HAMwrapper(eco.radeg[ecohamsel].to_numpy(), eco.dedeg[ecohamsel].to_numpy(), eco.cz[ecohamsel].to_numpy(),\
         eco.absrmag[ecohamsel].to_numpy(), eco.g3grp_l[ecohamsel].to_numpy(), ecovolume*(hubble_const/100.)**3., inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grpid[ecohamsel], return_index=True)
    halomass = halomass-np.log10(hubble_const/100.)
    for i,idv in enumerate(haloid):
        sel = np.where(ecog3grpid==idv)
        ecog3logmh[sel] = halomass[i] # m337b
    ecog3logmh[~ecohamsel]=-999.

    haloid200, halomass200, junk, junk = do_katie_HAM(eco.radeg[ecohamsel].to_numpy(), eco.dedeg[ecohamsel].to_numpy(), eco.cz[ecohamsel].to_numpy(),\
         eco.absrmag[ecohamsel].to_numpy(), eco.g3grp_l[ecohamsel].to_numpy(), ecovolume*(hubble_const/100.)**3., inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(ecog3grpid[ecohamsel], return_index=True)
    halomass200 = halomass200-np.log10(hubble_const/100.)
    for i,idv in enumerate(haloid200):
        sel = np.where(ecog3grpid==idv)
        ecog3logmh200[sel] = halomass200[i] # m337b
    ecog3logmh200[~ecohamsel]=-999.

    eco.loc[:,'g3logmh_l'] = ecog3logmh
    eco.loc[:,'g3logmh200_l'] = ecog3logmh200
    

    #
    # from here down, calculating group quantities and need to subselect galaxies that actually included in group quantities, i.e. Mr < -17.33
    eco_grpproperty_sel = (eco.absrmag<=-17.33)
    eco.loc[eco_grpproperty_sel,'g3grpabsrmag_l'] = ic.get_int_mag(eco[eco_grpproperty_sel].absrmag.to_numpy(),eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grpgiantabsrmag_l'] = ic.get_int_mag_giants(eco[eco_grpproperty_sel].absrmag.to_numpy(),eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grplogG_l'] = ic.get_int_mass(eco[eco_grpproperty_sel].logmgas.to_numpy(),eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grplogS_l'] = ic.get_int_mass(eco[eco_grpproperty_sel].logmstar.to_numpy(),eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grplogB_l'] = ic.get_int_mass(eco[eco_grpproperty_sel].logmbary.to_numpy(),eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grpmhi_l'] = np.log10(10**eco[eco_grpproperty_sel].g3grplogG_l / 1.4)
    ecogrpradeg,ecogrpdedeg,ecogrpcz = fof.group_skycoords(eco[eco_grpproperty_sel].radeg.to_numpy(),eco[eco_grpproperty_sel].dedeg.to_numpy(),eco[eco_grpproperty_sel].cz.to_numpy(),\
                                                           eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3grpradeg_l'] = ecogrpradeg
    eco.loc[eco_grpproperty_sel,'g3grpdedeg_l'] = ecogrpdedeg
    eco.loc[eco_grpproperty_sel,'g3grpcz_l'] = ecogrpcz
    eco.loc[eco_grpproperty_sel,'g3fc_l'] = fof.get_central_flag(eco[eco_grpproperty_sel].absrmag.to_numpy(), eco[eco_grpproperty_sel].g3grp_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3satmhi_l'] = fof.get_satint_mass(np.log10(10**eco[eco_grpproperty_sel].logmgas.to_numpy()/1.4), eco[eco_grpproperty_sel].g3grp_l.to_numpy(), eco[eco_grpproperty_sel].g3fc_l.to_numpy())
    eco.loc[eco_grpproperty_sel,'g3cenmhi_l'] = fof.get_central_mass(np.log10(10**eco[eco_grpproperty_sel].logmgas.to_numpy()/1.4), eco[eco_grpproperty_sel].g3grp_l.to_numpy(), eco[eco_grpproperty_sel].g3fc_l.to_numpy())
    ecog3grpngi=np.zeros_like(ecog3grpid)#(ecogrpradeg)
    ecog3grpndw=np.zeros_like(ecog3grpid)#(ecogrpradeg)
    for uid in np.unique(ecog3grpid):
        grpsel = np.where(np.logical_and(ecog3grpid==uid,eco.absrmag.to_numpy()<=-17.33))
        gisel = np.where(np.logical_and((ecog3grpid==uid),(eco.absrmag.to_numpy()<=-19.5)))
        dwsel = np.where(np.logical_and((ecog3grpid==uid),(eco.absrmag.to_numpy()>-19.5)))
        if len(gisel[0])>0:
            ecog3grpngi[grpsel]=len(gisel[0])
        if len(dwsel[0])>0:
            ecog3grpndw[grpsel]=len(dwsel[0])
    eco.loc[:,'g3grpngi_l']=ecog3grpngi
    eco.loc[:,'g3grpndw_l']=ecog3grpndw
    tmp=fof.dynmass(eco[eco_grpproperty_sel].radeg.to_numpy(),eco[eco_grpproperty_sel].dedeg.to_numpy(),eco[eco_grpproperty_sel].cz.to_numpy(),\
                    eco[eco_grpproperty_sel].g3grp_l.to_numpy(),9.9,hubble_const/100.)
    tmp[(eco[eco_grpproperty_sel].g3grpngi_l+eco[eco_grpproperty_sel].g3grpndw_l)<=7]=-999.
    eco.loc[eco_grpproperty_sel,'g3logmhdyn_l']=tmp
    densoutput = lss_dens_by_galaxy(eco[eco_grpproperty_sel].g3grp_l.to_numpy(),eco[eco_grpproperty_sel].radeg.to_numpy(), eco[eco_grpproperty_sel].dedeg.to_numpy(), eco[eco_grpproperty_sel].cz.to_numpy(),\
                eco[eco_grpproperty_sel].g3logmh_l.to_numpy(),Nnn=3, rarange=(130.05,237.45), decrange=(-1,50), czrange=(2530,7470))
    eco.loc[eco_grpproperty_sel,'g3grpnndens_l']=densoutput[0]
    eco.loc[eco_grpproperty_sel,'g3grpedgeflag_l']=densoutput[1]
    eco.loc[eco_grpproperty_sel,'g3grpnndens2d_l']=densoutput[2]
    eco.loc[eco_grpproperty_sel,'g3grpedgeflag2d_l']=densoutput[3]
    eco.loc[eco_grpproperty_sel,'g3grpedgescale2d_l']=densoutput[4]
    eco.loc[eco_grpproperty_sel,'g3grptcross_l']=vz.group_crossing_time(eco[eco_grpproperty_sel].radeg.to_numpy(), eco[eco_grpproperty_sel].dedeg.to_numpy(), eco[eco_grpproperty_sel].cz.to_numpy(),\
                eco[eco_grpproperty_sel].g3grp_l.to_numpy(),H0=hubble_const)

    ccb_remapped = np.zeros(len(eco))+1.
    central_names = eco[(eco.fc>0)&(eco.ccb>1)].name.to_numpy()
    ccb_vals = eco[(eco.fc>0)&(eco.ccb>1)].ccb.to_numpy()
    g3ccb_ids = [float(ecog3grpid[np.where(eco.name.to_numpy()==cnm)]) for cnm in central_names]
    for ii,idv in enumerate(g3ccb_ids):
        sel = (ecog3grpid==idv)
        ccb_remapped[sel]=ccb_vals[ii]
    eco.loc[:,'ccb_remapped']=ccb_remapped
    
    # make splines for halo mass extrapolation
    tmp = eco[eco.g3logmh_l>0].groupby('g3grp_l').first()
    tmp = tmp[['g3grpabsrmag_l','g3logmh_l','g3logmh200_l']]
    tmp = tmp[tmp.g3grpabsrmag_l>-19.5].sort_values(by='g3grpabsrmag_l')
    tmp = tmp.tail(100)
    params=np.polyfit(tmp.g3grpabsrmag_l,tmp.g3logmh_l,1)
    params200=np.polyfit(tmp.g3grpabsrmag_l,tmp.g3logmh200_l,1)
    mhspline_eco = lambda Mrtot: np.polyval(params,Mrtot)#UnivariateSpline(tmp.g3grpabsrmag_l,tmp.g3logmh_l,s=0.8)
    mhspline200_eco = lambda Mrtot: np.polyval(params200,Mrtot)#UnivariateSpline(tmp.g3grpabsrmag_l,tmp.g3logmh200_l,s=0.8)
    plt.figure()
    plt.plot(tmp.g3grpabsrmag_l,tmp.g3logmh_l,'o',color='palegreen',label='337')
    plt.plot(tmp.g3grpabsrmag_l,tmp.g3logmh200_l,'.',color='k',label='200',markersize=2)
    tx=np.linspace(-26,-15,1000)
    plt.plot(tx,mhspline200_eco(tx),'k-')
    plt.title("HAM curve for ECO")
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.show()
    # now, for each subfloor galaxy that was excluded from group finding need, to add group quantity
    # which is itself calculated based on above-floor galaxies only.
    for subfloorname,grpidvalue in zip(eco[~eco_grpproperty_sel].name.to_numpy(), eco[~eco_grpproperty_sel].g3grp_l.to_numpy()):
        fillsel = (eco.name==subfloorname)
        grpsel = (eco.g3grp_l==grpidvalue)&(eco.absrmag<=-17.33)
        if np.sum(grpsel)>0:
            eco.loc[fillsel,'g3grplogG_l']=np.mean(eco.loc[grpsel,'g3grplogG_l'])
            eco.loc[fillsel,'g3grplogS_l']=np.mean(eco.loc[grpsel,'g3grplogS_l'])
            eco.loc[fillsel,'g3grplogB_l']=np.mean(eco.loc[grpsel,'g3grplogB_l'])
            eco.loc[fillsel,'g3grpabsrmag_l']=np.mean(eco.loc[grpsel,'g3grpabsrmag_l'])
            eco.loc[fillsel,'g3grpgiantabsrmag_l']=np.mean(eco.loc[grpsel,'g3grpgiantabsrmag_l'])
            eco.loc[fillsel,'g3grpmhi_l']=np.mean(eco.loc[grpsel,'g3grpmhi_l'])
            eco.loc[fillsel,'g3satmhi_l']=np.mean(eco.loc[grpsel,'g3satmhi_l'])
            eco.loc[fillsel,'g3cenmhi_l']=np.mean(eco.loc[grpsel,'g3cenmhi_l'])
            eco.loc[fillsel,'g3logmh_l']=np.mean(eco.loc[grpsel,'g3logmh_l'])
            eco.loc[fillsel,'g3logmh200_l']=np.mean(eco.loc[grpsel,'g3logmh200_l'])
            eco.loc[fillsel,'g3grpradeg_l']=np.mean(eco.loc[grpsel,'g3grpradeg_l'])
            eco.loc[fillsel,'g3grpdedeg_l']=np.mean(eco.loc[grpsel,'g3grpdedeg_l'])
            eco.loc[fillsel,'g3grpcz_l']=np.mean(eco.loc[grpsel,'g3grpcz_l'])
            eco.loc[fillsel,'g3fc_l']=np.mean(eco.loc[grpsel,'g3fc_l'])
            eco.loc[fillsel,'g3grpngi_l']=np.mean(eco.loc[grpsel,'g3grpngi_l'])
            eco.loc[fillsel,'g3grpndw_l']=np.mean(eco.loc[grpsel,'g3grpndw_l'])
            eco.loc[fillsel,'g3grpnndens_l']=np.mean(eco.loc[grpsel,'g3grpnndens_l'])
            eco.loc[fillsel,'g3grpedgeflag_l']=np.mean(eco.loc[grpsel,'g3grpedgeflag_l'])
            eco.loc[fillsel,'g3grpnndens2d_l']=np.mean(eco.loc[grpsel,'g3grpnndens2d_l'])
            eco.loc[fillsel,'g3grpedgeflag2d_l']=np.mean(eco.loc[grpsel,'g3grpedgeflag2d_l'])
            eco.loc[fillsel,'g3grpedgescale2d_l']=np.mean(eco.loc[grpsel,'g3grpedgescale2d_l'])
            eco.loc[fillsel,'g3grptcross_l']=np.mean(eco.loc[grpsel,'g3grptcross_l'])
        else:
            # this means isolated dwarf galaxy below floor
            eco.loc[fillsel,'g3grplogG_l'] = eco.loc[fillsel,'logmgas']
            eco.loc[fillsel,'g3grplogS_l']= eco.loc[fillsel,'logmstar']
            eco.loc[fillsel,'g3grplogB_l']= eco.loc[fillsel,'logmbary']
            eco.loc[fillsel,'g3grpabsrmag_l'] = eco.loc[fillsel,'absrmag']
            eco.loc[fillsel,'g3grpgiantabsrmag_l'] = -999.
            eco.loc[fillsel,'g3grpmhi_l'] = np.log10(10**eco.loc[fillsel,'logmgas']/1.4)
            eco.loc[fillsel,'g3satmhi_l']= 0.
            eco.loc[fillsel,'g3cenmhi_l']= np.log10(10**eco.loc[fillsel,'logmgas']/1.4)
            eco.loc[fillsel,'g3logmh_l']= mhspline_eco(eco.loc[fillsel,'absrmag']) # extrapolate down curve
            eco.loc[fillsel,'g3logmh200_l']= mhspline200_eco(eco.loc[fillsel,'absrmag']) # extrapolate down curve
            eco.loc[fillsel,'g3grpradeg_l'] = eco.loc[fillsel,'radeg']
            eco.loc[fillsel,'g3grpdedeg_l'] = eco.loc[fillsel,'dedeg']
            eco.loc[fillsel,'g3grpcz_l'] = eco.loc[fillsel,'cz']
            eco.loc[fillsel,'g3fc_l']= 1.
            eco.loc[fillsel,'g3grpngi_l']= 0
            eco.loc[fillsel,'g3grpndw_l']= 1
            eco.loc[fillsel,'g3grpnndens_l']= -999.
            eco.loc[fillsel,'g3grpedgeflag_l']= -999.
            eco.loc[fillsel,'g3grpnndens2d_l']= -999.
            eco.loc[fillsel,'g3grpedgeflag2d_l']= -999.
    #eco.loc[~eco_grpproperty_sel,'g3grplogG_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grplogS_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grplogB_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpabsrmag_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpgiantabsrmag_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpmhi_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3satmhi_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3cenmhi_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3logmh_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3logmh200_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpradeg_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpdedeg_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpcz_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3fc_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpngi_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpndw_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpnndens_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpedgeflag_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpnndens2d_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpedgeflag2d_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grpedgescale2d_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3grptcross_l']=-999.
    #eco.loc[~eco_grpproperty_sel,'g3logmhdyn_l']=-999.
    eco.to_csv("ECOdata_G3catalog_luminosity.csv",index=False)
    ##########################
    # Group Finding: RESOLVE
    ##########################
    resolve = pd.read_csv('/srv/two/cielo/zhutchen/database/add_rs1519_resolve/RESOLVE_preupload_051123.csv')
    resolve = resolve[resolve.name!='rs1492'] # remove rs1492
    print('currently using pre-upload version of RESOLVE 5/11/2023!')
    resolve.loc[:,'logmbary']=np.log10(10**resolve.logmstar+10**resolve.logmgas)
    resolveg3grpid=np.zeros(len(resolve))-999.
    resolveg3logmh=np.zeros(len(resolve))-999.
    resolveg3logmh200=np.zeros(len(resolve))-999.
    # RESOLVE-B analogue group finding + HAM
    resbana = pd.read_csv("ECOdata_080822.csv")
    resbana=resbana[(resbana.absrmag<=-17.0)]
    resbanag3grpid, _, resbana_fofsep, resbana_rproj_fit_params, _, resbana_vproj_fit_params, _, resbana_gd_rproj_fit_params, _, resbana_gd_vproj_fit_params, _ = g3gf(resbana.radeg,resbana.dedeg, resbana.cz, resbana.absrmag,-19.5, **gfargseco)
    resbana.loc[:,'g3grp_l'] = resbanag3grpid
    resbanag3logmh = np.zeros_like(resbanag3grpid)-999.
    resbanag3logmh200 = np.zeros_like(resbanag3grpid)-999.
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
    resbanagiantintmag = ic.get_int_mag_giants(resbana.absrmag.to_numpy(),resbana.g3grp_l.to_numpy())
    _,uniqidx=np.unique(resbana.g3grp_l.to_numpy(),return_index=True)
    mhspline = interp1d(resbanaintmag[uniqidx],resbanag3logmh[uniqidx])
    tmp = pd.DataFrame(np.array([resbanaintmag[uniqidx],resbanag3logmh[uniqidx]]).T,columns=['grprmag','logmh']).sort_values(by='grprmag')
    tmp = tmp[tmp.grprmag>-19.5].tail(100)
    params_ = np.polyfit(tmp.grprmag,tmp.logmh,1)
    mhsplineEXTRAP = lambda Mrtot: np.polyval(params_, Mrtot)#UnivariateSpline(tmp.grprmag,tmp.logmh,s=0.8*len(tmp))
    resbana.loc[:,'g3grpabsrmag_l'] = resbanaintmag
    resbana.loc[:,'g3grpgiantabsrmag_l'] = resbanagiantintmag
    resbana.to_csv("resbanalogue.csv",index=False)

    haloid200, halomass200, junk, junk = do_katie_HAM(resbana.radeg[resbanahamsel].to_numpy(), resbana.dedeg[resbanahamsel].to_numpy(), resbana.cz[resbanahamsel].to_numpy(),\
         resbana.absrmag[resbanahamsel].to_numpy(), resbana.g3grp_l[resbanahamsel].to_numpy(), ecovolume*(hubble_const/100.)**3., inputfilename=None, outputfilename=None)
    junk, uniqindex = np.unique(resbanag3grpid[resbanahamsel], return_index=True)
    halomass200 = halomass200-np.log10(hubble_const/100.)
    for i,idv in enumerate(haloid200):
        sel = np.where(resbanag3grpid==idv)
        resbanag3logmh200[sel] = halomass200[i] # m337b
    resbana.loc[:,'g3logmh200_l'] = resbanag3logmh200
    resbanaintmag = ic.get_int_mag(resbana.absrmag.to_numpy(),resbana.g3grp_l.to_numpy())
    _,uniqidx=np.unique(resbana.g3grp_l.to_numpy(),return_index=True)
    mhspline200 = interp1d(resbanaintmag[uniqidx],resbanag3logmh200[uniqidx])
    tmp = pd.DataFrame(np.array([resbanaintmag[uniqidx],resbanag3logmh200[uniqidx]]).T,columns=['grprmag','logmh200']).sort_values(by='grprmag')
    tmp = tmp[tmp.grprmag>-19.5].tail(100)
    params200_ = np.polyfit(tmp.grprmag,tmp.logmh200,1)
    mhsplineEXTRAP200 = lambda Mrtot: np.polyval(params200_, Mrtot)#UnivariateSpline(tmp.grprmag,tmp.logmh200,s=0.8*len(tmp))
    resbana.to_csv("resbanalogue200.csv",index=False)
    
    plt.figure()
    tx=np.linspace(-25,-15,1000)
    plt.plot(tmp.grprmag,tmp.logmh200,'k.')
    plt.plot(tx,mhsplineEXTRAP200(tx),'k-')
    plt.title("RESOLVE curve")
    plt.show()

    # RESOLVE-B group finding
    resbgroupsel = (resolve.f_b==1) #& (resolve.absrmag<-17.) # removed -17 cut to associate gals below floor # fixed this on 2/19/23 - was using fl_insample (incorrect)
    tmpid = g3gf(resolve[resbgroupsel].radeg, resolve[resbgroupsel].dedeg, resolve[resbgroupsel].cz, resolve[resbgroupsel].absrmag, -19.5, fof_sep=resbana_fofsep,\
         rproj_fit_params=resbana_rproj_fit_params,vproj_fit_params=resbana_vproj_fit_params, gd_rproj_fit_params = resbana_gd_rproj_fit_params,\
         gd_vproj_fit_params = resbana_gd_vproj_fit_params, rproj_fit_multiplier=3,vproj_fit_multiplier = 4, vproj_fit_offset=200, gd_rproj_fit_multiplier=2, gd_vproj_fit_multiplier=4,\
         gd_vproj_fit_offset=100, H0=hubble_const, iterative_giant_only_groups=True)[0]

    resolveg3grpid[resbgroupsel] = tmpid
    # below this, need to adjust for galx gruop below floor
    resbgrppropertysel = (resolve.f_b==1)&(resolve.absrmag<=-17.)
    resolveg3logmh[resbgrppropertysel] = mhspline(ic.get_int_mag(resolve[resbgrppropertysel].absrmag.to_numpy(), resolveg3grpid[resbgrppropertysel]))
    resolveg3logmh200[resbgrppropertysel] = mhspline200(ic.get_int_mag(resolve[resbgrppropertysel].absrmag.to_numpy(), resolveg3grpid[resbgrppropertysel]))
    resolveg3grpradeg=np.zeros_like(resolveg3grpid)
    resolveg3grpdedeg=np.zeros_like(resolveg3grpid)
    resolveg3grpcz=np.zeros_like(resolveg3grpid)
    resolveg3grplogG_l = np.zeros_like(resolveg3grpid)
    resolveg3grplogS_l = np.zeros_like(resolveg3grpid)
    resolveg3grplogB_l = np.zeros_like(resolveg3grpid)
    resolveg3grpabsrmag = np.zeros_like(resolveg3grpid)
    resolveg3grpgiantabsrmag = np.zeros_like(resolveg3grpid)
    resolveg3fc = np.zeros_like(resolveg3grpid)
    resolveg3grpabsrmag[resbgrppropertysel] = ic.get_int_mag(resolve[resbgrppropertysel].absrmag.to_numpy(), resolveg3grpid[resbgrppropertysel])
    resolveg3grpgiantabsrmag[resbgrppropertysel] = ic.get_int_mag_giants(resolve[resbgrppropertysel].absrmag.to_numpy(), resolveg3grpid[resbgrppropertysel])
    resolveg3grplogG_l[resbgrppropertysel] = ic.get_int_mass(resolve[resbgrppropertysel].logmgas.to_numpy(), resolveg3grpid[resbgrppropertysel])
    resolveg3grplogS_l[resbgrppropertysel] = ic.get_int_mass(resolve[resbgrppropertysel].logmstar.to_numpy(), resolveg3grpid[resbgrppropertysel])
    resolveg3grplogB_l[resbgrppropertysel] = ic.get_int_mass(resolve[resbgrppropertysel].logmbary.to_numpy(), resolveg3grpid[resbgrppropertysel])

    resolveg3grpradeg[resbgrppropertysel],resolveg3grpdedeg[resbgrppropertysel],resolveg3grpcz[resbgrppropertysel]=fof.group_skycoords(resolve[resbgrppropertysel].radeg.to_numpy(),\
        resolve[resbgrppropertysel].dedeg.to_numpy(), resolve[resbgrppropertysel].cz.to_numpy(), resolveg3grpid[resbgrppropertysel])
    resolveg3fc[resbgrppropertysel] = fof.get_central_flag(resolve[resbgrppropertysel].absrmag.to_numpy(), resolveg3grpid[resbgrppropertysel])

    resolveg3satmhi=np.zeros_like(resolveg3grpid)
    resolveg3cenmhi=np.zeros_like(resolveg3grpid)
    resolveg3cenmhi[resbgrppropertysel] = fof.get_central_mass(np.log10(10**resolve[resbgrppropertysel].logmgas.to_numpy()/1.4), resolveg3grpid[resbgrppropertysel], resolveg3fc[resbgrppropertysel])
    resolveg3satmhi[resbgrppropertysel] = fof.get_satint_mass(np.log10(10**resolve[resbgrppropertysel].logmgas.to_numpy()/1.4), resolveg3grpid[resbgrppropertysel], resolveg3fc[resbgrppropertysel])
    resolveg3grpngi = np.zeros_like(resolveg3grpid)
    resolveg3grpndw = np.zeros_like(resolveg3grpid)
    resbg3grpngi = np.zeros_like(resolveg3grpid[resbgrppropertysel])
    resbg3grpndw = np.zeros_like(resolveg3grpid[resbgrppropertysel])
    for uid in np.unique(tmpid):
        grpsel = np.where(resolveg3grpid[resbgrppropertysel]==uid)
        gisel = np.where(np.logical_and((resolveg3grpid[resbgrppropertysel]==uid),(resolve[resbgrppropertysel].absrmag.to_numpy()<=-19.5)))
        dwsel = np.where(np.logical_and((resolveg3grpid[resbgrppropertysel]==uid),(resolve[resbgrppropertysel].absrmag.to_numpy()>-19.5)))
        if len(gisel[0])>0.:
            resbg3grpngi[grpsel]=len(gisel[0])
        if len(dwsel[0])>0.:
            resbg3grpndw[grpsel]=len(dwsel[0])
    resolveg3grpngi[resbgrppropertysel]=resbg3grpngi
    resolveg3grpndw[resbgrppropertysel]=resbg3grpndw
    resolveg3nndens=np.zeros_like(resolveg3grpid)
    resolveg3edgeflag=np.zeros_like(resolveg3grpid)
    resolveg3nndens2d=np.zeros_like(resolveg3grpid)
    resolveg3edgeflag2d=np.zeros_like(resolveg3grpid)
    resolveg3edgescale2d=np.zeros_like(resolveg3grpid)
    RESB_RADEG_REMAPPED = np.copy(resolve[resbgrppropertysel].radeg.to_numpy())
    REMAPSEL = np.where(RESB_RADEG_REMAPPED>18*15.)
    RESB_RADEG_REMAPPED[REMAPSEL] = RESB_RADEG_REMAPPED[REMAPSEL]-360.
    densoutput = lss_dens_by_galaxy(resolveg3grpid[resbgrppropertysel], RESB_RADEG_REMAPPED, resolve[resbgrppropertysel].dedeg.to_numpy(), resolve[resbgrppropertysel].cz.to_numpy(), resolveg3logmh[resbgrppropertysel],\
        Nnn=3, rarange=(-2*15.,3*15.), decrange=(-1.25,1.25), czrange=(4250,7250))
    resolveg3nndens[resbgrppropertysel]=densoutput[0] 
    resolveg3edgeflag[resbgrppropertysel]=densoutput[1] 
    resolveg3nndens2d[resbgrppropertysel]=densoutput[2] 
    resolveg3edgeflag2d[resbgrppropertysel]=densoutput[3] 
    resolveg3edgescale2d[resbgrppropertysel]=densoutput[4]
    resolveg3grptcross=np.zeros_like(resolveg3grpid)
    resolveg3grptcross[resbgrppropertysel]=vz.group_crossing_time(resolve[resbgrppropertysel].radeg.to_numpy(), resolve[resbgrppropertysel].dedeg.to_numpy(), resolve[resbgrppropertysel].cz.to_numpy(),\
        resolveg3grpid[resbgrppropertysel], H0=hubble_const)    

    resolveg3dynmass=np.zeros_like(resolveg3grpid)-999.
    resbdynmass = fof.dynmass(resolve[resbgrppropertysel].radeg.to_numpy(), resolve[resbgrppropertysel].dedeg.to_numpy(), resolve[resbgrppropertysel].cz.to_numpy(),\
        resolveg3grpid[resbgrppropertysel],9.9,hubble_const/100.)
    resolveg3dynmass[resbgrppropertysel]=resbdynmass
    # get RESOLVE-A from ECO
    resolvename = resolve.name.to_numpy()
    resname_in_eco = eco.resname.to_numpy()

    for ii,rn in enumerate(resolvename):
        if rn.startswith('rs'):
            ecosel = (resname_in_eco==rn)
            resolveg3grpid[ii]=eco.g3grp_l[ecosel]
            resolveg3logmh[ii]=eco.g3logmh_l[ecosel]
            resolveg3logmh200[ii]=eco.g3logmh200_l[ecosel]
            resolveg3grpradeg[ii]=eco.g3grpradeg_l[ecosel]
            resolveg3grpdedeg[ii]=eco.g3grpdedeg_l[ecosel]
            resolveg3grpcz[ii]=eco.g3grpcz_l[ecosel]
            resolveg3grplogG_l[ii]=eco.g3grplogG_l[ecosel]
            resolveg3grplogS_l[ii]=eco.g3grplogS_l[ecosel]
            resolveg3grplogB_l[ii]=eco.g3grplogB_l[ecosel]
            resolveg3grpabsrmag[ii]=eco.g3grpabsrmag_l[ecosel]
            resolveg3grpgiantabsrmag[ii]=eco.g3grpgiantabsrmag_l[ecosel]
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
    resolve.loc[:,'g3logmh200_l']=resolveg3logmh200
    resolve.loc[:,'g3grplogG_l']=resolveg3grplogG_l
    resolve.loc[:,'g3grplogS_l']=resolveg3grplogS_l
    resolve.loc[:,'g3grplogB_l']=resolveg3grplogB_l
    resolve.loc[:,'g3grpabsrmag_l']=resolveg3grpabsrmag
    resolve.loc[:,'g3grpgiantabsrmag_l']=resolveg3grpgiantabsrmag
    resolve.loc[:,'g3grpmhi_l']=np.log10(10**resolveg3grplogG_l/1.4)
    resolve.loc[(resolve.g3grp_l==-999.),'g3grpmhi_l']=-999.
    resolve.loc[:,'g3cenmhi_l']=resolveg3cenmhi
    resolve.loc[:,'g3satmhi_l']=resolveg3satmhi
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
    resolveg3dynmass[(resolveg3grpngi+resolveg3grpndw)<=7]=-999.
    resolve.loc[:,'g3logmhdyn_l']=resolveg3dynmass

    # these arrays still have -999's for B-semester sub-floor dwarfs.
    # need to add group quantities for those. If in group, fill from host group metrics.
    # if isolated sub-floor dwarf, set to galx properties.
    subfloordwarfsB = resolve[(resolve.f_b==1)&(resolve.absrmag>-17.)]
    for subfloorname,grpidvalue in zip(subfloordwarfsB.name.to_numpy(), subfloordwarfsB.g3grp_l.to_numpy()):
        fillsel = (resolve.name==subfloorname)
        grpsel = (resolve.g3grp_l==grpidvalue)&(resolve.absrmag<=-17.0)
        if np.sum(grpsel)>0:
            #eco.loc[fillsel,'g3grplogG_l']=np.mean(eco.loc[grpsel,'g3grplogG_l'])
            resolve.loc[fillsel,'g3logmh_l'] = np.mean(resolve.loc[grpsel,'g3logmh_l'])
            resolve.loc[fillsel,'g3logmh200_l']= np.mean(resolve.loc[grpsel,'g3logmh200_l'])
            resolve.loc[fillsel,'g3grplogG_l'] = np.mean(resolve.loc[grpsel,'g3grplogG_l'])
            resolve.loc[fillsel,'g3grplogS_l'] = np.mean(resolve.loc[grpsel,'g3grplogS_l'])
            resolve.loc[fillsel,'g3grplogB_l'] = np.mean(resolve.loc[grpsel,'g3grplogB_l'])
            resolve.loc[fillsel,'g3grpabsrmag_l']= np.mean(resolve.loc[grpsel,'g3grpabsrmag_l'])
            resolve.loc[fillsel,'g3grpgiantabsrmag_l']= np.mean(resolve.loc[grpsel,'g3grpgiantabsrmag_l'])
            resolve.loc[fillsel,'g3grpmhi_l']= np.mean(resolve.loc[grpsel,'g3grpmhi_l'])
            resolve.loc[fillsel,'g3cenmhi_l']= np.mean(resolve.loc[grpsel,'g3cenmhi_l'])
            resolve.loc[fillsel,'g3satmhi_l']= np.mean(resolve.loc[grpsel,'g3satmhi_l'])
            resolve.loc[fillsel,'g3grpradeg_l']= np.mean(resolve.loc[grpsel,'g3grpradeg_l'])
            resolve.loc[fillsel,'g3grpdedeg_l']= np.mean(resolve.loc[grpsel,'g3grpdedeg_l'])
            resolve.loc[fillsel,'g3grpcz_l']= np.mean(resolve.loc[grpsel,'g3grpcz_l'])
            resolve.loc[fillsel,'g3fc_l']=np.mean(resolve.loc[grpsel,'g3fc_l'])
            resolve.loc[fillsel,'g3grpngi_l']=np.mean(resolve.loc[grpsel,'g3grpngi_l'])
            resolve.loc[fillsel,'g3grpndw_l']=np.mean(resolve.loc[grpsel,'g3grpndw_l'])
            resolve.loc[fillsel,'g3grpnndens_l']=np.mean(resolve.loc[grpsel,'g3grpnndens_l'])
            resolve.loc[fillsel,'g3grpedgeflag_l']=np.mean(resolve.loc[grpsel,'g3grpedgeflag_l'])
            resolve.loc[fillsel,'g3grpnndens2d_l']= np.mean(resolve.loc[grpsel,'g3grpnndens2d_l'])
            resolve.loc[fillsel,'g3grpedgeflag2d_l']= np.mean(resolve.loc[grpsel,'g3grpedgeflag2d_l'])
            resolve.loc[fillsel,'g3grpedgescale2d_l']= np.mean(resolve.loc[grpsel, 'g3grpedgescale2d_l'])
            resolve.loc[fillsel,'g3grptcross_l']= np.mean(resolve.loc[grpsel, 'g3grptcross_l'])
            resolve.loc[fillsel,'g3logmhdyn_l'] = np.mean(resolve.loc[grpsel, 'g3logmhdyn_l'])
        else:
            resolve.loc[fillsel,'g3logmh_l'] = mhsplineEXTRAP(resolve.loc[fillsel,'absrmag'])
            resolve.loc[fillsel,'g3logmh200_l']= mhsplineEXTRAP200(resolve.loc[fillsel,'absrmag']) # fix here
            resolve.loc[fillsel,'g3grplogG_l'] = resolve.loc[fillsel,'logmgas']
            resolve.loc[fillsel,'g3grplogS_l']= resolve.loc[fillsel,'logmstar']
            resolve.loc[fillsel,'g3grplogB_l']= resolve.loc[fillsel,'logmbary']
            resolve.loc[fillsel,'g3grpabsrmag_l']=resolve.loc[fillsel,'absrmag']
            resolve.loc[fillsel,'g3grpgiantabsrmag_l']=-999.
            resolve.loc[fillsel,'g3grpmhi_l']=np.log10(10**resolve.loc[fillsel,'logmgas']/1.4)
            resolve.loc[fillsel,'g3cenmhi_l']=np.log10(10**resolve.loc[fillsel,'logmgas']/1.4)
            resolve.loc[fillsel,'g3satmhi_l']=0.
            resolve.loc[fillsel,'g3grpradeg_l']=resolve.loc[fillsel,'radeg']
            resolve.loc[fillsel,'g3grpdedeg_l']=resolve.loc[fillsel,'dedeg']
            resolve.loc[fillsel,'g3grpcz_l']=resolve.loc[fillsel,'cz']
            resolve.loc[fillsel,'g3fc_l']=1.
            resolve.loc[fillsel,'g3grpngi_l']=0.
            resolve.loc[fillsel,'g3grpndw_l']=1.
            resolve.loc[fillsel,'g3grpnndens_l']= -999.
            resolve.loc[fillsel,'g3grpedgeflag_l']= -999.
            resolve.loc[fillsel,'g3grpnndens2d_l']= -999.
            resolve.loc[fillsel,'g3grpedgeflag2d_l']= -999.
            resolve.loc[fillsel,'g3grpedgescale2d_l']= -999.
            resolve.loc[fillsel,'g3grptcross_l']= -999.
            resolve.loc[fillsel,'g3logmhdyn_l'] = -999.
    resolve.to_csv("RESOLVEdata_G3catalog_luminosity.csv",index=False)
