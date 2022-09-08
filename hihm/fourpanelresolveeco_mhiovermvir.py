import sys
sys.path.insert(0,'../g3algo/')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
import iterativecombination as ic
from center_binned_stats import center_binned_stats 
from sys import exit
from smoothedbootstrap import smoothedbootstrap as sbs
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import scipy.odr as odr
import matplotlib
from foftools import getmhoffset
matplotlib.use('TkAgg')
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']
#rcParams['text.usetex'] = True
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.500005949910059, 4.3880449973709)

def linmodel(B,x):
    return B[0]*x+B[1]


if __name__=='__main__':
    eco = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco.loc[:,'logmgas'] = np.log10(10**eco['logmgas']/1.4)
    eco = eco.set_index('name')

    ecog3 = eco[(eco.absrmag<-17.33) & (eco.g3grpcz_l<7000) & (eco.g3grpcz_l>3000) & (eco.dup<1)]
    ecog3.loc[:,'logmgas'] = np.log10(10**ecog3.logmgas / 10**ecog3.g3logmh_l)
    
    eco = pd.read_csv("ECODR3_Jul0822.csv")
    eco.loc[:,'logmgas'] = np.log10(10**eco['logmgas']/1.4)
    ecofof = eco[(eco.absrmag<-17.33) & (eco.grpcz<7000) & (eco.grpcz>3000) & (eco.dup<1)]
    ecofof.loc[:,'logmh']=ecofof.logmhvir
    ecofof.loc[:,'logmgas'] = np.log10(10**ecofof.logmgas / 10**ecofof.logmhvir)

    # (1) HI-halo mass relation for centrals, summed-over-satellites, total group-integrated HI
    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(figsize=(doublecolsize[0],2*doublecolsize[1]), nrows=2, ncols=2)
    ax1.annotate("ECO: G3 Groups", xy=(12.6,-2), )
    centralsel=(ecog3.g3fc_l==1)
    ax1.scatter(ecog3.g3logmh_l[centralsel], (ecog3.logmgas[centralsel]), marker='.', color='lightgreen', alpha=0.5, s=5, label='Central HI', rasterized=True)
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.15,14.75]
    centrallogmgas = np.array(ecog3[centralsel].logmgas)
    centrallogmh = np.array(ecog3[centralsel].g3logmh_l)
    median_central_hi, bincenters, binedges, jnk = center_binned_stats(ecog3[centralsel].g3logmh_l, ecog3[centralsel].logmgas, statistic='median', bins=binvalues)
    median_central_hi_error = np.std(np.array([sbs(centrallogmgas[np.where(np.logical_and(centrallogmh>binedges[i-1], centrallogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax1.errorbar(bincenters, median_central_hi, yerr=median_central_hi_error, fmt='^-', color='green', label=r'$M_{\rm HI,cen}/M_{\rm vir}$', rasterized=True)
    sats = ecog3[ecog3.g3fc_l==0]
    sats['totalhi_sat'] = ic.get_int_mass(sats.logmgas, sats.g3grp_l)
    sats = sats.drop_duplicates(subset=['g3grp_l'], keep='first')
    ax1.scatter(sats.g3logmh_l, sats.totalhi_sat, marker='.', color='mediumorchid', alpha=0.7, s=5, label='Sat. HI', rasterized=True)
    satlogmgas = np.array(sats.totalhi_sat)
    satlogmh = np.array(sats.g3logmh_l)
    median_sat_totalhi, bincenters, binedges, jnk = center_binned_stats(sats.g3logmh_l, sats.totalhi_sat, statistic='median', bins=binvalues)
    median_sat_hi_error = np.std(np.array([sbs(satlogmgas[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax1.errorbar(bincenters, median_sat_totalhi, yerr=median_sat_hi_error, fmt='^-', color='purple', label=r'$M_{\rm HI\, sat}/M_{\rm vir}$', rasterized=True)
    ecog3['totalgrouphi'] = ic.get_int_mass(ecog3.logmgas, ecog3.g3grp_l)
    ecog3groups = ecog3.drop_duplicates(subset=['g3grp_l'], keep='first')
    ecog3groupslogmh = np.array(ecog3groups.g3logmh_l)
    ecog3groupslogmgas = np.array(ecog3groups.totalgrouphi)
    median_totalhi, bincenters, binedges, jk = center_binned_stats(ecog3groupslogmh, ecog3groupslogmgas, statistic='median', bins=binvalues)
    median_totalhi_error = np.std(np.array([sbs(ecog3groupslogmgas[np.where(np.logical_and(ecog3groupslogmh>binedges[i-1], ecog3groupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    median_logmh_error = np.std(np.array([sbs(ecog3groupslogmh[np.where(np.logical_and(ecog3groupslogmh>binedges[i-1], ecog3groupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax1.errorbar(bincenters, median_totalhi, yerr=median_totalhi_error, xerr=median_logmh_error, fmt='^-', label=r'$M_{\rm HI,\, grp}/M_{\rm vir}$', color='k', rasterized=True)
    #############################################
    print("Best Fit Parameters (G3)")
    #params = np.polyfit(bincenters, median_totalhi, 1)
    #linear = odr.Model(linmodel)
    #fitdata = odr.Data(bincenters, median_totalhi, wd=1/(median_logmh_error)**2., we=1/(median_totalhi_error)**2.)
    #fit = odr.ODR(fitdata, linear, beta0=params)
    #fitoutput = fit.run()
    #fitoutput.pprint()
    poptg3, pcovg3 = np.polyfit(bincenters[bincenters<=11.5], median_totalhi[bincenters<=11.5], deg=1, cov=True)
    print("best fit for g3 (<11.5): ", poptg3, np.sqrt(np.diag(pcovg3)))
    poptg3_2, pcovg3_2 = np.polyfit(bincenters[bincenters>11.5], median_totalhi[bincenters>11.5], deg=1, cov=True)
    print("best fit for g3_2 (>11.5): ", poptg3_2, np.sqrt(np.diag(pcovg3_2)))
    
    sellow = ecog3groupslogmh<11.5
    residlow = np.abs(ecog3groupslogmgas[sellow]-np.poly1d(poptg3)(ecog3groupslogmh[sellow]))
    selhigh = ~sellow
    residhigh = np.abs(ecog3groupslogmgas[selhigh]-np.poly1d(poptg3_2)(ecog3groupslogmh[selhigh]))
    resid = np.concatenate([residlow,residhigh])

    print("median scatter: ", np.median(resid))
    #############################################
    #ax1.set_ylabel(r"log HI gas fraction")
    ax1.set_ylabel(r"log HI mass / $M_{\rm vir}$")
    ax1.legend(loc='lower left', framealpha=1)
    ax1.set_ylim(-6,-1.5)
    ax3.set_ylim(-6,-1.5)
    ax1.axvline(11.4, color='k', alpha=0.1)
    ax1.axvline(12.1, color='k', alpha=0.1)

    g3table = dict({r'$logM_h$':bincenters, 'median group HI':median_totalhi, 'median group HI error':median_totalhi_error,
                     'median satellite-summed HI':median_sat_totalhi, 'median satellite-summed HI error':median_sat_hi_error,
                    'median central HI':median_central_hi, 'median central HI error':median_central_hi_error})
    g3table = pd.DataFrame(g3table)

    ################################################
    ################################################
    ################################################
    # Do same thing for FoF

    centralsel=(ecofof.fc==1)
    #ax3.set_title("RESOLVE+ECO: FoF Groups")
    ax3.annotate("ECO: FoF Groups", xy=(12.6,-2), )
    ax3.scatter(ecofof.logmh[centralsel], (ecofof.logmgas[centralsel]), marker='.', color='lightgreen', alpha=0.5, s=5, label='Centrals', rasterized=True)
    centrallogmgas = np.array(ecofof[centralsel].logmgas)
    centrallogmh = np.array(ecofof[centralsel].logmh)
    median_central_hi, bincenters, binedges, jnk = center_binned_stats(ecofof[centralsel].logmh, ecofof[centralsel].logmgas, statistic='median', bins=binvalues)
    median_central_hi_error = np.std(np.array([sbs(centrallogmgas[np.where(np.logical_and(centrallogmh>binedges[i-1], centrallogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax3.errorbar(bincenters, median_central_hi, yerr=median_central_hi_error, fmt='^-', color='green', label='Centrals (Medians)', rasterized=True)
    sats = ecofof[ecofof.fc==0]
    sats['totalhi_sat'] = ic.get_int_mass(sats.logmgas, sats.grp)
    sats = sats.drop_duplicates(subset=['grp'], keep='first') # use only group properties after this point
    ax3.scatter(sats.logmh, sats.totalhi_sat, marker='.', color='mediumorchid', alpha=0.7, s=5, label='Summed over Satellites', rasterized=True)
    satslogmgas = np.array(sats.totalhi_sat)
    satslogmh = np.array(sats.logmh)
    median_sat_totalhi, bincenters, binedges, jnk = center_binned_stats(sats.logmh, sats.totalhi_sat, statistic='median', bins=binvalues)
    median_sat_hi_error = np.std(np.array([sbs(satlogmgas[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax3.errorbar(bincenters, median_sat_totalhi, yerr=median_sat_hi_error, fmt='^-', color='purple', label='Summed over Satellites (Medians)', rasterized=True)
    ecofof['totalgrouphi'] = ic.get_int_mass(ecofof.logmgas, ecofof.grp)
    ecofofgroups = ecofof.drop_duplicates(subset=['grp'], keep='first')
    ecofofgroupslogmgas = np.array(ecofofgroups.totalgrouphi)
    ecofofgroupslogmh = np.array(ecofofgroups.logmh)
    median_totalhi, bincenters, binedges, jk = center_binned_stats(ecofofgroupslogmh, ecofofgroupslogmgas, statistic='median', bins=binvalues)
    median_totalhi_error = np.std(np.array([sbs(ecofofgroupslogmgas[np.where(np.logical_and(ecofofgroupslogmh>binedges[i-1], ecofofgroupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)


    median_logmh_error = np.std(np.array([sbs(ecofofgroupslogmh[np.where(np.logical_and(ecofofgroupslogmh>binedges[i-1], ecofofgroupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)

    ax3.errorbar(bincenters, median_totalhi, xerr=median_logmh_error, yerr=median_totalhi_error, fmt='^-', label='Total Group-Integrated HI (Medians)', color='k', rasterized=True)
    ax1.set_xlim(10.9,14.7)
    ax3.set_xlim(10.9,14.7)

    #######################################################################################
    #######################################################################################
    #######################################################################################
    #######################################################################################
    #######################################################################################
    # now resolve

    eco = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco = eco[eco.resname!='notinresolve']
    eco.loc[:,'logmgas'] = np.log10(10**eco['logmgas']/1.4)
    eco = eco.set_index('name')

    ecog3 = eco[(eco.absrmag<-17.33) & (eco.g3grpcz_l<7000) & (eco.g3grpcz_l>3000) & (eco.dup<1)]
    ecog3.loc[:,'logmgas'] = np.log10(10**ecog3.logmgas / 10**ecog3.g3logmh_l)

    eco = pd.read_csv("ECODR3_Jul0822.csv")
    eco = eco[eco.resname!='notinresolve']
    eco.loc[:,'logmgas'] = np.log10(10**eco['logmgas']/1.4)
    ecofof = eco[(eco.absrmag<-17.33) & (eco.grpcz<7000) & (eco.grpcz>3000) & (eco.dup<1)]
    ecofof.loc[:,'logmgas'] = np.log10(10**ecofof.logmgas / 10**ecofof.logmhvir)

    resolve = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resolve = resolve.set_index('name')
    resolve['dup'] = np.zeros_like(np.array(resolve.radeg))
    resolve.loc[:,'logmgas'] = np.log10(10**resolve['logmgas']/1.4)
    resg3 = resolve[(resolve.fl_insample==1) & (resolve.g3grpcz_l>4500) & (resolve.g3grpcz_l<7000)]
    resg3.loc[:,'logmgas'] = np.log10(10**resg3.logmgas / 10**resg3.g3logmh_l)
    
    resolve = pd.read_csv("RESOLVEliving_071322_updatedfofgroups.csv")
    resolvefof = resolve[(resolve.fl_insample==1) & (resolve.grpcz<7000) & (resolve.grpcz>4500) & (resolve.logmh>0)]
    resolvefof.loc[:,'logmgas'] = np.log10(10**resolvefof.logmgas / 10**resolvefof.logmhvir)

    # merge resolve-b into ECO
    resolvefof = resolvefof[resolvefof.f_b==1]
    resolveg3 = resg3[resg3.f_b==1]

    ecofof = pd.concat([ecofof,resolvefof])
    ecog3 = pd.concat([ecog3,resolveg3])
    ecofof.loc[:,'logmh']=ecofof.logmhvir

    print('min halo mass')
    print(np.min(resolveg3.g3logmh_l),np.min(ecog3.g3logmh_l))
    # (1) HI-halo mass relation for centrals, summed-over-satellites, total group-integrated HI
    ax2.annotate("RESOLVE: G3 Groups", xy=(12.6,-2), )
    centralsel=(ecog3.g3fc_l==1)
    ax2.scatter(ecog3.g3logmh_l[centralsel], (ecog3.logmgas[centralsel]), marker='.', color='lightgreen', alpha=0.5, s=5, label='Central HI', rasterized=True)
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.15,14.75]
    centrallogmgas = np.array(ecog3[centralsel].logmgas)
    centrallogmh = np.array(ecog3[centralsel].g3logmh_l)
    median_central_hi, bincenters, binedges, jnk = center_binned_stats(ecog3[centralsel].g3logmh_l, ecog3[centralsel].logmgas, statistic='median', bins=binvalues)
    median_central_hi_error = np.std(np.array([sbs(centrallogmgas[np.where(np.logical_and(centrallogmh>binedges[i-1], centrallogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax2.errorbar(bincenters, median_central_hi, yerr=median_central_hi_error, fmt='^-', color='green', label=r'$M_{\rm HI,cen}$', rasterized=True)
    sats = ecog3[ecog3.g3fc_l==0]
    sats['totalhi_sat'] = ic.get_int_mass(sats.logmgas, sats.g3grp_l)
    sats = sats.drop_duplicates(subset=['g3grp_l'], keep='first')
    ax2.scatter(sats.g3logmh_l, sats.totalhi_sat, marker='.', color='mediumorchid', alpha=0.7, s=5, label='Satellite-Integrated HI', rasterized=True)
    satlogmgas = np.array(sats.totalhi_sat)
    satlogmh = np.array(sats.g3logmh_l)
    median_sat_totalhi, bincenters, binedges, jnk = center_binned_stats(sats.g3logmh_l, sats.totalhi_sat, statistic='median', bins=binvalues)
    median_sat_hi_error = np.std(np.array([sbs(satlogmgas[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax2.errorbar(bincenters, median_sat_totalhi, yerr=median_sat_hi_error, fmt='^-', color='purple', label=r'$M_{\rm HI\, sat}$', rasterized=True)
    ecog3['totalgrouphi'] = ic.get_int_mass(ecog3.logmgas, ecog3.g3grp_l)
    ecog3groups = ecog3.drop_duplicates(subset=['g3grp_l'], keep='first')
    ecog3groupslogmh = np.array(ecog3groups.g3logmh_l)
    ecog3groupslogmgas = np.array(ecog3groups.totalgrouphi)
    median_totalhi, bincenters, binedges, jk = center_binned_stats(ecog3groupslogmh, ecog3groupslogmgas, statistic='median', bins=binvalues)
    median_totalhi_error = np.std(np.array([sbs(ecog3groupslogmgas[np.where(np.logical_and(ecog3groupslogmh>binedges[i-1], ecog3groupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    median_logmh_error = np.std(np.array([sbs(ecog3groupslogmh[np.where(np.logical_and(ecog3groupslogmh>binedges[i-1], ecog3groupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax2.errorbar(bincenters, median_totalhi, yerr=median_totalhi_error, xerr=median_logmh_error, fmt='^-', label=r'$M_{\rm HI,\, grp}$', color='k', rasterized=True)
    #############################################
    print("Best Fit Parameters (G3)")
    #params = np.polyfit(bincenters, median_totalhi, 1)
    #linear = odr.Model(linmodel)
    #fitdata = odr.Data(bincenters, median_totalhi, wd=1/(median_logmh_error)**2., we=1/(median_totalhi_error)**2.)
    #fit = odr.ODR(fitdata, linear, beta0=params)
    #fitoutput = fit.run()
    #fitoutput.pprint()
    poptg3, pcovg3 = np.polyfit(bincenters[bincenters<=11.5], median_totalhi[bincenters<=11.5], deg=1, cov=True)
    print("best fit for g3 (<11.5): ", poptg3, np.sqrt(np.diag(pcovg3)))
    poptg3_2, pcovg3_2 = np.polyfit(bincenters[bincenters>11.5], median_totalhi[bincenters>11.5], deg=1, cov=True)
    print("best fit for g3_2 (>11.5): ", poptg3_2, np.sqrt(np.diag(pcovg3_2)))
    
    sellow = ecog3groupslogmh<11.5
    residlow = np.abs(ecog3groupslogmgas[sellow]-np.poly1d(poptg3)(ecog3groupslogmh[sellow]))
    selhigh = ~sellow
    residhigh = np.abs(ecog3groupslogmgas[selhigh]-np.poly1d(poptg3_2)(ecog3groupslogmh[selhigh]))
    resid = np.concatenate([residlow,residhigh])

    print("median scatter: ", np.median(resid))
    #############################################
    ax2.set_xlabel(r"log group $M_{\rm vir}$")
    ax4.set_xlabel(r"log group $M_{\rm vir}$")
    ax2.set_ylabel(r"log HI mass / $M_{\rm vir}$")
    #ax2.set_ylabel(r"log group-integrated HI mass fraction")
    ax2.set_ylim(-6,-1.5)
    ax2.axvline(11.4, color='k', alpha=0.1)
    ax2.axvline(12.1, color='k', alpha=0.1)

    g3table = dict({r'$logM_h$':bincenters, 'median group HI':median_totalhi, 'median group HI error':median_totalhi_error,
                     'median satellite-summed HI':median_sat_totalhi, 'median satellite-summed HI error':median_sat_hi_error,
                    'median central HI':median_central_hi, 'median central HI error':median_central_hi_error})
    g3table = pd.DataFrame(g3table)

    ################################################
    ################################################
    ################################################
    # Do same thing for FoF

    centralsel=(ecofof.fc==1)
    #ax4.set_title("RESOLVE+ECO: FoF Groups")
    ax4.annotate("RESOLVE: FoF Groups", xy=(12.6,-2), )
    ax4.scatter(ecofof.logmh[centralsel], (ecofof.logmgas[centralsel]), marker='.', color='lightgreen', alpha=0.5, s=5, label='Centrals', rasterized=True)
    centrallogmgas = np.array(ecofof[centralsel].logmgas)
    centrallogmh = np.array(ecofof[centralsel].logmh)
    median_central_hi, bincenters, binedges, jnk = center_binned_stats(ecofof[centralsel].logmh, ecofof[centralsel].logmgas, statistic='median', bins=binvalues)
    median_central_hi_error = np.std(np.array([sbs(centrallogmgas[np.where(np.logical_and(centrallogmh>binedges[i-1], centrallogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax4.errorbar(bincenters, median_central_hi, yerr=median_central_hi_error, fmt='^-', color='green', label='Centrals (Medians)', rasterized=True)
    sats = ecofof[ecofof.fc==0]
    sats['totalhi_sat'] = ic.get_int_mass(sats.logmgas, sats.grp)
    sats = sats.drop_duplicates(subset=['grp'], keep='first') # use only group properties after this point
    ax4.scatter(sats.logmh, sats.totalhi_sat, marker='.', color='mediumorchid', alpha=0.7, s=5, label='Summed over Satellites', rasterized=True)
    satslogmgas = np.array(sats.totalhi_sat)
    satslogmh = np.array(sats.logmh)
    median_sat_totalhi, bincenters, binedges, jnk = center_binned_stats(sats.logmh, sats.totalhi_sat, statistic='median', bins=binvalues)
    median_sat_hi_error = np.std(np.array([sbs(satlogmgas[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax4.errorbar(bincenters, median_sat_totalhi, yerr=median_sat_hi_error, fmt='^-', color='purple', label='Summed over Satellites (Medians)', rasterized=True)
    ecofof['totalgrouphi'] = ic.get_int_mass(ecofof.logmgas, ecofof.grp)
    ecofofgroups = ecofof.drop_duplicates(subset=['grp'], keep='first')
    ecofofgroupslogmgas = np.array(ecofofgroups.totalgrouphi)
    ecofofgroupslogmh = np.array(ecofofgroups.logmh)
    median_totalhi, bincenters, binedges, jk = center_binned_stats(ecofofgroupslogmh, ecofofgroupslogmgas, statistic='median', bins=binvalues)
    median_totalhi_error = np.std(np.array([sbs(ecofofgroupslogmgas[np.where(np.logical_and(ecofofgroupslogmh>binedges[i-1], ecofofgroupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)


    median_logmh_error = np.std(np.array([sbs(ecofofgroupslogmh[np.where(np.logical_and(ecofofgroupslogmh>binedges[i-1], ecofofgroupslogmh<=binedges[i]))],\
                               5000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)

    ax4.errorbar(bincenters, median_totalhi, xerr=median_logmh_error, yerr=median_totalhi_error, fmt='^-', label='Total Group-Integrated HI (Medians)', color='k', rasterized=True)
    ax2.set_xlim(10.9,14.7)
    ax4.set_ylim(-6,-1.5)
    ax4.set_xlim(10.9,14.7)



    #######################################################################################
    #######################################################################################
    #######################################################################################
    #######################################################################################
    #######################################################################################
    #ax3.set_ylim(-6,-1.5)
    ax3.axvline(11.4, color='k', alpha=0.1)
    ax3.axvline(12.1, color='k', alpha=0.1)
    plt.tight_layout()
    plt.savefig("ecoresolve_mhiovermvir.pdf")
    plt.show()
