import sys
sys.path.insert(0,'../g3algo/')
import matplotlib
matplotlib.use('TkAgg')
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
from scipy.stats import ks_2samp, mannwhitneyu
from scipy.interpolate import interp1d 
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
doublecolsize = (7.100005949910059, 4.3880449973709)

if __name__=='__main__':
    eco = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco.loc[:,'logmgas'] = np.log10(10**eco['logmgas']/1.4)
    eco = eco.set_index('name')

    ecog3 = eco[(eco.absrmag<-17.33) & (eco.g3grpcz_l<7000) & (eco.g3grpcz_l>3000) & (eco.dup<1)]

    ecofof = eco[(eco.absrmag<-17.33) & (eco.grpcz<7000) & (eco.grpcz>3000) & (eco.dup<1)]

    resolve = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resolve = resolve.set_index('name')
    resolve.loc[:,'dup'] = np.zeros_like(np.array(resolve.radeg))
    resolve.loc[:,'logmgas'] = np.log10(10**resolve['logmgas']/1.4)
    resg3 = resolve[(resolve.fl_insample==1) & (resolve.g3grpcz_l>4500) & (resolve.g3grpcz_l<7000)]
    resolvefof = resolve[(resolve.fl_insample==1) & (resolve.grpcz<7000) & (resolve.grpcz>4500) & (resolve.logmh>0)]

    # merge resolve-b into ECO
    resolvefof = resolvefof[resolvefof.f_b==1]
    resolveg3 = resg3[resg3.f_b==1]

    ecofof = pd.concat([ecofof,resolvefof])
    ecog3 = pd.concat([ecog3,resolveg3])

    #ecog3.loc[:,'latefrac'] = get_late_frac(np.array(ecog3.g3grp_l), np.array(ecog3.morphel))
    g3latefrac = np.zeros(len(ecog3))
    for uid in np.unique(ecog3.g3grp_l):
        sel = (ecog3.g3grp_l==uid)
        tempgrp = ecog3[sel]
        totalmembers = len(tempgrp)
        latefrac = len(tempgrp[tempgrp.morphel=='L'])/totalmembers
        g3latefrac[sel]=latefrac

    ecog3['latefrac'] = g3latefrac



    ##### examine satellites
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=doublecolsize,sharey=True)
    centralsel=(ecog3.g3fc_l==1)
    binvalues = np.arange(10.5,15,0.25)
    centralbinv = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.15,14.75]
    binvalues=centralbinv
    centrallogmgas = np.array(ecog3[centralsel].logmgas)
    centrallogmh = np.array(ecog3[centralsel].g3logmh_l)
    median_central_hi, bincenters, binedges, jnk = center_binned_stats(ecog3[centralsel].g3logmh_l, ecog3[centralsel].logmgas, statistic='median', bins=centralbinv)

    median_central_hi_error = np.std(np.array([sbs(centrallogmgas[np.where(np.logical_and(centrallogmh>binedges[i-1], centrallogmh<=binedges[i]))],\
                               1000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax1.errorbar(bincenters, median_central_hi, yerr=median_central_hi_error, fmt='^-', color='black', label=r'$M_{\rm HI,\, cen}$', zorder=15, rasterized=True)
    sats = ecog3[ecog3.g3fc_l==0]
    sats.loc[:,'totalhi_sat'] = np.log10(10**ic.get_int_mass(sats.logmgas, sats.g3grp_l))
    sats = sats.drop_duplicates(subset=['g3grp_l'], keep='first')
    sc=ax1.scatter(sats.g3logmh_l, sats.totalhi_sat, marker='.', c=sats.latefrac, alpha=0.7, s=5, rasterized=True)

    #===============================
    print('----------- medians -----------')
    print(np.median(sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac>0.6)]))
    print(np.median(sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac<0.4)]))
    print(ks_2samp(np.array(sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac>0.6)]),\
        np.array(sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac<0.4)]),
        alternative='less'))
    print(np.median(sats.totalhi_sat[(sats.latefrac>0.6)]))
    print(np.median(sats.totalhi_sat[(sats.latefrac<0.4)]))
    print(ks_2samp(np.array(sats.totalhi_sat[(sats.latefrac>0.6)]),\
        np.array(sats.totalhi_sat[(sats.latefrac<0.4)]),
        alternative='less'))
    #print(wx(sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac>0.5)],\
    #    sats.totalhi_sat[(sats.g3logmh_l>12)&(sats.g3logmh_l<13)&(sats.latefrac<0.5)],
    #    alternative='greater'))

    fig.colorbar(sc, label='Late-Type Satellite Fraction')
    satlogmgas = np.array(sats.totalhi_sat)
    satlogmh = np.array(sats.g3logmh_l)
    median_sat_totalhi, bincenters, binedges, jnk = center_binned_stats(sats.g3logmh_l, sats.totalhi_sat, statistic='median', bins=binvalues)
    median_sat_hi_error = np.std(np.array([sbs(satlogmgas[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               1000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax1.errorbar(bincenters, median_sat_totalhi, yerr=median_sat_hi_error, fmt='^-', color='purple', label=r'$M_{\rm HI,\, sat}$', rasterized=True)
    nanplaces = np.isnan(median_sat_totalhi)

    ax1.set_xlabel(r"log group $M_{\rm vir}$")
    ax1.set_ylabel(r"log satellite-integrated HI mass")
    ax1.legend(loc='best', framealpha=1)
    #ax1.set_ylim(8.8,10.5)
    #ax1.set_xlim(11,14)
    #ax1.axvline(11.4, color='k', alpha=0.1)
    #ax1.axvline(12.1, color='k', alpha=0.1)
    # measure scatter
    medianspline = interp1d(bincenters,median_sat_totalhi,'linear',fill_value='extrapolate')
    print('Median Residual from Spline: ', np.median(np.abs(sats.totalhi_sat - medianspline(sats.g3logmh_l))))

    #ax2.set_ylabel(r'log (HI mass / N$_{\rm grp}$)')
    ax2.set_ylabel(r"log satellite-integrated HI mass / $N_{\rm galaxies}$")
    ax2.set_xlabel(r"log group $M_{\rm vir}$")
    sc=ax2.scatter(sats.g3logmh_l, np.log10(10**sats.totalhi_sat/(sats.g3grpngi_l+sats.g3grpndw_l)), marker='.', c=sats.latefrac, alpha=0.7, s=5, rasterized=True)
    satlogmgas_per_n = np.asarray(np.log10(10**sats.totalhi_sat/(sats.g3grpngi_l+sats.g3grpndw_l)))
    median_sat_totalhi_pern, bincenters, binedges, _ = center_binned_stats(sats.g3logmh_l, np.log10(10**sats.totalhi_sat/(sats.g3grpngi_l+sats.g3grpndw_l)), statistic='median', bins=binvalues)
    median_sat_hi_pern_error = np.std(np.array([sbs(satlogmgas_per_n[np.where(np.logical_and(satlogmh>binedges[i-1], satlogmh<=binedges[i]))],\
                               1000, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    ax2.errorbar(bincenters, median_sat_totalhi_pern, yerr=median_sat_hi_pern_error, fmt='^-', color='purple', label='Medians',rasterized=True)
    ax2.set_xticks([11,12,13,14,15])
    ax1.set_xticks([11,12,13,14,15])
    ax2.legend(loc='lower right', framealpha=1)
    plt.tight_layout()
    plt.savefig("satelliteHIHM.pdf")
    plt.show()
