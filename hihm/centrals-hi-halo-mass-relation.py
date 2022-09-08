import sys
sys.path.insert(0,'../g3algo/')
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
import iterativecombination as ic
from sys import exit
from smoothedbootstrap import smoothedbootstrap as sbs
from center_binned_stats import center_binned_stats
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
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
 
    binv = binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.15,14.75] #np.arange(10.5,15,0.25) 
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=doublecolsize, sharey=True)
    g3centrals = ecog3[ecog3.g3fc_l==1]
    cm = plt.cm.get_cmap('RdYlBu_r')
    g3centrals_lt = g3centrals[g3centrals.morphel=='L']
    sc=ax.scatter(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, s=3, marker='.', color='lightblue', label='Late-Type Centrals', rasterized=True)#c=np.log10(g3centrals_lt.modelc_r), cmap=cm)
    medianlthi, bincenters, jk, jk = center_binned_stats(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, medianlthi, '^-',color='darkblue', label='Medians (Late Type Centrals)', rasterized=True)
    g3centrals_et = g3centrals[g3centrals.morphel=='E']
    sc=ax.scatter(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, s=5, marker='*', color='salmon', label='Early-Type Centrals', rasterized=True)#c=np.log10(g3centrals_et.modelu_r), cmap=cm)
    medianethi, jk, jk, jk = center_binned_stats(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, medianethi, '^-', color='darkred', label='Medians (Early Type Centrals)')
    mediantotal, jk, jk, jk = center_binned_stats(g3centrals.g3logmh_l, g3centrals.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, mediantotal, '^-', color='k', label=r'Medians (All Centrals, $M_{\rm HI,\, cen}$)', rasterized=True)
    ax.set_ylim(7,12)
    ax.set_xlim(10.9,15)
    ax.set_xlabel(r"log group $M_{\rm vir}$")
    ax.set_ylabel(r"log HI mass")
    ax.legend(loc='best')

    g3centrals = ecog3[ecog3.g3fc_l==1]
    cm = plt.cm.get_cmap('RdYlBu_r')
    sc=ax1.scatter(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, s=2, marker='.', c=g3centrals_lt.modelu_r, cmap=cm, rasterized=True)
    sc=ax1.scatter(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, s=6, marker='*', c=g3centrals_et.modelu_r, cmap=cm, rasterized=True)
    ax1.plot(0,0,'k.',markersize=2,label='Late-Type Centrals')  
    ax1.plot(0,0,'k*',markersize=6,label='Early-Type Centrals')  
    ax1.set_ylim(7,12)
    ax1.set_xlim(10.9,15)
    #ax1.set_xticks(np.arange(11,14.5,0.5))
    #ax1.set_xticklabels(ax.get_xticklabels())
    ax1.set_xlabel(r"log group $M_{\rm vir}$")
    clb=plt.colorbar(sc,label='$u-r$ color')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("centralHIHMrelation.pdf")
    plt.show()


    ###################################
    ###################################
    ###################################
    # Same plot with RESOLVE
    binv = np.arange(10.5,15,0.25)
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(14,7), sharey=True)
    g3centrals = resolveg3[resolveg3.g3fc_l==1]
    cm = plt.cm.get_cmap('RdYlBu')
    g3centrals_lt = g3centrals[g3centrals.morphel=='L']
    sc=ax.scatter(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, s=3, marker='.', color='lightblue', label='Late-Type Centrals')#c=np.log10(g3centrals_lt.modelu_r), cmap=cm)
    medianlthi, bincenters, jk, jk = center_binned_stats(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, medianlthi, '^-',color='darkblue', label='Median HI Mass (Late Type Centrals)')
    g3centrals_et = g3centrals[g3centrals.morphel=='E']
    sc=ax.scatter(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, s=5, marker='*', color='salmon', label='Early-Type Centrals')#c=np.log10(g3centrals_et.modelu_r), cmap=cm)
    medianethi, jk, jk, jk = center_binned_stats(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, medianethi, '^-', color='darkred', label='Median HI Mass (Early Type Centrals)')
    mediantotal, jk, jk, jk = center_binned_stats(g3centrals.g3logmh_l, g3centrals.logmgas, statistic=np.median, bins=binv)
    ax.plot(bincenters, mediantotal, '^-', color='k', label='Median HI Mass (All Centrals)')
    ax.set_ylim(7,12)
    ax.set_xlabel(r"log group $M_{337}$")
    ax.set_ylabel("log central HI mass")
    ax.legend(loc='best')

    g3centrals = resolveg3[resolveg3.g3fc_l==1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc=ax1.scatter(g3centrals_lt.g3logmh_l, g3centrals_lt.logmgas, s=2, marker='.', label='Late-Type Centrals', c=-1*g3centrals_lt.modelu_r, cmap=cm)
    sc=ax1.scatter(g3centrals_et.g3logmh_l, g3centrals_et.logmgas, s=6, marker='*', label='Early-Type Centrals', c=-1*g3centrals_et.modelu_r, cmap=cm)
    ax1.set_ylim(7,12)
    ax1.set_xlabel(r"log group $M_{337}$")
    clb=plt.colorbar(sc,label='$u-r$ color')
    plt.legend(loc='best')
    plt.show()
