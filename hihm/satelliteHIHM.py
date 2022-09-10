import sys
sys.path.insert(0,'../g3algo/')
from center_binned_stats import center_binned_stats as cbs
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from seaborn import kdeplot
from scipy.stats import ks_2samp
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
import pickle

if __name__=='__main__':
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75]
    ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)&(ecog3.g3fc_l<1)]
    resg3 = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resg3 = resg3[(resg3.fl_insample==1)&(resg3.g3grpcz_l>3000)&(resg3.g3grpcz_l<7000)&(resg3.g3fc_l<1)&(resg3.f_b==1)]
    g3 = pd.concat([ecog3,resg3])

    g3latefrac = np.zeros(len(g3))
    for uid in np.unique(g3.g3grp_l):
        sel = (g3.g3grp_l==uid)
        tempgrp = g3[sel]
        totalmembers = len(tempgrp)
        latefrac = len(tempgrp[tempgrp.morphel=='L'])/totalmembers
        g3latefrac[sel]=latefrac
    g3.loc[:,'latefrac'] = g3latefrac    

    g3 = g3.groupby('g3grp_l').first()

    #################
    ylimits = (7,11.1)
    xlimits = (10.9,15)
    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=(doublecolsize[0],doublecolsize[1]*0.8))
    
    xval = g3.g3logmh_l.to_numpy()
    yval = g3.g3satmhi_l.to_numpy()
    zval = g3.latefrac.to_numpy()
    ax.scatter(xval,yval,c=zval,s=2, cmap='brg')
    median, bc, _, _ = cbs(xval,yval,np.nanmedian,bins=binvalues)
    ax.plot(bc, median, color='purple', linewidth=3, label=r'$M_{\rm HI,\, sat}$ (Median)')
    cendata = pickle.load(open('cenhi.pkl','rb'))
    ax.plot(cendata[0],cendata[1],color='black',linewidth=2,label=r'$M_{\rm HI,\, cen}$ (Median)')
    ax.set_xlabel(r"log group $M_{\rm vir}$ [$M_{\odot}$]")
    ax.set_ylabel(r"log satellite-integrated HI mass [$M_{\odot}$]")
    ax.legend(loc='best')
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)

    print('do f_late dominated groups have statistically higher MHI at fixed mass?')
    print(ks_2samp(yval[zval<0.4], yval[zval>0.6], 'greater')) 
    print(ks_2samp(yval[zval<0.4], yval[zval>0.6], 'less')) 
    yval = np.log10(10**yval/(g3.g3grpngi_l+g3.g3grpndw_l-1).to_numpy())
    sc=ax1.scatter(xval, yval, c=zval, s=2, cmap='brg') 
    median, bc, _, _ = cbs(xval,yval,np.nanmedian,bins=binvalues)
    ax1.plot(bc, median, color='purple', linewidth=3, label=r'$M_{\rm HI,\, sat}$ (Median)') 
    ax1.set_xlabel(r"log group $M_{\rm vir}$ [$M_{\odot}$]")
    ax1.set_ylabel(r"log satellite-integrated HI mass / $N_{\rm sat}$ [$M_{\odot}$]")
    ax1.set_xlim(*xlimits)
    ax1.set_ylim(*ylimits)
    fig.colorbar(sc, label='Late-type Satellite Fraction')
    ax1.legend(loc='best')
    plt.tight_layout() 
    plt.show()

    
