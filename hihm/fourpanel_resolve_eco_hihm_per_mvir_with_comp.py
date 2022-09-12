import sys
sys.path.insert(0,'../g3algo/')
import iterativecombination as ic
from center_binned_stats import center_binned_stats as cbs
from smoothedbootstrap import smoothedbootstrap as sbs
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from scipy.ndimage import median_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.100005949910059, 4.3880449973709)

def make_panel(cax,xv,yv,binvalues,xlim,ylim,xlab,ylab,ptcolor='gray',linecolor='k',ptalpha=0.3,linelabel=None):
    cax.scatter(xv,yv,color=ptcolor,alpha=ptalpha,s=2)
    median,bc,binedges,_ = cbs(xv, yv, 'median', bins=binvalues)
    medianerr = np.std(np.array([sbs(yv[np.where(np.logical_and(xv>binedges[i-1], xv<=binedges[i]))],\
                       50, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    cax.errorbar(bc,median,yerr=medianerr,fmt='o-',color=linecolor,rasterized=True, markersize=4, label=linelabel)
    #idx = np.argsort(xv)
    #xmed = median_filter(xv[idx],winsize)
    #ymed = median_filter(yv[idx],winsize)
    #cax.plot(xmed,ymed,color='k')
    cax.set_xlim(xlim)
    cax.set_ylim(ylim)
    cax.set_xlabel(xlab)
    cax.set_ylabel(ylab)
    return cax

if __name__=='__main__':
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75] # 14.15
    fig, ((ax,ax1)) = plt.subplots(figsize=(doublecolsize[0],1.*doublecolsize[1]), nrows=1, ncols=2)
    ylimits = (-5,-1.3)
    xlimits = (10.9,14.6)
    annopos = (13.8,-1.75)

    ### ax -- ECO - RESOLVE-A (G3) ###
    ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)&(ecog3.resname=='notinresolve')]
    ecog3 = ecog3[ecog3.g3fc_l>0]
    xvalue = ecog3.g3logmh_l.to_numpy() 
    yvalue = np.log10(10**ecog3.g3grpmhi_l.to_numpy()/10**xvalue)
    ax=make_panel(ax,xvalue,yvalue,binvalues,xlimits,ylimits,None,"log group-integrated mass fraction", linelabel=r'$M_{\rm HI,\, grp}/M_{\rm vir}$ (ECO - RES-A)')

    yvalue_star = np.log10(10**ecog3.g3grplogS_l.to_numpy()/10**xvalue)
    ax=make_panel(ax,xvalue,yvalue_star,binvalues,None,None,None,None,linecolor='midnightblue',ptalpha=0,linelabel=r'$M_{*,\,\rm grp}/M_{\rm vir}$ (ECO - RES-A)')
    yvalue_bary = np.log10(10**ecog3.g3grplogB_l.to_numpy()/10**xvalue)
    ax=make_panel(ax,xvalue,yvalue_bary,binvalues,None,None,None,None,linecolor='saddlebrown',ptalpha=0,linelabel=r'$M_{\rm bary,\, grp}/M_{\rm vir}$ (ECO - RES-A)')
    ### ax -- RESOLVE (all) (G3) ###
    resg3 = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resg3 = resg3[(resg3.fl_insample==1)&(resg3.g3grpcz_l>4500)&(resg3.g3grpcz_l<7000)]
    resg3 = resg3[resg3.g3fc_l>0] 
    xvalue = resg3.g3logmh_l.to_numpy()
    yvalue = np.log10(10**resg3.g3grpmhi_l.to_numpy()/10**xvalue)
    make_panel(ax,xvalue,yvalue,binvalues,xlimits,ylimits,xlab=None,ylab=r"log group-integrated mass fraction",ptcolor='mediumorchid',linecolor='purple',ptalpha=0.9,linelabel=r'$M_{\rm HI,\, grp}/M_{\rm vir}$ (RESOLVE)')
    ax.legend(loc='lower left',ncol=1, fontsize=8.5, framealpha=1)
    ax.set_xlabel(r"$\log M_{\rm vir}$")
    ax.annotate("G3",xy=annopos,fontsize=16)

    ##########################
    # ax1 - ECO with RESOLVE-A
    ecofof = pd.read_csv("ECODR3_Jul0822.csv")
    ecofof = ecofof[(ecofof.absrmag<-17.33)&(ecofof.grpcz>3000)&(ecofof.grpcz<7000)&(ecofof.resname=='notinresolve')]
    ecofof = ecofof[ecofof.fc>0]
    xvalue = ecofof.logmhvir.to_numpy()
    yvalue = np.log10(10**ecofof.logmhigrp.to_numpy()/10**xvalue)
    ax1=make_panel(ax1,xvalue,yvalue,binvalues,xlimits,ylimits,None,"log group-integrated mass fraction", linelabel=r'$M_{\rm HI,\, grp}/M_{\rm vir}$ (ECO - RES-A)')

    yvalue_star = np.log10(10**ecofof.logmstargrp.to_numpy()/10**xvalue)
    ax1=make_panel(ax1,xvalue,yvalue_star,binvalues,None,None,None,None,linecolor='midnightblue',ptalpha=0,linelabel=r'$M_{*,\,\rm grp}/M_{\rm vir}$ (ECO - RES-A)')
    yvalue_bary = np.log10(10**ecofof.logmbarygrp.to_numpy()/10**xvalue)
    ax1=make_panel(ax1,xvalue,yvalue_bary,binvalues,None,None,None,None,linecolor='saddlebrown',ptalpha=0,linelabel=r'$M_{\rm bary,\, grp}/M_{\rm vir}$ (ECO - RES-A)')

    resfof = pd.read_csv("RESOLVEliving_071322_updatedfofgroups.csv")
    resfof = resfof[(resfof.fl_insample==1)&(resfof.grpcz>4500)&(resfof.grpcz<7000)]
    xvalue = resfof.logmhvir.to_numpy()
    yvalue = np.log10(10**resfof.logmhigrp.to_numpy()/10**xvalue)
    make_panel(ax1,xvalue,yvalue,binvalues,xlimits,ylimits,xlab=None,ylab=None,ptcolor='mediumorchid',linecolor='purple',ptalpha=0.9,linelabel=r'$M_{\rm HI,\, grp}/M_{\rm vir}$ (RESOLVE)')
    ax1.legend(loc='lower left',ncol=1, fontsize=8.5, framealpha=1)
    ax1.set_xlabel(r"$\log M_{\rm vir}$")
    ax1.annotate("FoF",xy=annopos,fontsize=16)
    plt.tight_layout()
    plt.savefig("../figures/MHI_over_Mvir.pdf",dpi=300)
    plt.show()
