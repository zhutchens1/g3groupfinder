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
from scipy.optimize import curve_fit

def linmodel(x,a,b):
    return a*x+b

def make_panel(cax,xv,yv,binvalues,xlim,ylim,xlab,ylab,ptcolor='gray',linecolor='k',ptalpha=0.3,fit=False):
    cax.scatter(xv,yv,color=ptcolor,alpha=ptalpha,s=2)
    median,bc,binedges,_ = cbs(xv, yv, 'median', bins=binvalues)
    pt16,bc,binedges,_=cbs(xv,yv,lambda x:np.percentile(x,16), bins=binvalues)
    pt84,bc,binedges,_=cbs(xv,yv,lambda x:np.percentile(x,84), bins=binvalues)
    #medianerr = np.std(np.array([sbs(yv[np.where(np.logical_and(xv>binedges[i-1], xv<=binedges[i]))],\
    #                   50, np.median, kwargs=dict({'axis':1})) for i in range(1,len(binedges))]), axis=1)
    #cax.errorbar(bc,median,yerr=medianerr,fmt='o-',color=linecolor,rasterized=True,markersize=2)
    cax.errorbar(bc,median,yerr=[pt84-median,median-pt16],fmt='o-',color=linecolor,rasterized=True,markersize=2)
    #cax.scatter(bc,median,marker='o',rasterized=True,s=2)
    #cax.fill_between(bc,pt16,pt84,alpha=0.2)
    #idx = np.argsort(xv)
    #xmed = median_filter(xv[idx],winsize)
    #ymed = median_filter(yv[idx],winsize)
    #cax.plot(xmed,ymed,color='k')
    cax.set_xlim(*xlim)
    cax.set_ylim(*ylim)
    cax.set_xlabel(xlab)
    cax.set_ylabel(ylab)
    if fit:
        print('Linear fits to relation across thresh. scale:')
        sel = (bc<=11.5)
        popt,pcov = curve_fit(linmodel, bc[sel], median[sel])
        perr = np.sqrt(np.diagonal(pcov))
        print('below 11.5 (param,err): ', popt, perr)
        popt,pcov = curve_fit(linmodel, bc[~sel], median[~sel])
        perr = np.sqrt(np.diagonal(pcov))
        print('above 11.5 (param,err): ', popt, perr)
    return cax

if __name__=='__main__':
    #binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75] # 14.15
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,14.75] # 14.15
    fig, ((ax,ax1)) = plt.subplots(figsize=(doublecolsize[0],1.*doublecolsize[1]), nrows=1, ncols=2)
    ylimits = (7,11.25)
    xlimits = (10.9,14.6)
    annopos = (11.5,7.4)

    ### ax -- ECO - RESOLVE-A (G3) ###
    print("ECO - RESOLVE-A (G3)")
    ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)&(ecog3.resname=='notinresolve')]
    ecog3 = ecog3[ecog3.g3fc_l>0]
    xvalue = ecog3.g3logmh_l.to_numpy() 
    yvalue = ecog3.g3grpmhi_l.to_numpy()
    ax=make_panel(ax,xvalue,yvalue,binvalues,xlimits,ylimits,None,"log group-integrated HI mass",fit=True)
    ax.annotate("ECO - RESOLVE-A: G3 Groups",xy=annopos)


    ### ax -- RESOLVE (all) (G3) ###
    resg3 = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resg3 = resg3[(resg3.fl_insample==1)&(resg3.g3grpcz_l>4500)&(resg3.g3grpcz_l<7000)]
    resg3 = resg3[resg3.g3fc_l>0] 
    yvalue = resg3.g3grpmhi_l.to_numpy()
    xvalue = resg3.g3logmh_l.to_numpy()
    make_panel(ax,xvalue,yvalue,binvalues,xlimits,ylimits,r"log group $M_{\rm halo}$ [$\rm M_\odot$]",r"log group-integrated HI mass [$\rm M_{\odot}$]",'mediumorchid','purple',0.9)
    ax.annotate("RESOLVE: G3 Groups", xy=(annopos[0],annopos[1]-0.2), color='purple') 

    ##########################
    # ax1 - ECO -  RESOLVE-A
    print('ECO - RESOLVE-A (FOF)')
    ecofof = pd.read_csv("ECODR3_Jul0822.csv")
    ecofof = ecofof[(ecofof.absrmag<-17.33)&(ecofof.grpcz>3000)&(ecofof.grpcz<7000)&(ecofof.resname=='notinresolve')]
    #ecofof.loc[:,'mhigrp']=ic.get_int_mass(np.log10(10.**ecofof.logmgas/1.4),ecofof.grp)
    ecofof = ecofof[ecofof.fc>0]
    xvalue = ecofof.logmhvir.to_numpy()
    yvalue = ecofof.logmhigrp.to_numpy()
    make_panel(ax1,xvalue,yvalue,binvalues,xlimits,ylimits,None,"log group-integrated HI mass", fit=True)
    ax1.annotate("ECO - RESOLVE-A: FoF Groups",xy=annopos)

    resfof = pd.read_csv("RESOLVEliving_071322_updatedfofgroups.csv")
    resfof = resfof[(resfof.fl_insample==1)&(resfof.grpcz>4500)&(resfof.grpcz<7000)]
    xvalue = resfof.logmhvir.to_numpy()
    yvalue = resfof.logmhigrp.to_numpy()
    make_panel(ax1,xvalue,yvalue,binvalues,xlimits,ylimits,r"log group $M_{\rm halo}$ [$\rm M_\odot$]",None,'mediumorchid','purple',0.9)
    ax1.annotate("RESOLVE: FoF Groups", xy=(annopos[0],annopos[1]-0.2), color='purple')
    plt.tight_layout()
    plt.savefig("../figures/HIHM.pdf", dpi=300)
    plt.savefig("../MSdefensefigs/figs/hihm.png",dpi=1000)
    plt.show()
