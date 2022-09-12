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
from scipy.stats import spearmanr

def srt_by_bin(xv, yv, mh, mhbinedges):
    mhbinedges=np.array(mhbinedges)
    mhleftedges=mhbinedges[:-1] 
    mhrightedges=mhbinedges[1:]
    for ii,(ll,rr) in enumerate(zip(mhleftedges,mhrightedges)):
        sel = (mh>=ll)&(mh<=rr)
        print('Bin for ', ll, ' to ', rr, ': ', spearmanr(xv[sel],yv[sel]))

def make_panel(cax, xv, yv, mh, mhbinedges, labels, xbinedges=[-2,-1,0,1,2]):
    mhbinedges=np.array(mhbinedges)
    mhleftedges=mhbinedges[:-1] 
    mhrightedges=mhbinedges[1:]
    colors=['tomato','midnightblue','purple','green','mediumorchid']
    for ii,(ll,rr) in enumerate(zip(mhleftedges,mhrightedges)):
        sel = (mh>=ll)&(mh<=rr)
        #cax.scatter(xv[sel],yv[sel],s=2)
        median,bc,_,_ = cbs(xv[sel],yv[sel],'median',bins=xbinedges) 
        pt25,bc,_,_ = cbs(xv[sel],yv[sel],lambda x:np.percentile(x,25),bins=xbinedges) 
        pt75,bc,_,_ = cbs(xv[sel],yv[sel],lambda x:np.percentile(x,75),bins=xbinedges)
        cax.plot(bc,median,'.-',linewidth=2,color=colors[ii],markersize=10, label=labels[ii])
        if ii%2==0:
            cax.fill_between(bc, pt25, pt75, alpha=0.5, color=colors[ii])
        else:
            #cax.fill_between(bc, pt25, pt75, alpha=0.25, color=colors[ii])
            cax.plot(bc,pt25,'--', color=colors[ii])
            cax.plot(bc,pt75,'--', color=colors[ii])

if __name__=='__main__':
    ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)]
    resg3 = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")
    resg3 = resg3[(resg3.fl_insample==1)&(resg3.g3grpcz_l>3000)&(resg3.g3grpcz_l<7000)&(resg3.f_b==1)]
    g3 = pd.concat([ecog3,resg3])
    g3 = g3[g3.g3fc_l>0]
    xx = (10**g3.g3grpnndens2d_l/g3.g3grpedgescale2d_l)
    g3.loc[:,'lsdens'] = np.log10(xx/np.median(xx)) 
    
    fig, (ax,ax1) = plt.subplots(ncols=2, figsize=(doublecolsize[0],doublecolsize[1]*0.8))
    binlabels = [r'$\log M_{\rm vir}\leq 11.4$', r'$11 < \log M_{\rm vir} \leq 12$', r'$12 < \log M_{\rm vir} \leq 13$', r'$\log M_{\rm vir} > 13$']
    ### ax (LS dens)
    xval = g3.lsdens.to_numpy()
    yval = np.log10(10**g3.g3grpmhi_l.to_numpy() / 10**g3.g3logmh_l.to_numpy())
    zval = g3.g3logmh_l.to_numpy()
    make_panel(ax, xval, yval, zval, [11,11.4,12,13,15], labels=binlabels)
    ax.set_xlabel(r"$\log \, \rho_{\rm LS}$")
    ax.set_ylabel(r"$\log \, \left(M_{\rm HI,\, grp} / M_{\rm vir}\right)$")
    ax.legend(loc='lower left', ncol=1, framealpha=1)
    #ax.set_ylim(7.9,10.6)
    print('LS density spearman-rank results:')
    srt_by_bin(xval,yval,zval,[11,11.4,12,13,15]) 

    ## ax1 (tcross)
    g3 = g3[g3.g3grptcross_l>0.0]
    xval = np.log10(g3.g3grptcross_l.to_numpy())
    yval = np.log10(10**g3.g3grpmhi_l.to_numpy() / 10**g3.g3logmh_l.to_numpy())
    zval = g3.g3logmh_l.to_numpy()
    make_panel(ax1, xval, yval, zval, [11,11.4,12,13,15], labels=binlabels, xbinedges=np.log10(np.array([0.001, 5, 13 , 40])))
    ax1.set_xlabel(r"log group $t_{\rm cross}$ [Gyr]")#\, \rho_{\rm LS}$")
    #ax1.set_ylabel(r"$\log \, M_{\rm vir}$ [$\rm M_\odot$]")
    print('tcross spearman-rank results:')
    srt_by_bin(xval,yval,zval,[11,11.4,12,13,15]) 

    print(np.median(g3.g3grptcross_l.to_numpy()))
    plt.tight_layout()
    plt.savefig("../figures/groupmhi_lss_tcross.pdf",dpi=300)
    plt.show()

        
     
