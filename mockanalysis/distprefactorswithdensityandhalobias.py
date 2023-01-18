import sys
sys.path.insert(0,'../g3algo/')
import foftools as fof
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator, LogFormatter
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from sklearn.preprocessing import normalize
from scipy.stats import median_abs_deviation as mad
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

if __name__=='__main__':
    subdirs = np.array([['fiducial/', 'central/'], ['dv0_8/', 'dv1_2/']],dtype='object')
    titles = [['Fiducial', 'Central-Offset'], [r'$b_v=0.8$', r'$b_v = 1.2$']]
    fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(doublecolsize[0],doublecolsize[1]*1.1))
    for ii in range(0,2):
        for jj in range(0,2):
            currentsd = subdirs[ii][jj]
            path='../halobiasgroupcats/'+currentsd
            files=os.listdir(path)
            dens=[]
            purity=[]
            comp=[]
            l1norm=[]
            scale=[]
            for fname in files:
                mock = pd.read_csv(path+fname)
                dens.append(len(mock)/192351./(0.7**3.))
                myhalongal = fof.multiplicity_function(mock.haloid.to_numpy(),True)
                mock=mock[mock.halo_ngal.to_numpy()==myhalongal]
                mock = mock[(mock.g3grp_l>0)&(mock.g3logmh_l>0)]
                dm = np.abs((mock.g3logmh_l+np.log10(0.7))-mock.loghalom)
                print(currentsd,fname)
                scale.append(mad(dm))

                mock = mock[mock.g3grpn_l>1]
                mock = mock.groupby('g3grp_l').first()
                comp.append(np.array(mock.group_comp))
                purity.append(np.array(mock.group_purity))

            dens,purity,comp,scale = zip(*sorted(zip(dens,purity,comp,scale)))
            xpos = np.arange(len(dens))
            parts = axs[ii][jj].violinplot(purity, xpos, showextrema=False, widths=1)
            for pc in parts['bodies']:
                pc.set_facecolor("None")
                pc.set_facecolor("blue")
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in purity], marker='*', s=60, label=r'Mean ${P}_g$', edgecolor='blue', facecolor='None', zorder=99)
            axs[ii][jj].scatter(xpos, [np.median(xx) for xx in purity], color='blue', marker='+', s=40, label=r'Median ${P}_g$',zorder=98)

 
            parts = axs[ii][jj].violinplot(comp, xpos, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor("None")
                pc.set_edgecolor('darkorange')
                pc.set_alpha(0.7)
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in comp], marker='.', s=40, label=r'Mean ${C}_g$',color='darkorange')
            axs[ii][jj].scatter(xpos, [np.median(xx) for xx in comp], marker='s', s=40, label=r'Median ${C}_g$', edgecolor='darkorange', facecolor="None")
            axs[ii][jj].set_xticks(xpos,labels=["{:0.3f}".format(dd) for dd in dens],fontsize=8)
            axs[ii][jj].set_ylim(0.4,1.02)
            #axs[ii][jj].set_xticklabels(["{:0.3f}".format(dd) for dd in dens])
            #dens,meanpurity,meancomp,l1norm,scale,medianpurity,mediancomp = zip(*sorted(zip(dens,meanpurity,meancomp,l1norm,scale,medianpurity,mediancomp)))
            #axs[ii][jj].axvline(12700/192351.,["{:0.3f}".format(dd) for dd in dens] color='k', alpha=0.6)
            #axs[ii][jj].scatter(dens,meanpurity, marker='*', s=60, label=r'$\bar{P}$', edgecolor='blue', facecolor='None')
            #axs[ii][jj].scatter(dens,medianpurity, color='blue', marker='+', s=40, label=r'$\bar{P}$')
            #axs[ii][jj].scatter(dens,meancomp,marker='.', s=20, label=r'$\bar{C}$',color='orange')
            #axs[ii][jj].scatter(dens,mediancomp,marker='s', s=40, label=r'Median ${C}$', edgecolor='orange', facecolor="None")
            #axs[ii][jj].plot(dens, np.array(l1norm)/(192351.*np.array(dens)), '-o', markersize=2, label=r'$\Sigma_{\rm HME}$/$N_{\rm mock}$')
            #axs[ii][jj].plot(dens, np.array(scale), '-o', markersize=2, label=r'$\sigma_{\rm HME}$')
            axs[ii][jj].set_title(titles[ii][jj])
            #for lp in np.arange(0,1.5,0.1):
            #    axs[ii][jj].axhline(lp,color='k',zorder=0,alpha=0.2)
            #axs[ii][jj].set_ylim(0.,1.05)
            #axs[ii][jj].grid(which='both', axis='x')
    axs[1][0].legend(loc='lower left', framealpha=1, ncol=2)
    fig.text(0.5, 0.003, r"Galaxy Number Density [${\rm Mpc}^{-3}$]", ha='center')
    fig.text(0.01, 0.35, r"Purity or Completeness", ha='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig("../figures/prefacs_with_dens.pdf",dpi=300)
    plt.savefig("../figures/prefacs_with_dens.jpg",dpi=300)
    plt.show()
