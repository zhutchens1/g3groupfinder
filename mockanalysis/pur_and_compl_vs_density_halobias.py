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
    fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(doublecolsize[0],doublecolsize[1]*1.3))
    for ii in range(0,2):
        for jj in range(0,2):
            currentsd = subdirs[ii][jj]
            path='../halobiasgroupcats/'+currentsd
            files=os.listdir(path)
            dens=[]
            purity_g=[]
            comp_g=[]
            purity_h=[]
            comp_h=[]
            for fname in files:
                mock = pd.read_csv(path+fname)
                dens.append(len(mock)/192351./(0.7**3.))
                myhalongal = fof.multiplicity_function(mock.haloid.to_numpy(),True)
                mock=mock[mock.halo_ngal.to_numpy()==myhalongal]
                mock = mock[(mock.g3grp_l>0)&(mock.g3logmh_l>0)]
                print(currentsd,fname)

                mock = mock[mock.g3grpn_l>0]
                mock_by_group = mock.groupby('g3grp_l').first()
                comp_g.append(np.array(mock_by_group.group_comp))
                purity_g.append(np.array(mock_by_group.group_purity))
            
                mock_by_halo = mock.groupby('haloid').first()
                comp_h.append(mock_by_halo.halo_comp.to_numpy())
                purity_h.append(mock_by_halo.halo_purity.to_numpy())

            dens,purity_g,comp_g = zip(*sorted(zip(dens,purity_g,comp_g)))
            xpos = np.arange(len(dens))
            parts = axs[ii][jj].violinplot(purity_g, xpos, showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor("None")
                pc.set_facecolor("#8da0cb")
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in purity_g], marker='D', s=40, label=r'$\bar{P}_g$', edgecolor='#63708e', facecolor='None', zorder=99)
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in purity_h], marker='x', s=30, label=r'$\bar{P}_h$', color='k', zorder=99)
            #axs[ii][jj].scatter(xpos, [np.median(xx) for xx in purity_g], color='blue', marker='+', s=40, label=r'Median ${P}_g$',zorder=98)

 
            parts = axs[ii][jj].violinplot(comp_g, xpos, showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor("None")
                pc.set_edgecolor('#fc8d62')
                pc.set_alpha(0.7)
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in comp_g], marker='*', s=40, label=r'$\bar{C}_g$',edgecolor='#97553b',facecolor='None')
            axs[ii][jj].scatter(xpos, [np.mean(xx) for xx in comp_h], marker='+', s=40, label=r'$\bar{C}_h$',color='k')
            #axs[ii][jj].scatter(xpos, [np.median(xx) for xx in comp_g], marker='s', s=40, label=r'Median ${C}_g$', edgecolor='darkorange', facecolor="None")
            #axs[ii][jj].set_xticks(xpos,labels=["{:0.3f}".format(dd) for dd in dens],fontsize=8)
            axs[ii][jj].set_xticks(xpos,labels=["{a:d}".format(a=dd) for dd in xpos+1],fontsize=9)
            axs[ii][jj].set_yticks(np.arange(0.84,1.02,0.02))
            axs[ii][jj].set_ylim(0.84,1.0)
            axs[ii][jj].set_title(titles[ii][jj])
    axs[0][0].set_ylabel(" ")
    axs[1][1].set_xlabel(" ")
    axs[1][0].legend(loc='lower left', framealpha=1, ncol=2)
    fig.text(0.5, 0.03, r"Mock Catalog Number", ha='center')
    fig.text(0.03, 0.35, r"Purity or Completeness", ha='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig("../figures/prefacs_with_dens.pdf",dpi=300)
    plt.savefig("../figures/prefacs_with_dens.jpg",dpi=300)
    plt.show()

    print(dens)
