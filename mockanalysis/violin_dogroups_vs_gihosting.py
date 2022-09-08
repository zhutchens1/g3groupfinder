import sys
sys.path.insert(0,'../g3algo/')
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
import numpy as np
from scipy.stats import mannwhitneyu as mwu, ks_2samp
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import matplotlib.patches as mpatches
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
cspeed=2.998e+5
H0=70.

import matplotlib
matplotlib.use('TkAgg')

if __name__=='__main__':
    mock = pd.read_hdf('../halobiasmocks/fiducial/ECO_cat_0_Planck_memb_cat.hdf5')
    mock=mock[mock.M_r<=-17.33]
    #mock = mock[(mock.M_r<=-17.33) & (mock.cz<7000) & (mock.cz>3000)]
    mock['halora'],mock['halodec'],mock['halocz']=fof.group_skycoords(np.array(mock.ra),\
        np.array(mock.dec), np.array(mock.cz), np.array(mock.haloid))    
    mock['rproj']=(1/H0)*(mock.cz+mock.halocz)*np.sin(fof.angular_separation(mock.ra,mock.dec,mock.halora,mock.halodec)/2.)
    mock.loc[:,'loghalom'] = mock.loghalom - np.log10(H0/100.)
    dwarfonly = mock.groupby('haloid').filter(lambda h: (h.M_r>-19.4).all())
    gianthost = mock.groupby('haloid').filter(lambda h: (h.M_r<=-19.4).any())
    print(len(mock),len(dwarfonly),len(gianthost))
    dwarfonly = dwarfonly[(dwarfonly.halo_ngal>1)]
    gianthost = gianthost[(gianthost.halo_ngal>1)]

    #fig, ax1 = plt.subplots(figsize=singlecolsize)
    #ax1.plot(gianthost.loghalom, gianthost.rproj, '.', color='darkgreen', label='Giant-Hosting Halo Galaxies', rasterized=True)
    #ax1.plot(dwarfonly.loghalom, dwarfonly.rproj, 'r.', markersize=4, label='Dwarf-Only Halo Galaxies', rasterized=True)
    plt.figure(figsize=(singlecolsize[0],singlecolsize[1]*1.2))
    ax1 = plt.gca()
    massbins=np.array([11.,11.33,11.66,12])-np.log10(H0/100.)
    db=(massbins[1]-massbins[0])
    vv=ax1.violinplot([np.array(gianthost.rproj[(gianthost.loghalom>massbins[ii])&(gianthost.loghalom<=massbins[ii+1])]) for ii in range(0,len(massbins)-1)], massbins[:-1]+db/2., showextrema=False,\
        widths=db)
    for pc in vv['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor('lightgreen')
    vv=ax1.violinplot([np.array(dwarfonly.rproj[(dwarfonly.loghalom>massbins[ii])&(dwarfonly.loghalom<=massbins[ii+1])]) for ii in range(0,len(massbins)-1)], massbins[:-1]+db/2., showextrema=False,\
        widths=db)
    for pc in vv['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor('lightcoral')
    ax1.scatter(massbins[:-1]+db/2.,[np.percentile(gianthost.rproj[(gianthost.loghalom>massbins[ii])&(gianthost.loghalom<=massbins[ii+1])],90) for ii in range(0,len(massbins)-1)],\
         marker='_', color='green',s=30)
    ax1.scatter(massbins[:-1]+db/2.,[np.percentile(dwarfonly.rproj[(dwarfonly.loghalom>massbins[ii])&(dwarfonly.loghalom<=massbins[ii+1])],90) for ii in range(0,len(massbins)-1)],\
         marker='_', color='red',s=30)
    ax1.scatter(massbins[:-1]+db/2.,[np.median(gianthost.rproj[(gianthost.loghalom>massbins[ii])&(gianthost.loghalom<=massbins[ii+1])]) for ii in range(0,len(massbins)-1)],\
         marker='D', color='green', s=15)
    ax1.scatter(massbins[:-1]+db/2.,[np.median(dwarfonly.rproj[(dwarfonly.loghalom>massbins[ii])&(dwarfonly.loghalom<=massbins[ii+1])]) for ii in range(0,len(massbins)-1)],\
         marker='D', color='red', s=15, facecolors='none')

    pvals = np.array([ks_2samp(gianthost.rproj[(gianthost.loghalom>massbins[ii])&(gianthost.loghalom<=massbins[ii+1])],dwarfonly.rproj[(dwarfonly.loghalom>massbins[ii])&(dwarfonly.loghalom<=massbins[ii+1])])[1] for ii in range(0,len(massbins)-1)])
    for ii,xpos in enumerate(massbins[:-1]+db/2):
        ax1.annotate(r"$p=$"+"{:0.1E}".format(pvals[ii]), (xpos-db/2.5,-0.02))
    print(pvals)
    ax1.set_xlim(11-np.log10(H0/100.),12-np.log10(H0/100.))
    ax1.set_xlabel(r"$\log M_{\rm vir}$")
    ax1.set_ylabel(r"$R_{\rm proj,\, gal}$")
    labels=['Giant-Hosting Halo Galaxies', 'Dwarf-Only Halo Galaxies']
    fake_handles = [mpatches.Patch(color='lightgreen'), mpatches.Patch(color='lightcoral')]
    ax1.legend(fake_handles,labels,loc='upper left',framealpha=0)
    ax1.set_xticks(massbins[:-1]+db/2, ["{:0.2f}".format(bb) for bb in massbins[:-1]+db/2])

    ax1.scatter(11.25,0.145,color='k',marker='D',s=15)
    ax1.scatter(11.25,0.12,color='k',marker='_',s=15)
    ax1.annotate('Median', xy=(11.325,0.14))
    ax1.annotate('90%', xy=(11.325,0.115))
    plt.tight_layout()
    plt.savefig("../figures/dwarfonlycalmotivation.pdf")
    plt.show() 
