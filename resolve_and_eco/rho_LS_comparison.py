import sys
sys.path.insert(0,'../g3algo/')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lss_dens import lss_dens_by_galaxy
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from seaborn import kdeplot
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
    ecog3 = pd.read_csv("ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[ecog3.g3grp_l>0]

    ecog3nndens, ecog3edgeflag, ecog3nndens2d, ecog3edgeflag2d, ecog3edgescale2d = lss_dens_by_galaxy(ecog3.g3grp_l,\
        ecog3.radeg, ecog3.dedeg, ecog3.cz, ecog3.g3logmh_l, Nnn=3, rarange=(130.05,237.45), decrange=(-1,50), czrange=(2530,7470))
    ecog3.loc[:,'g3grpnndens2d_l']=ecog3nndens2d
    ecog3.loc[:,'g3grpedgescale2d_l']=ecog3edgescale2d
    ecog3 = ecog3.groupby('g3grp_l').first()

    g3dens = 10**ecog3.g3grpnndens2d_l/ecog3.g3grpedgescale2d_l
    g3dens = np.log10(g3dens/np.median(g3dens))
    g3masses = ecog3.g3logmh_l 
    

    dr2 = pd.read_csv("ECODR2.csv")
    dr2 = dr2[dr2.grp_e17>0]
    dr2nndens, dr2edgeflag, dr2nndens2d, dr2edgeflag2d, dr2edgescale2d = lss_dens_by_galaxy(dr2.grp_e17,\
        dr2.radeg, dr2.dedeg, dr2.cz, dr2.logmh_e17, Nnn=3, rarange=(130.05,237.45), decrange=(-1,50), czrange=(2530,7470))
    dr2.loc[:,'dr2grpnndens2d_l']=dr2nndens2d
    dr2.loc[:,'dr2grpedgescale2d_l']=dr2edgescale2d
    dr2 = dr2.groupby('grp_e17').first()

    dr2dens = 10**dr2.dr2grpnndens2d_l/dr2.dr2grpedgescale2d_l
    dr2dens = np.log10(dr2dens/np.median(dr2dens))
    dr2masses = dr2.logmh_e17


    # make plot
    fig,(ax,ax1)=plt.subplots(ncols=2,figsize=(doublecolsize[0],doublecolsize[1]*0.8),sharey=True)
    kdeplot(g3masses,g3dens,ax=ax,color='orange')
    ax.scatter(g3masses,g3dens,color='k',s=1)
    ax.set_xlabel(r"$\log M_{\rm vir}$")
    ax.set_ylabel(r"$\log \rho_{\rm LS}$")
    ax.annotate("G3", xy=(13,-2.5),fontsize=14)

    kdeplot(dr2masses,dr2dens,ax=ax1,color='orange')
    ax1.scatter(dr2masses,dr2dens,color='k',s=1)
    ax1.set_xlabel(r"$\log M_{\rm vir}$")
    ax1.annotate("FoF", xy=(13,-2.5), fontsize=14)
    plt.tight_layout()
    plt.savefig("../figures/rho_LS_comp.pdf",dpi=300)
    plt.show()
