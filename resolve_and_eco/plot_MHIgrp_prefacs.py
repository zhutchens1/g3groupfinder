import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.500005949910059, 4.3880449973709)
import os
from center_binned_stats import center_binned_stats as cbs

def plot_lines_gh(cax,direc, binv):
    files = os.listdir(direc)
    files.sort()
    files=[files[1],files[2],files[3],files[4],files[0]]
    points_per_group = len(files)
    group_xsize = 0.05
    groupxoffsets = np.linspace(-1*group_xsize/2., group_xsize/2., points_per_group)
    for ii,ff in enumerate(files):
        eco=pd.read_csv(direc+ff)
        eco=eco[(eco.absrmag<-17.33)]
        median,bc,_,_ = cbs(eco.g3logmh_l.to_numpy(),eco.g3grpmhi_l.to_numpy(),'median',bins=binv)
        parts = ff.split('__')
        rprojmult = parts[1]
        vprojmult = parts[2]
        vprojoffs = parts[3]
        label=rprojmult+r'$R_{\rm proj}^{\rm fit}$ ; '+vprojmult+r'$\Delta v_{\rm proj}^{\rm fit} + $'+vprojoffs+r' $\rm km\, s^{-1}$'
        cax.plot(bc+groupxoffsets[ii],median,'.',label=label, markersize=3)
    return None

def plot_lines_do(cax,direc, binv):
    files = os.listdir(direc)
    files.sort()
    files=[files[2],files[0],files[3],files[4],files[1]]
    points_per_group = len(files)
    group_xsize = 0.05
    groupxoffsets = np.linspace(-1*group_xsize/2., group_xsize/2., points_per_group)
    for ii,ff in enumerate(files):
        eco=pd.read_csv(direc+ff)
        eco=eco[(eco.absrmag<-17.33)]
        eco=eco.groupby('g3grp_l').filter(lambda g: (g.absrmag>-19.5).all())
        median,bc,_,_ = cbs(eco.g3logmh_l.to_numpy(),eco.g3grpmhi_l.to_numpy(),'median',bins=binv)
        parts = ff.split('__')
        print(parts)
        rprojmult = parts[1]
        vprojmult = parts[2]
        label=rprojmult+r'$R_{\rm proj,\,fit}^{\rm gi,\, dw}$ ; '+vprojmult+r'$\Delta v_{\rm proj,\,fit}^{\rm gi,\, dw}$'
        cax.plot(bc+groupxoffsets[ii],median,'*',label=label,markerfacecolor='None', markersize=6)
    return None

if __name__=='__main__':
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75]
    fig, ax=plt.subplots(figsize=(doublecolsize[0],1*doublecolsize[1]))
    plot_lines_gh(ax,'ad_prefac_eco_catls/',binvalues)
    plot_lines_do(ax,'ic_prefac_eco_catls/',binvalues)
    ax.set_xlabel(r"$\log M_{\rm vir}$ [$\rm M_\odot$]")
    ax.set_ylabel(r"$\log M_{\rm HI,\,grp}$ [$\rm M_\odot$]")
    ax.legend(loc='best',ncol=2,fontsize=8,framealpha=1)
    ax.set_ylim(8.5,11)
    plt.grid()
    plt.tight_layout()
    plt.savefig("../figures/MHIrelation_gfparams.pdf",dpi=300)
    plt.show()

