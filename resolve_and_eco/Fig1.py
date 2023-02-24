import pandas as pd
import numpy as np
from seaborn import kdeplot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogFormatter
from matplotlib import rcParams
from matplotlib.colors import LogNorm
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
    resolve = pd.read_csv("RESOLVEdata_080822.csv")
    resolve = resolve[(resolve.grpcz>4500)&(resolve.grpcz<7000)&(resolve.fl_insample==1)]
    resolve.loc[:,'logmbary'] = np.log10(10**resolve.logmstar + 10**resolve.logmgas)

    fig,axs = plt.subplots(ncols=2,figsize=(doublecolsize[0],0.7*doublecolsize[1]))
    axs[0].scatter(resolve.absrmag, resolve.logmstar,s=1,alpha=0.5)
    kdeplot(resolve.absrmag, resolve.logmstar, ax=axs[0], color='k', levels=5)
    axs[0].axhline(9.7, color='red', label=r'$\log M_* = 9.7$')
    axs[0].axvline(-19.5, color='red', linestyle='--', label=r'$M_r = -19.5$')
    axs[0].invert_xaxis()
    axs[0].set_ylabel(r"log stellar mass [M$_\odot$]")
    axs[0].set_xlabel("r-band absolute magnitude")
    axs[0].legend(loc='best')
    axs[0].set_ylim(7.9,11.5)

    axs[1].scatter(resolve.absrmag, resolve.logmbary, s=1, alpha=0.5)
    kdeplot(resolve.absrmag, resolve.logmbary, ax=axs[1], color='k', levels=5)
    axs[1].axhline(9.9, color='red', label=r'$\log M_{\rm bary} = 9.9$')
    axs[1].axvline(-19.5, color='red', linestyle='--', label=r'$M_r = -19.5$')
    axs[1].invert_xaxis()
    axs[1].set_ylabel(r"log baryonic mass [M$_\odot$]")
    axs[1].set_xlabel("r-band absolute magnitude")
    axs[1].legend(loc='best')
    axs[1].set_ylim(7.9,11.5)
    plt.tight_layout()
    plt.savefig("../figures/dwarfgiantdivide_paper1.pdf")
    plt.show()

    
    plt.figure()
    sel = (resolve.logmbary>9.8)&(resolve.logmbary<10.0)
    plt.hist(resolve.absrmag[sel], bins=10)
    plt.axvline(np.median(resolve.absrmag[sel]), color='k',label='Median')
    print(np.median(resolve.absrmag[sel]))
    plt.xlabel("absolute magnitude for 9.8 < logMbary < 10")
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.show()
