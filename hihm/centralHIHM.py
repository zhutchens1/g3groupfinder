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

def getstats(x,y,bins):
    median,bc,_,_ = cbs(x,y,'median',bins=bins)
    pt16,bc,_,_ = cbs(x,y,lambda j: np.percentile(j,16),bins=bins)
    pt84,bc,_,_ = cbs(x,y,lambda j: np.percentile(j,84),bins=bins)
    return bc, pt16, median, pt84

if __name__=='__main__':
    xlimits=(10.95,14.6)
    ylimits = (7.5,10.5)
    binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75]
    ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)&(ecog3.g3fc_l>0)]
    resg3 = pd.read_csv("../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv")    
    resg3 = resg3[(resg3.fl_insample==1)&(resg3.g3grpcz_l>3000)&(resg3.g3grpcz_l<7000)&(resg3.g3fc_l>0)&(resg3.f_b==1)]
    resbg3logmhalo = resg3.g3logmh_l.to_numpy()
    resbg3logmhi = np.log10(10**resg3.logmgas.to_numpy() / 1.4)
    ecog3logmhalo = ecog3.g3logmh_l.to_numpy()
    ecog3logmhi = np.log10(10**ecog3.logmgas.to_numpy() / 1.4)
    resburcolor = resg3.modelu_rcorr.to_numpy()
    ecourcolor = ecog3.modelu_rcorr.to_numpy()
    logmhalo = np.concatenate([resbg3logmhalo, ecog3logmhalo])
    logmhi = np.concatenate([resbg3logmhi, ecog3logmhi])
    urcolor = np.concatenate([resburcolor,ecourcolor])
    morphel = np.concatenate([resg3.morphel.to_numpy(), ecog3.morphel.to_numpy()]) 

    fig, ax = plt.subplots(ncols=1, figsize=(singlecolsize[0],1.2*singlecolsize[1]))

    latesel = np.where(morphel=='L')
    bc, latept16, latemedian, latept84 = getstats(logmhalo[latesel],logmhi[latesel],binvalues)
    ax.plot(bc, latemedian, color='blue', label='Late-Type Centrals')
    ax.fill_between(bc, latept16, latept84, color='cornflowerblue', alpha=0.7)
    
    earlysel = np.where(morphel=='E')
    bc, earlypt16, earlymedian, earlypt84 = getstats(logmhalo[earlysel],logmhi[earlysel],binvalues)
    ax.plot(bc, earlymedian, color='red', label='Early-Type Centrals')
    ax.fill_between(bc, earlypt16, earlypt84, color='tomato', alpha=0.3)

    bc, _, median, _ = getstats(logmhalo, logmhi, binvalues)
    plt.plot(bc,median,color='k',label='All Centrals')
    pickle.dump([bc,median],open('cenhi.pkl','wb'))    

    ax.set_xlabel(r"log group $M_{\rm vir}$ [$\rm M_\odot$]")
    ax.set_ylabel(r"log central galaxy HI mass [$\rm M_\odot$]")
    ax.legend(loc='lower right')
    ax.set_xlim(np.min(bc), np.max(bc))
    ax.set_ylim(*ylimits)

    ################################
    ################################
    #cm = plt.cm.get_cmap('RdYlBu_r')    
    #sc=ax1.scatter(logmhalo[latesel], logmhi[latesel], c=urcolor[latesel], marker='.', cmap=cm, rasterized=True)
    #sc=ax1.scatter(logmhalo[earlysel], logmhi[earlysel], c=urcolor[earlysel], marker='*', cmap=cm, rasterized=True, edgecolor='k')

    #clb = plt.colorbar(sc,label='x')
    #.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../figures/centralHIHM.pdf',dpi=300)
    plt.show()
