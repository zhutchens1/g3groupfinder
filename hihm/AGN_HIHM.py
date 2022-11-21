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
#from seaborn import kdeplot
from scipy.stats import ks_2samp
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

def find_agn_in_group(f_agn, grp):
    """
    Mark groups as containing AGN or not (1/0).
    f_agn: 1/0 galaxy agn flag
    grp: group ID for each galaxy
    returns group AGN flag with same length
    """
    outflag=np.zeros_like(grp)-999.
    for ii,gg in enumerate(np.unique(grp)):
        grpsel = np.where(grp==gg)
        f_agn_in_group = f_agn[grpsel]
        if (f_agn_in_group>0).any():
            outflag[grpsel]=np.sum(f_agn_in_group[f_agn_in_group>0])/len(grpsel[0])
        else:
            outflag[grpsel]=0
    return outflag

if __name__=='__main__':
    AGNfile = pd.read_csv("/srv/two/cielo/mugpol/res_eco_data/ECO_AGN_inventory.csv")
    AGNfile=AGNfile.set_index('name')
    eco = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
    eco=eco.set_index('name')
    eco = eco.join(other=AGNfile,rsuffix='_2')

    agnflag = (eco.jhu_sf.to_numpy()==0) & (eco.jhu_inbptsample.to_numpy()==1)
    print(len(agnflag[agnflag>0]))
    eco.loc[:,'agnflag']=agnflag

    grpagnflag = find_agn_in_group(agnflag, eco.g3grp_l.to_numpy())
    eco.loc[:,'grpagnflag']=grpagnflag

    plt.style.use('dark_background')
    eco=eco[(eco.g3grp_l>0)&(eco.g3fc_l>0)]
    plt.figure()
    sel=(eco.grpagnflag>0)
    sc=plt.scatter(eco.g3logmh_l[sel], np.log10(10**eco.g3grplogG_l[sel]/10**eco.g3logmh_l[sel]), c=eco.grpagnflag[sel], s=2, cmap='seismic_r')
    plt.colorbar(sc)
    plt.show()

    plt.figure()
    sel=(eco.agnflag>-999)
    sc=plt.scatter(eco.g3logmh_l[sel], np.log10(10**eco.g3grplogG_l[sel]/10**eco.g3logmh_l[sel]), c=eco.agnflag[sel], s=2, cmap='seismic_r')
    plt.colorbar(sc)
    plt.show()
