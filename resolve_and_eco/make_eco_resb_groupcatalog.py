import sys
sys.path.insert(0,'../g3algo/')
from g3groupfinder import g3groupfinder_luminosity as g3gf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

if __name__=='__main__':
    eco = pd.read_csv('ECOdata_080822.csv')
    eco1733 = eco[eco.absrmag<-17.33]
    ecog3grpid, _, meansep, rp1, _, vp1, _, rp2, _, vp2, _  = g3gf(eco.radeg,eco.dedeg,eco.cz,eco.absrmag,-19.4,volume=192351.,rproj_fit_multiplier=3.5,vproj_fit_multiplier=6.5,\
                    gd_rproj_fit_multiplier=1., gd_vproj_fit_multiplier=2.5, gd_fit_bins=np.arange(-24,-19,0.25), gd_rproj_fit_guess=[1e-5, 0.4], gd_vproj_fit_guess=[3e-5,4e-1])
    print(meansep)
    print(rp1)
    print(vp1)
    print(rp2)
    print(vp2)
