import sys
sys.path.insert(0,'../g3algo/')
import iterativecombination as ic
from center_binned_stats import center_binned_stats as cbs
from smoothedbootstrap import smoothedbootstrap as sbs
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from scipy.ndimage import median_filter
import foftools as fof
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
from scipy.optimize import curve_fit

def hihalomodel(Mh, M0, Mmin, alpha):
    return M0*((Mh/Mmin)**alpha)*np.exp(-1*Mmin/Mh)


plt.figure(figsize=(doublecolsize))
#plt.figure(figsize=(9,9))

# plot G3 data
#binvalues = [11,11.1,11.2,11.3,11.4,11.5,11.6,11.7,11.95,12.2,12.45,12.7,12.95,13.55,14.75]
binvalues = [11,11.15,11.3,11.45,11.6,11.75,12.,12.25,12.5,12.75,12.95,13.3,14.75]
ecog3 = pd.read_csv("../resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
ecog3 = ecog3[(ecog3.absrmag<-17.33)&(ecog3.g3grpcz_l>3000)&(ecog3.g3grpcz_l<7000)] # add RES-B?
ecog3 = ecog3[ecog3.g3fc_l>0]
xv = ecog3.g3logmh_l.to_numpy()
yv = ecog3.g3grpmhi_l.to_numpy()
median,bc,binedges,_ = cbs(xv, yv, 'median', bins=binvalues)
pt25,bc,binedges,_ = cbs(xv, yv, lambda x:np.percentile(x,25), bins=binvalues)
pt75,bc,binedges,_ = cbs(xv, yv, lambda x:np.percentile(x,75), bins=binvalues)
plt.scatter(bc,median, marker='o', color='k', label=r'G3 Groups (Median $\pm$ 25%)', zorder=99)
plt.fill_between(bc,pt25,pt75,color='gray',alpha=0.3)
#plt.errorbar(bc,median,yerr=(pt75-pt25)/2.,marker='o',color='k',label=r'G3 Groups (Median $\pm$ 25%)', zorder=99)

# fit G3 data
popt, pcov = curve_fit(hihalomodel, 10**bc, 10**median, p0=[1.689e9, 1.2698e11, 3.686e-1], maxfev=2000)
err = np.sqrt(np.diagonal(pcov))
print('Best-fit params.: ', np.log10(popt[0]), np.log10(popt[1]), popt[2])
print('Errors: ', np.log10(err[0]), np.log10(err[1]), err[2])
logtx = np.linspace(11,14.5,1000)
mhihat = hihalomodel(10**logtx, *popt)
plt.plot(logtx, np.log10(mhihat), color='k', linewidth=2, alpha=1, label='G3 Groups (Fit to Medians)', zorder=98)

# show fit from obuljen et al 2019
obuljen_alpha = 0.48
obuljen_Mmin = 10**(11.18) / 0.7 / fof.getmhoffset(180,337,1,1,6)
obuljen_M0 = 10**9.44 / 0.7 / fof.getmhoffset(180,337,1,1,6)
obuljen_params = (obuljen_M0, obuljen_Mmin, obuljen_alpha) # FIX!! their delta=180???
plt.plot(logtx, np.log10(hihalomodel(10**logtx,*obuljen_params)), color='red', label='Obuljen+2019 (ALFALFA)')

# show guo et al 2020
guofile = open("TableA1Guo2020.txt", 'r')
guohalomass = []
guohimass = []
guohimasserr = []
guobins = []
for line in guofile:
    if line.startswith('['):
        split = line.split()
        binstart = float(split[0][1:-1])
        binend = float(split[1][:-1])
        guobins.append(binstart)
        guohalomass.append((binstart+binend)/2.)
        guohimass.append(float(split[3]))
        guohimasserr.append(float(split[5]))
guobins.append(binend)
guohalomass = np.log10(10**np.array(guohalomass)/fof.getmhoffset(200,337,1,1,6))
plt.plot(guohalomass, guohimass, color='green', label='Guo+2020 (ALFALFA)',linewidth=2)

# show zhang 22
zhang = pd.read_csv("Zhang22xGASS.csv")
zhang.loc[:,'Mhalo']=np.log10(10**zhang.Mhalo / fof.getmhoffset(200,337,1,1,6))
plt.plot(zhang.Mhalo,zhang.Mhi, color='orange', linestyle='dashed', label='Zhang+2022 (xGASS)')

# show chauhan et al 2020
c20data = pd.read_csv("Chauhan2020MedianTotalGroupHI.csv")
c20_logm337 = np.log10(10**c20data.logm200 / fof.getmhoffset(200,337,1,1,6))
c20_himass = np.array(c20data.mhi)
plt.plot(c20_logm337, c20_himass, color='blue', linestyle='-.', label='Chauhan+2020 (SAM)')

# show callette 2021
c21data = pd.read_csv("Callette2021relation.csv") # use Delta=333
plt.plot(c21data.logDM, c21data.logMHItot, color='purple', linestyle='dotted', label="Calette+2021 (Sim)")

plt.xlabel(r'$\log M_{\rm halo}$ [$\rm M_\odot$]')
plt.ylabel(r'$\log M_{\rm HI,\, grp}$ [$\rm M_\odot$]')
plt.legend(loc='best')
plt.xlim(11,14.5)
plt.ylim(8.4,11.5)
plt.show()

