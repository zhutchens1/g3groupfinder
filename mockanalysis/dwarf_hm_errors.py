import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from center_binned_stats import center_binned_stats
from scipy.stats import gaussian_kde, median_abs_deviation as mad
from smoothedbootstrap import smoothedbootstrap as sbs
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('TkAgg')
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.100005949910059, 4.3880449973709)



# Program to get error bars on G3 group g3logmh_l
# by comparing to true halo masses and looking at
# the 1-sigma dispersion for all groups in all
# 32 catalogs

if __name__=='__main__':
    subdirs = ['fiducial/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock['g3logmh_l'] = mock['g3logmh_l'].apply(lambda x: x+np.log10(0.7)) # h=1 units
                mock = mock[(mock.g3logmh_l>0)] # to do by group, just also select the galaxy central flag=1
                mock = mock.groupby('g3grp_l').filter(lambda grp: (grp.M_r>-19.4).all())
                errors.append(mock.g3logmh_l-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l))
                truehalomass.append(np.array(mock.loghalom))
                i+=1


    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)

    print('MAD (iter comb.): ', mad(np.abs(errors)))
    print("frac less than 0.5 dex: ", len(errors[np.where(np.abs(errors)<=0.5)])/len(errors))

    # get median and 1-sigma CI
    median, bincenters, binedges, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)    
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10) 
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10) 
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10) 
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10) 
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10) 
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10) 

    medianer = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, np.median, kwargs=dict({'axis':1 })) for i in range(0,len(bincenters))]), axis=1)
    line16er = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, lambda x:np.nanpercentile(x,16,axis=1))  for i in range(0,len(bincenters))]), axis=1)
    print(line16er)


    # plot
    fig, (ax1,ax) = plt.subplots(sharey=True, ncols=2, figsize=(doublecolsize[0],0.8*doublecolsize[1]))
    #ax.scatter(g3logmh[0], errors[0], color='green', alpha=1, s=5, label='Mock Galaxies')
    #ax.hist2d(g3logmh, errors, bins=30, rasterized=True, norm=LogNorm())
    print("Median (G3): ", np.median(np.abs(errors)))
    ax.plot(bincenters, median, '^-', color='k', label='Median')
    ax.plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    ax.plot(bincenters, line84, linestyle='dashed', color='tomato')
    ax.plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    ax.plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    ax.plot(bincenters, ci5, color='tomato')
    ax.plot(bincenters, ci95, color='tomato', label='90th Percentile Confidence Interval')
    ax.set_xlabel(r'$\log M_{\rm HAM}$')
    ax.set_title(r"G3 Group Finding + HAM")
    #ax.set_title(r"G3 Groups: Linking Lengths $b_{||}\cdot{\rm MIN}\left(1s_0,s_{\rm adaptive,\, i}\right)$ and $b_{\perp}\cdot{\rm MIN}\left(\frac{7}{6}s_0,s_{\rm adaptive,\, i}\right)$")


    subdirs = ['fiducial/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    j=0
    for sd in subdirs:
        path = '/srv/scratch/zhutchen/dwarfonlymocks/'+sd
        files = os.listdir(path)
        for fname in files:
            if j<30:
                mock = pd.read_csv(path+fname)
                mock['g3logmh_l_sim']=mock['g3logmh_l_sim'].apply(lambda X:X+np.log10(0.7)) # h=1 units
                mock = mock[(mock.g3logmh_l_sim>0)]# & (mock.g_galtype==1)] # to do by group, just also select the galaxy central flag=1
                mock = mock.groupby('g3grp_l_sim').filter(lambda grp: (grp.M_r>-19.4).all())
                errors.append(mock.g3logmh_l_sim-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l_sim))
                truehalomass.append(np.array(mock.loghalom))
                j+=1

    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)

    # get median and 1-sigma CI
    median, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10)
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10)
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10)
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10)
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10)
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10)

    # plot
    #ax1.scatter(g3logmh[0], errors[0], color='green', alpha=1, s=5, label='Mock Galaxy Groups', rasterized=True)
    #ax1.hist2d(g3logmh, errors, bins=30, rasterized=True, norm=LogNorm())
    print("Median (True): ", np.median(np.abs(errors)))
    ax1.plot(bincenters, median, '^-', color='k', label='Median')
    ax1.plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    ax1.plot(bincenters, line84, linestyle='dashed', color='tomato')
    ax1.plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    ax1.plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    ax1.plot(bincenters, ci5, color='tomato')
    ax1.plot(bincenters, ci95, color='tomato', label='90th Percentile Confidence Interval')
    #ax1.legend(loc='best')
    #ax1.set_ylabel('Observed HAM Halo Mass - True Mock Halo Mass')
    ax1.set_xlabel(r'$\log M_{\rm HAM}$')
    ax1.set_ylabel(r'$\log M_{\rm HAM} - \log M_{\rm true}$')
    ax1.set_title(r"True Groups + HAM")
    ax.set_ylim(-2,1)
    ax1.set_ylim(-2,1)
    ax.grid()
    ax1.grid()
    ax1.legend(loc='best', framealpha=1)
    plt.tight_layout()
    plt.savefig("fig_dwarf_hmf_errors.pdf", dpi=150)
    plt.show()

    plt.clf()
    plt.hist(errors,bins='fd',color='lightgreen',log=True)
    plt.axvline(np.median(errors), color='k', label='Median')
    plt.xlabel("logM(HAM)-logM(true)")
    plt.show()
