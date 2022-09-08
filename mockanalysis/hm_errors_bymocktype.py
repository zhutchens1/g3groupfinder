import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from center_binned_stats import center_binned_stats
from scipy.stats import gaussian_kde, median_abs_deviation as mad
from smoothedbootstrap import smoothedbootstrap as sbs
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from astroML.plotting import scatter_contour
from matplotlib.colors import LogNorm
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
doublecolsize = (7.100005949910059, 4.3880449973709)
HUBBLE_CONST = 70.

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

def weighted_median_err(data,weights,perc=0.5,n_bs=1000):
    data=np.array(data)
    weights=np.array(weights)
    df = pd.DataFrame(np.array([data,weights]).T, columns=['data','weights'])
    target=np.zeros(n_bs)
    for ii in range(0,n_bs):
        tmp =  df.sample(frac=1,replace=True)
        target[ii] = weighted_percentile(np.array(tmp.data), np.array(tmp.weights), perc)
    return np.std(target) 

# Program to get error bars on G3 group g3logmh_l
# by comparing to true halo masses and looking at
# the 1-sigma dispersion for all groups in all
# 32 catalogs

if __name__=='__main__':
    subdirs = ['fiducial/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    grpn=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock = mock[(mock.g3logmh_l>0)] # to do by group, just also select the galaxy central flag=1
                mock.loc[:,'g3logmh_l']=mock.g3logmh_l-np.log10(HUBBLE_CONST/100.)
                mock.loc[:,'loghalom']=mock.loghalom-np.log10(HUBBLE_CONST/100.)
                errors.append(mock.g3logmh_l-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l))
                truehalomass.append(np.array(mock.loghalom))
                grpn.append(np.array(mock.g3grpn_l))
                i+=1

    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)
    grpn = np.concatenate(grpn)
    # get median and 1-sigma CI
    wts = (1/grpn)/np.sum(1/grpn) # np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    mu_HME = weighted_percentile(np.abs(errors), wts, 0.5)
    mu_HME_err = weighted_median_err(np.abs(errors), wts, 0.5, 500)
    median, bincenters, binedges, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)    
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10) 
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10) 
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10) 
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10) 
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10) 
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10) 

    sel=(bincenters>13)
    print("G3 fid: Mean width of 1-sig CI: ", np.median(line84[sel]-line16[sel]))
    print(np.sum(np.abs(errors[g3logmh>13])))
    print(mad(np.abs(errors[g3logmh>13])))

    medianer = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, np.median, kwargs=dict({'axis':1 })) for i in range(0,len(bincenters))]), axis=1)
    line16er = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, lambda x:np.nanpercentile(x,16,axis=1))  for i in range(0,len(bincenters))]), axis=1)
    print(line16er)


    # plot
    #fig, ((ax,ax1),(ax2,ax3)) = plt.subplots(nrows=4, ncols=2, figsize=(doublecolsize[0], 2*doublecolsize[1]))
    fig=plt.figure(figsize=(doublecolsize[0], 2.1*doublecolsize[1]))

    axd = fig.subplot_mosaic(
        """  AA
             BC
             DE
        """
    )
    #axd['B'].scatter(g3logmh, errors, color='green', alpha=0.05, s=5, rasterized=True)
    #scatter_contour(g3logmh,errors,ax=axd['B'],levels=20)
    #axd['B'].hist2d(g3logmh,errors,bins=40,norm=LogNorm(),rasterized=True,cmap='Purples')
    axd['B'].plot(bincenters, median, '^-', color='k', label='Median')
    axd['B'].plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    axd['B'].plot(bincenters, line84, linestyle='dashed', color='tomato')
    axd['B'].plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    axd['B'].plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    axd['B'].plot(bincenters, ci5, color='tomato')
    axd['B'].plot(bincenters, ci95, color='tomato', label='90th Percentile Confidence Interval')
    axd['B'].set_title(r"G3 Group Finder (Fiducial)")
    axd['B'].set_ylabel(r"$\log M_{\rm HAM} - \log M_{\rm true}$")
    axd['B'].set_ylim(-3,4)
    axd['B'].grid()
    #axd['B'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['B']Etext(10.7,3,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    axd['B'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')




    #################################################################
    #################################################################

    subdirs = ['central/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    grpn=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock = mock[(mock.g3logmh_l>0)] # to do by group, just also select the galaxy central flag=1
                mock.loc[:,'g3logmh_l']=mock.g3logmh_l-np.log10(HUBBLE_CONST/100.)
                mock.loc[:,'loghalom']=mock.loghalom-np.log10(HUBBLE_CONST/100.)
                errors.append(mock.g3logmh_l-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l))
                truehalomass.append(np.array(mock.loghalom))
                grpn.append(np.array(mock.g3grpn_l))
                i+=1


    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)
    grpn=np.concatenate(grpn)

    # get median and 1-sigma CI
    wts = (1/grpn)/np.sum(1/grpn) # np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    mu_HME = weighted_percentile(np.abs(errors), wts, 0.5)
    mu_HME_err = weighted_median_err(np.abs(errors), wts, 0.5, 500)
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

    #axd['C'].hist2d(g3logmh,errors,bins=40,norm=LogNorm(),rasterized=True)
    axd['C'].plot(bincenters, median, '^-', color='k', label='Median')
    axd['C'].plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    axd['C'].plot(bincenters, line84, linestyle='dashed', color='tomato')
    axd['C'].plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    axd['C'].plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    axd['C'].plot(bincenters, ci5, color='tomato')
    axd['C'].plot(bincenters, ci95, color='tomato', label='90th Percentile Confidence Interval')
    #axd['B'].legend(loc='best')
    #axd['B'].set_ylabel('Observed HAM Halo Mass - True Mock Halo Mass')
    #axd['B'].set_xlabel('Observed HAM Halo Mass')
    axd['C'].set_title(r"G3 Group Finder (Central-Offset)")
    axd['C'].set_ylim(-3,4)
    axd['C'].grid()
    #axd['C'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['C'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['C'].text(10.7,3,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    axd['C'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')


    ######################################
    ######################################
    ######################################

    subdirs = ['dv0_8/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    grpn=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock = mock[(mock.g3logmh_l>0)] # to do by group, just also select the galaxy central flag=1
                mock.loc[:,'g3logmh_l']=mock.g3logmh_l-np.log10(HUBBLE_CONST/100.)
                mock.loc[:,'loghalom']=mock.loghalom-np.log10(HUBBLE_CONST/100.)
                errors.append(mock.g3logmh_l-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l))
                truehalomass.append(np.array(mock.loghalom))
                grpn.append(np.array(mock.g3grpn_l))
                i+=1


    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)
    grpn = np.concatenate(grpn)

    # get median and 1-sigma CI
    wts = (1/grpn)/np.sum(1/grpn) # np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    mu_HME = weighted_percentile(np.abs(errors), wts, 0.5)
    mu_HME_err = weighted_median_err(np.abs(errors), wts, 0.5, 500)
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

    #ax.scatter(g3logmh[0], errors[0], color='green', alpha=1, s=5, label='Mock Galaxies')
    #axd['D'].scatter(g3logmh, errors, color='green', alpha=0.05, s=5, rasterized=True)
    #axd['D'].hist2d(g3logmh,errors,bins=40,norm=LogNorm(),rasterized=True)
    axd['D'].plot(bincenters, median, '^-', color='k', label='Median')
    axd['D'].plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    axd['D'].plot(bincenters, line84, linestyle='dashed', color='tomato')
    axd['D'].plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    axd['D'].plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    axd['D'].plot(bincenters, ci5, color='tomato')
    axd['D'].plot(bincenters, ci95, color='tomato', label='90th Percentile Confidence Interval')
    axd['D'].set_ylim(-3,4)
    axd['D'].grid()
    #axd['B'].legend(loc='best')
    #axd['B'].set_ylabel('Observed HAM Halo Mass - True Mock Halo Mass')
    #axd['B'].set_xlabel('Observed HAM Halo Mass')
    axd['D'].set_title(r"G3 Group Finder ($b_v=0.8$)")
    axd['D'].set_xlabel(r"$\log M_{\rm HAM}$")
    #axd['D'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['D'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    axd['D'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')

    ##################################################
    ##################################################
    ##################################################
    subdirs = ['dv1_2/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    grpn=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock = mock[(mock.g3logmh_l>0)] # to do by group, just also select the galaxy central flag=1
                mock.loc[:,'g3logmh_l']=mock.g3logmh_l-np.log10(HUBBLE_CONST/100.)
                mock.loc[:,'loghalom']=mock.loghalom-np.log10(HUBBLE_CONST/100.)
                errors.append(mock.g3logmh_l-mock.loghalom)
                observedhalomass.append(np.array(mock.g3logmh_l))
                truehalomass.append(np.array(mock.loghalom))
                grpn.append(np.array(mock.g3grpn_l))
                i+=1


    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)
    grpn=np.concatenate(grpn)

    # get median and 1-sigma CI
    wts = (1/grpn)/np.sum(1/grpn) # np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    mu_HME =  weighted_percentile(np.abs(errors), wts, 0.5)
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

    #ax.scatter(g3logmh[0], errors[0], color='green', alpha=1, s=5, label='Mock Galaxies')
    #axd['E'].scatter(g3logmh, errors, color='green', alpha=0.05, s=5, rasterized=True)
    #axd['E'].hist2d(g3logmh,errors,bins=40,norm=LogNorm(),rasterized=True)
    axd['E'].plot(bincenters, median, '^-', color='k', label='Median')
    axd['E'].plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$ Dispersion')
    axd['E'].plot(bincenters, line84, linestyle='dashed', color='tomato')
    axd['E'].plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$ Dispersion')
    axd['E'].plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    axd['E'].plot(bincenters, ci5, color='tomato')
    axd['E'].plot(bincenters, ci95, color='tomato', label='90% Confidence Interval')
    axd['E'].set_ylim(-3,4)
    axd['E'].grid()
    #axd['B'].legend(loc='best')
    #axd['B'].set_ylabel('Observed HAM Halo Mass - True Mock Halo Mass')
    #axd['B'].set_xlabel('Observed HAM Halo Mass')
    #axd['E'].set_xlabel(r"Mock HAM Halo Mass [$\log\rm M_\odot$]")
    axd['E'].set_xlabel(r"$\log M_{\rm HAM}$")
    axd['E'].set_title(r"G3 Group Finder ($b_v=1.2$)")
    #axd['E'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['E'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    axd['E'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')

    ##################################################
    ##################################################
    ##################################################
    subdirs = ['fiducial/']#, 'dv0_8/', 'dv1_2/', 'central/']
    errors = []
    observedhalomass=[]
    truehalomass=[]
    grpn=[]
    i=0
    for sd in subdirs:
        path = '../halobiasgroupcats/'+sd
        files = os.listdir(path)
        for fname in files:
            if i<30:
                mock = pd.read_csv(path+fname)
                mock = mock[(mock.fofe17logmh>0)] # to do by group, just also select the galaxy central flag=1
                mock.loc[:,'fofe17logmh']=mock.fofe17logmh-np.log10(HUBBLE_CONST/100.)
                mock.loc[:,'loghalom']=mock.loghalom-np.log10(HUBBLE_CONST/100.)
                errors.append(mock.fofe17logmh-mock.loghalom)
                observedhalomass.append(np.array(mock.fofe17logmh))
                truehalomass.append(np.array(mock.loghalom))
                grpn.append(np.array(mock.fofe17grpn))
                i+=1


    # now combine
    errors = np.concatenate(errors)
    g3logmh = np.concatenate(observedhalomass)
    truehalomass = np.concatenate(truehalomass)
    grpn = np.concatenate(grpn)

    # get median and 1-sigma CI
    wts = (1/grpn)#/np.sum(1/grpn) # np.array((1/mockdf.ecog3grpn_l)/np.sum(1/mockdf.ecog3grpn_l))
    mu_HME = weighted_percentile(np.abs(errors), wts, 0.5)
    mu_HME_err = weighted_median_err(np.abs(errors), wts, 0.5, 500)
    median, bincenters, binedges, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10)
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10)
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10)
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10)
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10)
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10)

    sel=(bincenters>13)
    print("FoF: Mean width of 1-sig CI: ", np.median(line84[sel]-line16[sel]))
    print(np.sum(np.abs(errors[g3logmh>13])))
    print(mad(np.abs(errors[g3logmh>13])))

    medianer = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, np.median, kwargs=dict({'axis':1 })) for i in range(0,len(bincenters))]), axis=1)
    line16er = np.std(np.array([sbs(errors[np.where(np.logical_and(g3logmh>binedges[:-1][i], g3logmh<=binedges[1:][i]))], 1000, lambda x:np.nanpercentile(x,16,axis=1))  for i in range(0,len(bincenters))]), axis=1)
    print(line16er)

    #ax.scatter(g3logmh[0], errors[0], color='green', alpha=1, s=5, label='Mock Galaxies')
    #axd['A'].hist2d(g3logmh,errors,bins=40,norm=LogNorm(),rasterized=True)
    #axd['A'].scatter(g3logmh, errors, color='green', alpha=0.05, s=5, rasterized=True)
    axd['A'].plot(bincenters, median, '^-', color='k', label='Median')
    axd['A'].plot(bincenters, line16, linestyle='dashed', color='tomato', label=r'$\pm 1\sigma$')
    axd['A'].plot(bincenters, line84, linestyle='dashed', color='tomato')
    axd['A'].plot(bincenters, line2pt5, linestyle='dashdot', color='tomato', label=r'$\pm 2\sigma$')
    axd['A'].plot(bincenters, line97pt5, linestyle='dashdot', color='tomato')
    axd['A'].plot(bincenters, ci5, color='tomato')
    axd['A'].plot(bincenters, ci95, color='tomato', label='90% CI')
    axd['A'].legend(loc='lower right', framealpha=1)
    #axd['B'].set_ylabel('Observed HAM Halo Mass - True Mock Halo Mass')
    #axd['B'].set_xlabel('Observed HAM Halo Mass')
    axd['A'].set_title(r"FoF + False Pair Splitting (Fiducial)")
    axd['A'].set_ylim(-3,4)
    axd['A'].grid()
    #axd['A'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    axd['A'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')

    #fig.text(0.5,0.04, 'log group HAM virial mass', ha='center')
    #fig.text(0.04,0.5, 'Observed HAM Halo Mass - True Mock Halo Mass', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig("../figures/halomasserrors_g3.pdf", dpi=150, bbox_inches='tight')
    plt.show()

