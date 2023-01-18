import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from center_binned_stats import center_binned_stats
from scipy.stats import gaussian_kde, median_abs_deviation as mad
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib import rcParams
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
                print(np.max(mock.M_r))
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

    # plot
    fig=plt.figure(figsize=(doublecolsize[0], 2.1*doublecolsize[1]))

    axd = fig.subplot_mosaic(
        """
             BC
             DE
             AF
        """
    )
    REF_bincenters=bincenters
    REF_median=median
    REF_line16=line16
    REF_line84=line84
    REF_line2pt5=line2pt5
    REF_line97pt5=line97pt5
    REF_line5=ci5
    REF_line95=ci95
    def plot_ref_shading(char):
        #axd[char].axhline(0,color='k',linewidth=0.3)
        axd[char].plot(REF_bincenters,REF_median,color='#105f47',linewidth=4, label='Median (G3 Fiducial)') #524e7d
        axd[char].fill_between(REF_bincenters, REF_line16, REF_line84, color='#1b9e77',alpha=0.6, label=r'16$^{\rm th}$-84$^{\rm th}$ Percentiles (G3 Fiducial)')
        axd[char].fill_between(REF_bincenters, REF_line5, REF_line95, color='#1b9e77',alpha=0.4, label=r'5$^{\rm th}$-95$^{\rm th}$ Percentiles (G3 Fiducial)')
        axd[char].fill_between(REF_bincenters, REF_line2pt5, REF_line97pt5, color='#1b9e77',alpha=0.3, label=r'2.5$^{\rm th}$-97.5$^{\rm th}$ Percentiles (G3 Fiducial)')
        axd[char].xaxis.set_minor_locator(AutoMinorLocator())
        axd[char].yaxis.set_minor_locator(AutoMinorLocator())
        if char=='E' or char=='A':
            axd[char].set_xlabel(r"$\log M_{\rm HAM}$ [$\rm M_\odot$]")
        if char=='D':
            axd[char].set_ylabel(r"$\log M_{\rm HAM} - \log M_{\rm true}$")
    plot_ref_shading('B')

    axd['B'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['B'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'16$^{\rm th}$-84$^{\rm th}$ Percentiles')
    axd['B'].plot(bincenters, ci95, color='#d95f02', label=r'5$^{\rm th}$-95$^{\rm th}$ Percentiles')
    axd['B'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['B'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'2.5$^{\rm th}$-97.5$^{\rm th}$ Percentiles')
    axd['B'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['B'].plot(bincenters, ci5, color='#d95f02')
    #axd['B'].set_title(r"G3 Group Finder (Fiducial)", fontsize=10)
    axd['B'].annotate("G3 Groups\n(Fiducial)",xy=(11.2,2),fontsize=10)
    axd['B'].set_ylim(-2.5,3.5)
    #axd['B'].legend(loc='best', bbox_to_anchor=(1.5,-4))
    #axd['B'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['B']Etext(10.7,3,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['B'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')




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
    median, bincenters, binedges, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10)
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10)
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10)
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10)
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10)
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10)

    plot_ref_shading('C')
    axd['C'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['C'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'$\pm 1\sigma$ Dispersion')
    axd['C'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['C'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'$\pm 2\sigma$ Dispersion')
    axd['C'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['C'].plot(bincenters, ci5, color='#d95f02')
    axd['C'].plot(bincenters, ci95, color='#d95f02', label='90th Percentile Confidence Interval')
    #axd['C'].set_title(r"G3 Group Finder (Central-Offset)")
    axd['C'].annotate("G3 Groups\n(Central-Offset)",xy=(11.2,2),fontsize=10)
    axd['C'].set_ylim(-2.5,3.5)
    #axd['C'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['C'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['C'].text(10.7,3,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['C'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')


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
    median, bincenters, binedges, jk = center_binned_stats(g3logmh, errors, statistic='median', bins=10)
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10)
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10)
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10)
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10)
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10)
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10)

    plot_ref_shading("D")
    axd['D'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['D'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'$\pm 1\sigma$ Dispersion')
    axd['D'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['D'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'$\pm 2\sigma$ Dispersion')
    axd['D'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['D'].plot(bincenters, ci5, color='#d95f02')
    axd['D'].plot(bincenters, ci95, color='#d95f02', label='90th Percentile Confidence Interval')
    axd['D'].set_ylim(-2.5,3.5)
    #axd['D'].set_title(r"G3 Group Finder ($b_v=0.8$)")
    axd['D'].annotate("G3 Groups\n"+r"($b_v=0.8$)",xy=(11.2,2))
    #axd['D'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['D'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['D'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')

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
    line16, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,16) , bins=10)
    line2pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,2.5) , bins=10)
    line84, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,84) , bins=10)
    line97pt5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,97.5) , bins=10)
    ci5, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,5) , bins=10)
    ci95, bincenters, jk, jk = center_binned_stats(g3logmh, errors, statistic=lambda x:np.nanpercentile(x,95) , bins=10)

    plot_ref_shading('E')
    axd['E'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['E'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'$\pm 1\sigma$ Dispersion')
    axd['E'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['E'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'$\pm 2\sigma$ Dispersion')
    axd['E'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['E'].plot(bincenters, ci5, color='#d95f02')
    axd['E'].plot(bincenters, ci95, color='#d95f02', label='90% Confidence Interval')
    axd['E'].set_ylim(-2.5,3.5)
    #axd['E'].set_title(r"G3 Group Finder ($b_v=1.2$)")
    axd['E'].annotate("G3 Groups\n"+r"($b_v=1.2$)",xy=(11.2,2))
    #axd['E'].annotate(r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME), xy=(11,2.5), fontsize=12, alpha=1)
    #axd['E'].text(11,2.8,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME),fontsize=12,backgroundcolor='white')
    #axd['E'].text(11.1,3.1,r"$\mu_{\rm HME} = $ "+"{:0.3f}".format(mu_HME)+r"$\pm$"+"{:0.4f}".format(mu_HME_err),fontsize=12,backgroundcolor='white')

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

    plot_ref_shading('A')
    axd['A'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['A'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'$\pm 1\sigma$')
    axd['A'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['A'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'$\pm 2\sigma$')
    axd['A'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['A'].plot(bincenters, ci5, color='#d95f02')
    axd['A'].plot(bincenters, ci95, color='#d95f02', label='90% CI')
    #axd['A'].set_title(r"FoF + False Pair Splitting (Fiducial)", fontsize=9)
    axd['A'].annotate("FoF (E17) Groups\n(Fiducial)",xy=(11.2,2),fontsize=10)
    axd['A'].set_ylim(-2.5,3.5)
    #axd['A'].legend(loc='center left', bbox_to_anchor=(2,0.5))

    #fig.text(0.5,0.04, 'log group HAM virial mass', ha='center')
    #fig.text(0.04,0.5, 'Observed HAM Halo Mass - True Mock Halo Mass', va='center', rotation='vertical')


    plot_ref_shading('F')
    axd['F'].plot(bincenters, median, '-o', color='#d95f02', label='Median')
    axd['F'].plot(bincenters, line16, linestyle='dashed', color='#d95f02', label=r'16$^{\rm th}$-84$^{\rm th}$ Percentiles')
    axd['F'].plot(bincenters, ci95, color='#d95f02', label=r'5$^{\rm th}$-95$^{\rm th}$ Percentiles')
    axd['F'].plot(bincenters, line84, linestyle='dashed', color='#d95f02')
    axd['F'].plot(bincenters, line2pt5, linestyle='dashdot', color='#d95f02', label=r'2.5$^{\rm th}$-97.5$^{\rm th}$ Percentiles')
    axd['F'].plot(bincenters, line97pt5, linestyle='dashdot', color='#d95f02')
    axd['F'].plot(bincenters, ci5, color='#d95f02')
    axd['F'].axis('off')
    axd['F'].set_xlim(100,101)
    axd['F'].legend(loc='center',ncol=1)
    plt.tight_layout()
    plt.savefig("../figures/halomasserrors_g3.pdf", dpi=150, bbox_inches='tight')
    plt.show()

