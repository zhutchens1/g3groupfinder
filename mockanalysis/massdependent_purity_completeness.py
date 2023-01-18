
import sys
sys.path.insert(0,'../g3algo/')
from foftools import multiplicity_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator, LogFormatter, AutoLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from sklearn.preprocessing import normalize
from scipy.stats import median_abs_deviation as mad
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
    fig,axs=plt.subplots(ncols=2,nrows=2,figsize=(doublecolsize[0],doublecolsize[1]*1.1))

    ####################################################
    # Get data for first 2 axes --- by group
    ####################################################
    g3pur=[]
    g3comp=[]
    g3logmh=[]
    for ii in range(0,8):
        g3=pd.read_csv('../halobiasgroupcats/fiducial/ECO_cat_{}_Planck_memb_cat.csv'.format(ii))
        my_halo_ngal = multiplicity_function(g3.haloid.to_numpy(),True)
        g3=g3[(g3.halo_ngal.to_numpy()==my_halo_ngal)]
        #g3=g3[(g3.g3grpngi_l+g3.g3grpndw_l)>1]
        g3=g3[(g3.g3logmh_l>0)].groupby('g3grp_l').first() # get each group
        g3pur.append(np.array(g3.group_purity))
        g3comp.append(np.array(g3.group_comp))
        g3logmh.append(np.array(g3.g3logmh_l))
    g3pur=np.concatenate(g3pur)
    g3comp=np.concatenate(g3comp)
    g3logmh=np.concatenate(g3logmh)

    fofpur=[]
    fofcomp=[]
    foflogmh=[]
    for ii in range(0,8):
        fof=pd.read_csv('../halobiasgroupcats/fiducial/ECO_cat_{}_Planck_memb_cat.csv'.format(ii))
        my_halo_ngal = multiplicity_function(fof.haloid.to_numpy(),True)
        fof = fof[(fof.halo_ngal.to_numpy()==my_halo_ngal)]
        fof.loc[:,'fofe17grpn']=multiplicity_function(np.array(fof.fofe17id),return_by_galaxy=True)
        #fof=fof[fof.fofe17grpn>1]
        fof=fof[(fof.fofe17logmh>0)].groupby('fofe17id').first() # get each group
        fofpur.append(np.array(fof.fofe17purity_g))
        fofcomp.append(fof.fofe17completeness_g)
        foflogmh.append(np.array(fof.fofe17logmh))
    fofpur=np.concatenate(fofpur)
    fofcomp=np.concatenate(fofcomp)
    foflogmh=np.concatenate(foflogmh)

    # axs[0][0] - group purity
    binv = np.array([11,12,13,14,15])#np.arange(11,14.5,0.75)
    vdata = [np.array(g3pur[np.where(np.logical_and(g3logmh>binv[ii],g3logmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Purity (G3): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[0][0].violinplot(vdata,binc,showextrema=False)
    axs[0][0].scatter(binc,medians,marker='s',facecolor='none',edgecolor='blue', label='Median (G3)')
    axs[0][0].plot(binc,means,'b*', label='Mean (G3)')
    axs[0][0].set_ylim(0.6,1.05)    


    vdata = [np.array(fofpur[np.where(np.logical_and(foflogmh>binv[ii],foflogmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Purity (FoF): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[0][0].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("None")
        pc.set_edgecolor('darkorange')
        pc.set_linewidth(2)
        pc.set_alpha(0.8)
    axs[0][0].plot(binc,medians,'.',color='peru', label='Median (FoF)',markersize=10)
    axs[0][0].scatter(binc,means,marker='D',edgecolor='peru',facecolor='None',label='Mean (FoF)',s=40)
    axs[0][0].legend(loc='lower left',ncol=1)
    axs[0][0].set_xticks(binc, label=['{:0.2f}'.format(bv) for bv in binv])

    # axs[1][0] - group completeness
    vdata = [np.array(g3comp[np.where(np.logical_and(g3logmh>binv[ii],g3logmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Completeness (G3): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[1][0].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.2)
    axs[1][0].scatter(binc,medians,marker='s', facecolor='none', edgecolor='blue')
    axs[1][0].plot(binc,means,'b*')    

    vdata = [np.array(fofcomp[np.where(np.logical_and(foflogmh>binv[ii],foflogmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Completeness (FoF): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[1][0].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("None")
        pc.set_edgecolor('orange')
        pc.set_linewidth(2)
        pc.set_alpha(0.8)
    axs[1][0].plot(binc,medians,'.', color='peru',markersize=10)
    axs[1][0].scatter(binc,means,marker='D', edgecolor='peru',facecolor='None',s=40)
    axs[1][0].set_xticks(binc, label=['{:0.2f}'.format(bv) for bv in binv])

    #############################################################################################
    #############################################################################################
    # now do for halo mapping

    g3pur=[]
    g3comp=[]
    g3logmh=[]
    for ii in range(0,8):
        g3=pd.read_csv('../halobiasgroupcats/fiducial/ECO_cat_{}_Planck_memb_cat.csv'.format(ii))
        my_halo_ngal = multiplicity_function(g3.haloid.to_numpy(),True)
        g3=g3[(g3.halo_ngal.to_numpy()==my_halo_ngal)]
        g3=g3.groupby('haloid').first()
        g3pur.append(np.array(g3.halo_purity))
        g3comp.append(np.array(g3.halo_comp))
        g3logmh.append(np.array(g3.loghalom))
    g3pur=np.concatenate(g3pur)
    g3comp=np.concatenate(g3comp)
    g3logmh=np.concatenate(g3logmh)

    fofpur=[]
    fofcomp=[]
    foflogmh=[]
    for ii in range(0,8):
        fof=pd.read_csv('../halobiasgroupcats/fiducial/ECO_cat_{}_Planck_memb_cat.csv'.format(ii))
        my_halo_ngal = multiplicity_function(fof.haloid.to_numpy(),True)
        fof = fof[(fof.halo_ngal.to_numpy()==my_halo_ngal)]
        fof.loc[:,'fofe17grpn']=multiplicity_function(np.array(fof.fofe17id),return_by_galaxy=True)
        #fof=fof[fof.fofe17grpn>1]
        #fof=fof[(fof.fofe17logmh>0)].groupby('fofe17id').first() # get each group
        fof = fof.groupby('haloid').first()
        fofpur.append(np.array(fof.fofe17purity_h))
        fofcomp.append(fof.fofe17completeness_h)
        foflogmh.append(np.array(fof.loghalom))
    fofpur=np.concatenate(fofpur)
    fofcomp=np.concatenate(fofcomp)
    foflogmh=np.concatenate(foflogmh)

    # axs[0][1] - halo purity
    vdata = [np.array(g3pur[np.where(np.logical_and(g3logmh>binv[ii],g3logmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Purity (G3): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[0][1].violinplot(vdata,binc,showextrema=False)
    axs[0][1].scatter(binc,medians,marker='s',facecolor='none',edgecolor='blue', label='Median (G3)')
    axs[0][1].plot(binc,means,'b*', label='Mean (G3)')
    axs[0][1].set_ylim(0.6,1.05)

    vdata = [np.array(fofpur[np.where(np.logical_and(foflogmh>binv[ii],foflogmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Purity (FoF): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[0][1].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("None")
        pc.set_edgecolor('darkorange')
        pc.set_linewidth(2)
        pc.set_alpha(0.8)
    axs[0][1].plot(binc,medians,'.',color='peru', label='Median (FoF)',markersize=10)
    axs[0][1].scatter(binc,means,marker='D',edgecolor='peru', facecolor='None', label='Mean (FoF)',s=40)
    #axs[0][1].legend(loc='best',ncol=2)
    axs[0][1].set_xticks(binc, label=['{:0.2f}'.format(bv) for bv in binv])

    # axs[1][1] - halo completeness
    vdata = [np.array(g3comp[np.where(np.logical_and(g3logmh>binv[ii],g3logmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Completeness (G3): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[1][1].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.2)
    axs[1][1].scatter(binc,medians,marker='s', facecolor='none', edgecolor='blue')
    axs[1][1].plot(binc,means,'b*')

    vdata = [np.array(fofcomp[np.where(np.logical_and(foflogmh>binv[ii],foflogmh<=binv[ii+1]))]) for ii in range(0,len(binv)-1)]
    vdata=np.array(vdata,dtype=object)
    vstats = np.array([np.percentile(col,[16,50,84]) for col in vdata])
    means = np.array([np.mean(col) for col in vdata])
    medians = np.array([np.median(col) for col in vdata])
    print("Completeness (FoF): ",medians)
    binc = 0.5*(binv[:-1]+binv[1:])
    parts=axs[1][1].violinplot(vdata,binc,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor("None")
        pc.set_edgecolor('orange')
        pc.set_linewidth(2)
        pc.set_alpha(0.8)
    axs[1][1].plot(binc,medians,'.', color='peru',markersize=10)
    axs[1][1].scatter(binc,means,marker='D', edgecolor='peru',facecolor='None',s=40)
    axs[1][1].set_xticks(binc, label=['{:0.2f}'.format(bv) for bv in binv])


    #########################
    # decorations
    axs[0][0].set_ylabel(r"$P_g$")
    axs[0][1].set_ylabel(r"$P_h$")
    axs[1][0].set_ylabel(r"$C_g$")
    axs[1][1].set_ylabel(r"$C_h$")
    axs[1][0].set_xlabel(r"log group $M_{\rm HAM}$")
    axs[1][1].set_xlabel(r"log mock halo $M_{\rm true}$")
    for ii in range(0,2):
        for jj in range(0,2):
            axs[ii][jj].set_yticks(np.arange(0,1.2,0.2))
            axs[ii][jj].set_ylim(0,1.02)
            axs[ii][jj].yaxis.set_minor_locator(AutoLocator())
    plt.tight_layout()
    plt.savefig("../figures/pur_comp_dists.pdf",dpi=300)
    plt.show() 
