import os
import sys
sys.path.insert(0,'../g3algo/')
import foftools as fof
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from center_binned_stats import center_binned_stats as cbs
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
doublecolsize = (7.100005949910059, 4.3880449973709)
H0 = 70.
subdirs = ['fiducial/']

if __name__=='__main__':
    # mock distributions of R_proj,gal + v_proj,gal
    mockRprojgal=[]
    mockvprojgal=[]
    mockhalomass=[]
    for ii,sd in enumerate(subdirs):
        path='../halobiasgroupcats/'+sd
        files=os.listdir(path)
        for fname in files:
            mock = pd.read_csv(path+fname)
            mock = mock[(mock.halo_ngal>1)&(mock.M_r<=-17.33)]
            haloradeg, halodedeg, halocz = fof.group_skycoords(np.array(mock.ra), np.array(mock.dec), np.array(mock.cz), np.array(mock.haloid))
            mock.loc[:,'haloradeg']=haloradeg
            mock.loc[:,'halodedeg']=halodedeg
            mock.loc[:,'halocz']=halocz
            mock=mock[(mock.halocz>3000)&(mock.halocz<7000)]
            mock.loc[:,'Rproj_gal'] = (1/H0)*(mock['cz']+mock['halocz'])*np.sin(fof.angular_separation(mock['ra'], mock['dec'],\
                     mock['haloradeg'], mock['halodedeg'])/2.)
            mock.loc[:,'vproj_gal'] = np.abs(mock.cz-mock.halocz)
            mock.loc[:,'loghalom'] = mock.loghalom - np.log10(H0/100.)
            mockRprojgal.append(mock.Rproj_gal.to_numpy())
            mockvprojgal.append(mock.vproj_gal.to_numpy())
            mockhalomass.append(mock.loghalom.to_numpy())
    mockRprojgal = np.concatenate(mockRprojgal)
    mockvprojgal = np.concatenate(mockvprojgal)
    mockhalomass = np.concatenate(mockhalomass)

    # G3 distributions (in mocks) of R_proj,gal + v_proj,gal
    g3Rprojgal=[]
    g3vprojgal=[]
    g3halomass=[]
    for ii,sd in enumerate(subdirs):
        path='../halobiasgroupcats/'+sd
        files=os.listdir(path)
        for fname in files:
            mock = pd.read_csv(path+fname)
            mock = mock[(mock.g3grpn_l>1)&(mock.M_r<=-17.33)]
            grpradeg,grpdedeg,grpcz = fof.group_skycoords(mock.ra.to_numpy(),mock.dec.to_numpy(),mock.cz.to_numpy(),mock.g3grp_l.to_numpy())
            mock.loc[:,'grpradeg']=grpradeg
            mock.loc[:,'grpdedeg']=grpdedeg
            mock.loc[:,'grpcz']=grpcz
            mock=mock[(mock.grpcz>3000)&(mock.grpcz<7000)]
            mock.loc[:,'Rproj_gal'] = (1/H0)*(mock['cz']+mock['grpcz'])*np.sin(fof.angular_separation(mock['ra'], mock['dec'],\
                     mock['grpradeg'], mock['grpdedeg'])/2.)
            mock.loc[:,'vproj_gal'] = np.abs(mock.cz-mock.grpcz)
            mock.loc[:,'g3logmh_l'] = mock.g3logmh_l - np.log10(H0/100.)
            g3Rprojgal.append(mock.Rproj_gal.to_numpy())
            g3vprojgal.append(mock.vproj_gal.to_numpy())
            g3halomass.append(mock.g3logmh_l.to_numpy())
    g3Rprojgal = np.concatenate(g3Rprojgal)
    g3vprojgal = np.concatenate(g3vprojgal)
    g3halomass = np.concatenate(g3halomass)

    # FoF distributions (in mocks) of R_proj,gal + v_proj,gal
    fofRprojgal=[]
    fofvprojgal=[]
    fofhalomass=[]
    for ii,sd in enumerate(subdirs):
        path='../halobiasgroupcats/'+sd
        files=os.listdir(path)
        for fname in files:
            mock = pd.read_csv(path+fname)
            mock = mock[(mock.fofe17grpn>1)&(mock.M_r<=-17.33)]
            grpradeg,grpdedeg,grpcz = fof.group_skycoords(mock.ra.to_numpy(),mock.dec.to_numpy(),mock.cz.to_numpy(),mock.fofe17id.to_numpy())
            mock.loc[:,'grpradeg']=grpradeg
            mock.loc[:,'grpdedeg']=grpdedeg
            mock.loc[:,'grpcz']=grpcz
            mock=mock[(mock.grpcz>3000)&(mock.grpcz<7000)]
            mock.loc[:,'Rproj_gal'] = (1/H0)*(mock['cz']+mock['grpcz'])*np.sin(fof.angular_separation(mock['ra'], mock['dec'],\
                     mock['grpradeg'], mock['grpdedeg'])/2.)
            mock.loc[:,'vproj_gal'] = np.abs(mock.cz-mock.grpcz)
            mock.loc[:,'fofe17logmh'] = mock.fofe17logmh - np.log10(H0/100.)
            fofRprojgal.append(mock.Rproj_gal.to_numpy())
            fofvprojgal.append(mock.vproj_gal.to_numpy())
            fofhalomass.append(mock.fofe17logmh.to_numpy())
    fofRprojgal = np.concatenate(fofRprojgal)
    fofvprojgal = np.concatenate(fofvprojgal)
    fofhalomass = np.concatenate(fofhalomass)

    #############################################
    #############################################
    binv = np.arange(11,14.25,0.25)
    fig, (ax,ax1)=plt.subplots(ncols=2,figsize=doublecolsize)

    medianmockR,binc,_,_ = cbs(mockhalomass,mockRprojgal,'median',bins=binv)
    perc84mockR,binc,_,_ = cbs(mockhalomass,mockRprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16mockR,binc,_,_ = cbs(mockhalomass,mockRprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975mockR,binc,_,_ = cbs(mockhalomass,mockRprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25mockR,binc,_,_ = cbs(mockhalomass,mockRprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax.plot(binc,medianmockR,color='blue',linewidth=3,label='True Groups')
    ax.fill_between(binc,perc16mockR,perc84mockR,color='cornflowerblue',alpha=0.7)
    ax.fill_between(binc,perc25mockR,perc975mockR,color='cornflowerblue',alpha=0.5)

    mediang3R,binc,_,_ = cbs(g3halomass,g3Rprojgal,'median',bins=binv)
    perc84g3R,binc,_,_ = cbs(g3halomass,g3Rprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16g3R,binc,_,_ = cbs(g3halomass,g3Rprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975g3R,binc,_,_ = cbs(g3halomass,g3Rprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25g3R,binc,_,_ = cbs(g3halomass,g3Rprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax.plot(binc,mediang3R,color='chocolate',linewidth=2,label='G3')
    ax.plot(binc,perc16g3R,color='chocolate',linestyle='-.')
    ax.plot(binc,perc84g3R,color='chocolate',linestyle='-.')
    ax.plot(binc,perc25g3R,color='chocolate',linestyle='dotted')
    ax.plot(binc,perc975g3R,color='chocolate',linestyle='dotted')

    medianfofR,binc,_,_ = cbs(fofhalomass,fofRprojgal,'median',bins=binv)
    perc84fofR,binc,_,_ = cbs(fofhalomass,fofRprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16fofR,binc,_,_ = cbs(fofhalomass,fofRprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975fofR,binc,_,_ = cbs(fofhalomass,fofRprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25fofR,binc,_,_ = cbs(fofhalomass,fofRprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax.plot(binc,medianfofR,color='darkmagenta',linewidth=2, label='FoF + False Pair Splitting')
    ax.plot(binc,perc16fofR,color='darkmagenta',linestyle='-.')
    ax.plot(binc,perc84fofR,color='darkmagenta',linestyle='-.')
    ax.plot(binc,perc25fofR,color='darkmagenta',linestyle='dotted')
    ax.plot(binc,perc975fofR,color='darkmagenta',linestyle='dotted')
   
    ax.legend(loc='upper left')
    ax.set_xlim(np.min(binc),np.max(binc))
    ax.set_ylim(0,0.6) 
    ax.set_ylabel(r'$R_{\rm proj,\, gal}$ [Mpc]')
    ax.set_xlabel("log group halo mass")



    medianmockv,binc,_,_ = cbs(mockhalomass,mockvprojgal,'median',bins=binv)
    perc84mockv,binc,_,_ = cbs(mockhalomass,mockvprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16mockv,binc,_,_ = cbs(mockhalomass,mockvprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975mockv,binc,_,_ = cbs(mockhalomass,mockvprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25mockv,binc,_,_ = cbs(mockhalomass,mockvprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax1.plot(binc,medianmockv,color='blue',linewidth=3,label='True Groups')
    ax1.fill_between(binc,perc16mockv,perc84mockv,color='cornflowerblue',alpha=0.7)
    ax1.fill_between(binc,perc25mockv,perc975mockv,color='cornflowerblue',alpha=0.5)

    mediang3v,binc,_,_ = cbs(g3halomass,g3vprojgal,'median',bins=binv)
    perc84g3v,binc,_,_ = cbs(g3halomass,g3vprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16g3v,binc,_,_ = cbs(g3halomass,g3vprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975g3v,binc,_,_ = cbs(g3halomass,g3vprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25g3v,binc,_,_ = cbs(g3halomass,g3vprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax1.plot(binc,mediang3v,color='chocolate',linewidth=2,label='G3')
    ax1.plot(binc,perc16g3v,color='chocolate',linestyle='-.')
    ax1.plot(binc,perc84g3v,color='chocolate',linestyle='-.')
    ax1.plot(binc,perc25g3v,color='chocolate',linestyle='dotted')
    ax1.plot(binc,perc975g3v,color='chocolate',linestyle='dotted')

    medianfofv,binc,_,_ = cbs(fofhalomass,fofvprojgal,'median',bins=binv)
    perc84fofv,binc,_,_ = cbs(fofhalomass,fofvprojgal,lambda x: np.percentile(x,84),bins=binv)
    perc16fofv,binc,_,_ = cbs(fofhalomass,fofvprojgal,lambda x: np.percentile(x,16),bins=binv)
    perc975fofv,binc,_,_ = cbs(fofhalomass,fofvprojgal,lambda x: np.percentile(x,97.5),bins=binv)
    perc25fofv,binc,_,_ = cbs(fofhalomass,fofvprojgal,lambda x: np.percentile(x,2.5),bins=binv)

    ax1.plot(binc,medianfofv,color='darkmagenta',linewidth=2, label='FoF + False Pair Splitting')
    ax1.plot(binc,perc16fofv,color='darkmagenta',linestyle='-.')
    ax1.plot(binc,perc84fofv,color='darkmagenta',linestyle='-.')
    ax1.plot(binc,perc25fofv,color='darkmagenta',linestyle='dotted')
    ax1.plot(binc,perc975fofv,color='darkmagenta',linestyle='dotted')

    ax1.legend(loc='upper left')
    ax1.set_xlim(np.min(binc),np.max(binc))
    ax1.set_ylim(0,600)
    ax1.set_ylabel(r'$\Delta v_{\rm proj,\, gal}$ [km s$^{-1}$]')
    ax1.set_xlabel("log group halo mass")

    plt.tight_layout()
    plt.savefig("../figures/rproj_vproj_dists_halo_mass.pdf",dpi=300)
    plt.show()

    print('statistical distances (L1 norm)')
    print("G3 vs true: ", np.sum(np.abs(mediang3R-medianmockR)), np.sum(np.abs(mediang3v-medianmockv)))
    print("FoF vs true: ", np.sum(np.abs(medianfofR-medianmockR)), np.sum(np.abs(medianfofv-medianmockv)))

