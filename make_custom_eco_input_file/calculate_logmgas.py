import numpy as np
import pred_loggs_dist as pgf # code from Eckert, K.+ 2015 (https://github.com/keckert7/codes)
import matplotlib.pyplot as plt
# note - edited Jul 21 2023 so that flag is
# 0 = clean HI detection, strong UL, successfully deconfused
# 1 = PGF estimate constrained by badly confused detection
# 2 = PGF estimate constrained by weak upper limit
# 3 = unconstrained PGF (missing HI data)

def calculate_logmgas_hutchens23(mhidet,mhilim,mhicorr,emhicorr_sys,emhicorr_rand,confused,modeluj,b_a,logmstar):
    mhidet = np.array(mhidet)
    mhilim = np.array(mhilim)
    mhicorr = np.array(mhicorr)
    emhicorr_sys = np.array(emhicorr_sys)
    emhicorr_rand = np.array(emhicorr_rand)
    confused = np.array(confused)
    modeluj = np.array(modeluj)
    b_a = np.array(b_a)
    logmstar = np.array(logmstar)
    combinedmhi = mhidet+mhilim # combine upper limits and detections
    combinedmhi[(combinedmhi==0)]=1. # where unobserved, set combinedmhi = 1 Msun to avoid log errors
    logmhi = np.log10(combinedmhi)
    bestlogmhi = logmhi.copy() # combine limits and detections
    typeflag = np.zeros(len(bestlogmhi))

    mhicorr[np.where(mhicorr == 0)] = np.nan # treat zero in f21corr as NaN
    notempty = np.where(~np.isnan(mhicorr))
    ngal = len(mhicorr)
    deconfokay = np.zeros(ngal,dtype=bool)
    deconfokay[notempty] = ((emhicorr_sys[notempty]/mhicorr[notempty]) < 0.25) & (mhicorr[notempty]<=mhidet[notempty]) # added second condition 7/25/2023
    goodconf = (confused == 1) & deconfokay # successfully deconfused
    badconf = (confused == 1) & ~deconfokay # unsuccessfully deconfused

    #################################################################
    #################################################################
    # sub in successfully deconfused data
    bestlogmhi[np.where(goodconf)] = np.log10(mhicorr[np.where(goodconf)])
    # use *constrained* phot-gas for unsuccessfuly-deconfused upper limit data
    #2D distribution of G/S

    modcolor = 1.140*modeluj + 0.594*(b_a) # from Eckert et al (2015) -- 1D fit with survival analysis on Mgas-to-M* limits
    photloggovers = 3.659 -0.981*modcolor # from Eckert et al (2015) volume-limited calibration, ~0.3 dex scatter
    logmhiphot = np.log10((10**photloggovers/1.4) * 10**logmstar)

    calibration = 11
    pars = pgf.getpars(calibration)
    ploton = 0
    loggsvals,p_loggs = pgf.estimategovers(modcolor,pars,ploton)
    # loggsvals = np.arange(-2,2.04,0.04) so 101 edges of 100 bins from -2 to +2
    # p_loggs is normalized to sum to 1 so is implicitly per 0.04 dex

    thresh=0.1
    badcount = np.sum((badconf) & ((1.4*mhidet/10**logmstar) > thresh))
    badind = (np.where(badconf & ((1.4*mhidet/10**logmstar) > thresh)))[0]
    renorms = np.zeros(badcount)
    for j in range(badcount):
        if badconf[badind[j]]: #renormalize probability distribution to reflect measured total
            lmhi_j = loggsvals - np.log10(1.4) + logmstar[badind[j]] # convert G/S tick marks to HI mass tick marks
            p_loggs[np.where(lmhi_j >= logmhi[badind[j]]), badind[j]] = 0 # set probability distribution to 0 above observed measured total
            renorms[j] = np.sum(p_loggs[np.where(lmhi_j < logmhi[badind[j]]),badind[j]])
            p_loggs[np.where(lmhi_j < logmhi[badind[j]]),badind[j]] = p_loggs[np.where(lmhi_j < logmhi[badind[j]]),badind[j]] / renorms[j]
            cum_p_loggs = np.cumsum(p_loggs,axis=0)
            logmhiphot[badind[j]] = np.interp(0.5,cum_p_loggs[:,badind[j]],lmhi_j) # take median of remaining distribution
        else:
            print("This should NEVER happen")
    bestlogmhi[(badind)] = logmhiphot[(badind)]
    typeflag[(badind)] = 1 #np.ones_like(logmhiphot[(badind)])
    # edited below 7/25/23: don't want to use mhicorr if it is >mhidet even if 1.4mhidet/mstar>thresh.
    # using mhidet if mhicorr is missing or bad ("bad" meaning mhicorr>mhidet)
    bestlogmhi[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & ~np.isnan(mhicorr) & (mhicorr<=mhidet))] = np.log10(mhicorr[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & ~np.isnan(mhicorr) & (mhicorr<=mhidet))])
    bestlogmhi[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & ~np.isnan(mhicorr) & (mhicorr>mhidet))] = np.log10(mhidet[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & ~np.isnan(mhicorr) & (mhicorr>mhidet))]) # mhicorr bad in this case
    bestlogmhi[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & np.isnan(mhicorr))] = np.log10(mhidet[np.where((badconf) & ~((1.4*mhidet/10**logmstar) > thresh) & np.isnan(mhicorr))])

    #################################################################
    #################################################################
    # use phot-gas for missing data
    missing = (logmhi==0) #np.isnan(logmhi)
    bestlogmhi[np.where(missing)] = logmhiphot[np.where(missing)]
    typeflag[np.where(missing)] = 3 

    #################################################################
    #################################################################
    # use *constrained* phot-gas for weak upper limit data
    weaklimit = np.zeros(ngal,dtype=bool)
    weaklimit[np.where(mhilim>0)] =  (1.4 * 10**logmhi[np.where(mhilim>0)])/(10**logmstar[np.where(mhilim>0)]) > 0.05
    # choice of 0.05 follows Eckert et al. 2016 with 1.4 for Helium correction, could make less strict
    #2D distribution of G/S
    calibration = 11
    pars = pgf.getpars(calibration)
    ploton = 0
    loggsvals,p_loggs = pgf.estimategovers(modcolor,pars,ploton)
    # loggsvals = np.arange(-2,2.04,0.04) so 101 edges of 100 bins from -2 to +2
    # p_loggs is normalized to sum to 1 so is implicitly per 0.04 dex
    limcount = np.sum(mhilim>0)
    limind = (np.where(mhilim>0))[0]
    renorms = np.zeros(limcount)
    for j in range(limcount): 
        if weaklimit[limind[j]]: #renormalize probability distribution to reflect upper limit
            lmhi_j = loggsvals - np.log10(1.4) + logmstar[limind[j]] # convert G/S tick marks to HI mass tick marks
            p_loggs[np.where(lmhi_j >= logmhi[limind[j]]), limind[j]] = 0 # set probability distribution to 0 above observed limit
            renorms[j] = np.sum(p_loggs[np.where(lmhi_j < logmhi[limind[j]]),limind[j]])
            p_loggs[np.where(lmhi_j < logmhi[limind[j]]),limind[j]] = p_loggs[np.where(lmhi_j < logmhi[limind[j]]),limind[j]] / renorms[j]
            cum_p_loggs = np.cumsum(p_loggs,axis=0)
            logmhiphot[limind[j]] = np.interp(0.5,cum_p_loggs[:,limind[j]],lmhi_j) # take median of remaining distribution
        else: # this limit is so strong, might as well treat it as a number
            logmhiphot[limind[j]] = logmhi[limind[j]] # unused assignment for clarity, strong limits already set to logmhi
    bestlogmhi[np.where(weaklimit)] = logmhiphot[np.where(weaklimit)]
    typeflag[np.where(weaklimit)] = 2
    return np.log10(1.4 * 10**bestlogmhi), typeflag

if __name__=='__main__':
    # test output
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    eco = pd.read_csv("ECOdata_051123.csv")
    test1=False
    if test1:
        logmgas = eco.logmgas.to_numpy().copy()
        typeflag = eco.logmgastype.to_numpy().copy()
        logmgas2, typeflag2 = calculate_logmgas_hutchens23(eco.mhidet, eco.mhilim, eco.mhi_corr, eco.emhi_corr_sys, eco.emhi_corr_rand, eco.confused,\
            eco.modelu_j, eco.b_a, eco.logmstar)

        print(logmgas - logmgas2)
        print(typeflag - typeflag2)

        plt.figure()
        plt.plot(logmgas, logmgas2 - logmgas, 'k.')
        plt.show()

        assert (np.abs(logmgas-logmgas2)<1e-6).all()
        assert ((typeflag - typeflag2)<1e-6).all()
        assert ((typeflag - typeflag2)>-0.001).all()
    else:
        # also check _a100 cols
        logmgas3, typeflag3 = calculate_logmgas_hutchens23(eco.mhidet_a100, eco.mhilim_a100, np.zeros(len(eco)), np.zeros(len(eco)), np.zeros(len(eco)), eco.confused_a100,\
            eco.modelu_j, eco.b_a, eco.logmstar)
        logmgas_a100 = eco.logmgas_a100.to_numpy().copy()
        typeflag_a100 = eco.logmgastype_a100.to_numpy().copy()
    
        print(logmgas_a100 - logmgas3)
        print(typeflag_a100 - typeflag3)

        plt.figure()
        plt.plot(logmgas_a100, logmgas3 - logmgas_a100, 'k.')
        plt.show()
        
        assert (np.abs(logmgas_a100-logmgas3)<1e-6).all()
        assert ((typeflag_a100 - typeflag3)<1e-6).all()
        assert ((typeflag_a100 - typeflag3)>-0.001).all()
