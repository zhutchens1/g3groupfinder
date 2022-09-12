import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
from scipy.stats import anderson
import math
from numba import njit, prange
from heapq import nsmallest
np.random.seed(46)


def group_color_gap(grpid, galmag, galur):
    """
    Return the group u-r color gap.
    Adapted from a code by Ella Castelloe.

    Parameters
    ------------------
    grpid : iterable
        List containing group ID number for each galaxy.
    galmag : iterable
        List containing group absolute r magnitude for each galaxy.
    galur : iterable
        List containing galaxy u-r color for each galaxy.

    Returns
    ------------------
    grpurcolor : np.array
        List containing color gap for each group.
        Length matches `galur` (galaxy-wise).
    """
    grpid=np.array(grpid)
    galur=np.array(galur)
    galmag=np.array(galmag)
    uniqids=np.unique(grpid)
    grpurcolor=np.zeros_like(grpid)
    masses = (galmag>0).any()
    for uid in uniqids:
        grpsel = np.where(grpid==uid)
        membermag = galmag[grpsel]
        memberur = galur[grpsel]
        Ngrp = len(grpsel[0])
        if Ngrp==1:
            grpurcolor[grpsel]=0.
        else:
            colors = memberur[np.argsort(membermag)]
            if masses:
                grpurcolor[grpsel]=(colors[-1]-colors[-2])
            else:
                grpurcolor[grpsel]=(colors[0]-colors[1])
    return grpurcolor

def group_crossing_time(galaxyra, galaxydec, galaxycz, grpid, H0=70.):
    """
    Return the group crossing time <R>/<V> in Gyr.
    Adapted from a code by Ella Castelloe

    Parameters
    ------------------
    galaxyra : iterable
        RA in decimal degrees of all galaxies.
    galaxydec : iterable
        Dec in decimal degrees of all galaxies.
    galaxycz : iterable
        Galaxy recessional velocity in km/s
    grpid : iterable
        Galaxy group ID numbers. Length matches galaxyra.

    Returns
    ------------------
    crossingtime : np.array
        Group crossing times in Gyr (length matches galaxyra).
        Zero for N=1 groups.
    """
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    grpid=np.array(grpid)
    tcross=np.zeros_like(galaxyra)
    #grpra, grpdec, grpcz = fof.group_skycoords(galaxyra,galaxydec,galaxycz,grpid)
    for uid in np.unique(grpid):
        sel = np.where(grpid==uid)
        if len(sel[0])>1:
            grpra = np.mean(galaxyra[sel])
            grpdec = np.mean(galaxydec[sel])
            grpcz = np.mean(galaxycz[sel])
            rproj = (grpcz+galaxycz[sel])/H0*np.sin(fof.angular_separation(galaxyra[sel],galaxydec[sel],grpra,grpdec)/2.)*1e3
            dv = np.abs(grpcz-galaxycz[sel])
            tcross[sel]=np.mean(rproj)/np.mean(dv)
    return tcross


def AD_test(grpcz, grpid):
    """
    Return the Anderson-Darling alpha value for a galaxy group.
    Computed only for N>5 groups (N<=5 groups return 0).
    Adapted from a code by Ella Castelloe

    Parameters
    -----------------------
    grpid : iterable
        Galaxy group ID numbers. Length is number of galaxies (not groups).
    galaxycz : iterable
        Galaxy recessional velocity in km/s

    Returns
    -----------------------
    andstat : np.array
        Anderson-Darling alpha statistic for N>5 groups.
    """
    grpid=np.array(grpid)
    galcz=np.array(grpcz)
    toosmall=5
    uniqids = np.unique(grpid)
    andstat = np.zeros_like(grpid)
    a_const = 3.6789468
    b_const = 0.1749916 # see Hou et al. (2009)
    for uid in uniqids:
        grpsel = np.where(grpid==uid)
        ngrp=len(grpsel[0])
        if ngrp>toosmall:
            A2notstar, junk, junk = anderson(galcz[grpsel], dist='norm')
            A2 = A2notstar * (1 + 0.75/ngrp + 2.25/(ngrp*ngrp))
            alpha = a_const*math.exp(-1*A2/b_const)
            andstat[grpsel]=alpha
        else:
            andstat[grpsel]=0.
    return andstat

@njit(parallel=True)
def fast_DS_test(galaxyra,galaxydec,galaxycz,grpid,niter=1000):
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxycz=np.asarray(galaxycz)
    grpid=np.asarray(grpid)
    dtor=np.pi/180.
    H0=70.
    uniqids = np.unique(grpid)
    size=11
    p_value = np.zeros_like(grpid)
    for uid in uniqids:
        grpsel=np.where(grpid==uid)
        n = len(grpsel[0])
        if n>=size:
            # get group quantities
            ras = galaxyra[grpsel]
            decs = galaxydec[grpsel]
            czs = galaxycz[grpsel]
            delta = np.zeros_like(czs)
            del_sim = np.zeros((len(czs), niter))
            sig = math.sqrt(np.sum((czs - np.mean(czs))**2)/(n - 1))
            racol = ras.reshape(n,1)
            deccol=decs.reshape(n,1)
            czcol=czs.reshape(n,1)
            angles = np.sqrt(((ras - racol)*np.cos(dtor*deccol))**2 + (decs - deccol)*(decs-deccol))
            alldists = dtor * angles * czcol / H0
            for j in prange(0, len(ras)): # can use normal range
                #angles = np.sqrt(((ras - ras[j])*np.cos(dtor*decs[j]))**2 + (decs - decs[j])*(decs-decs[j]))
                #dists = dtor * angles * czs[j] / H0
                #print(dists)
                dists = alldists[j]
                close = np.argsort(dists)[1:size] #indices of closest galaxies
                czs_loc = czs[close]    #czs of closest gals
                sig_loc = math.sqrt(np.sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
                delta[j] = (size/(sig*sig)) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)*(sig_loc - sig))
                for k in prange(0,niter): # can use normal range
                    cz_scram = np.random.permutation(czs)
                    #angles  = np.sqrt(((ras - ras[j])*np.cos(dtor*decs[j]))**2 + (decs - decs[j])*(decs-decs[j]))
                    #dists = dtor * angles * cz_scram[j] / H0
                    dists *= cz_scram[j]/czs[j]
                    close = np.argsort(dists)[1:size] #indices of closest galaxies
                    sig = np.sqrt(np.sum((cz_scram - np.mean(cz_scram))**2)/(n - 1))
                    czs_loc = cz_scram[close]    #czs of closest gals
                    sig_loc = np.sqrt(np.sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
                    del_sim[j][k] = (size/(sig*sig)) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)*(sig_loc - sig))
            DELTA=np.sum(np.sqrt(delta[0:n]))
            DELSIM=np.sum(np.sqrt(del_sim[0:n]),axis=0)
            p_value[grpsel]=np.sum(np.greater(DELSIM,DELTA))/niter
        else:
            p_value[grpsel]=0.
    return p_value

def DS_test(galaxyra, galaxydec, galaxycz, grpid, niter=500):
    """
    Return the Dressler-Schecman statistic for
    a galaxy group. Adapated from a code by
    Ella Castelloe

    Parameters
    ------------------
    galaxyra : iterable
        RA in decimal degrees of all galaxies.
    galaxydec : iterable
        Dec in decimal degrees of all galaxies.
    galaxycz : iterable
        Galaxy recessional velocity in km/s
    grpid : iterable
        Galaxy group ID numbers. Length matches galaxyra.

    Returns
    ------------------
    p_value : np.array
        DS test p-values for N>=11 groups.
    """
    galaxyra=np.asarray(galaxyra)
    galaxydec=np.asarray(galaxydec)
    galaxycz=np.asarray(galaxycz)
    grpid=np.asarray(grpid)

    H0=70.
    uniqids = np.unique(grpid)
    size=11
    p_value = np.zeros_like(grpid)
    for uid in uniqids:
        grpsel=np.where(grpid==uid)
        n = len(grpsel[0])
        if n>=size:
            # get group quantities
            ras = galaxyra[grpsel]
            decs = galaxydec[grpsel]
            czs = galaxycz[grpsel]

            delta=[0.]*len(czs)
            del_sim = np.array([[0.]*niter]*len(czs))

            #loop through each galaxy in group
            for j in range(0,len(ras)):
                angles = np.sqrt(((ras - ras[j])*np.cos((np.pi/180.)*decs[j]))**2 + (decs - decs[j])**2)
                dists = (np.pi/180.)* angles * czs[j] / H0
                #if uid==15085 and j==1: print(angles, dists)
                close = np.argsort(dists)[1:size] #indices of closest galaxies
                #if uid==15085 and j==1: print(close)
                sig = np.sqrt(np.sum((czs - np.mean(czs))**2)/(n - 1))
                #if uid==15085 and j==1: print(sig)
                czs_loc = czs[close]    #czs of closest gals
                #if uid==15085 and j==1: print(czs[close])
                sig_loc = np.sqrt(np.sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
                delta[j] = (size/sig**2) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)**2)
                #if uid==15085 and j==1: print(delta[j])
                #simulated values
                for k in range(0,niter):
                    cz_scram = np.random.permutation(czs)#scrambled(czs)
                    angles  = np.sqrt(((ras - ras[j])*np.cos((np.pi/180.)*decs[j]))**2 + (decs - decs[j])**2)
                    dists = (np.pi/180.)* angles * cz_scram[j] / H0
                    close = np.argsort(dists)[1:size] #indices of closest galaxies
                    sig = np.sqrt(np.sum((cz_scram - np.mean(cz_scram))**2)/(n - 1))
                    czs_loc = cz_scram[close]    #czs of closest gals
                    sig_loc = np.sqrt(np.sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
                    del_sim[j][k] = (size/sig**2) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)**2)
            DELTA = sum(np.sqrt(np.asarray(delta[0:n])))
            DELSIM = sum(np.sqrt(np.asarray(del_sim[0:n])))
            pv = np.sum(np.greater(DELSIM,DELTA))/niter 
            p_value[grpsel]=pv
        else:
            p_value[grpsel]=0.
    return p_value

def DS_test_ella(czs, ras, decs):
    H0 = 70 #h

    ras = np.array(ras)
    decs = np.array(decs)
    czs = np.array(czs)
    print("Ella's values:\n", czs)


    del_sim = np.array([[0.]*1000]*len(czs))
    delta = [0.] * len(czs)
    #DELSIM= [0.] * len(czs)
    #p_valueuntitled2 = [0.] * len(czs)

    n = int(np.shape(czs)[0])
    #print(n)
    size=11

    if n >= size:

        #loop through each galaxy in group
        for j in range(0,len(ras)):
            angles = np.sqrt(((ras - ras[j])*np.cos((np.pi/180.)*decs[j]))**2 + (decs - decs[j])**2)
            dists = (np.pi/180.)* angles * czs[j] / H0
            if j==1: print(angles, dists)
            close = np.argsort(dists)[1:size] #indices of closest galaxies
            if j==1: print(close)
            sig = np.sqrt(sum((czs - np.mean(czs))**2)/(n - 1))
            if j==1: print(sig)
            czs_loc = czs[close]    #czs of closest gals
            if j==1: print(czs[close])
            sig_loc = np.sqrt(sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
            delta[j] = (size/sig**2) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)**2)

            #simulated values
            for k in range(0,10):
                cz_scram = np.array(scrambled(list(czs)))
                angles  = np.sqrt(((ras - ras[j])*np.cos((np.pi/180.)*decs[j]))**2 + (decs - decs[j])**2)
                dists = (np.pi/180.)* angles * cz_scram[j] / H0
                close = np.argsort(dists)[1:size] #indices of closest galaxies
                sig = np.sqrt(sum((cz_scram - np.mean(cz_scram))**2)/(n - 1))
                czs_loc = cz_scram[close]    #czs of closest gals
                sig_loc = np.sqrt(sum((czs_loc - np.mean(czs_loc))**2)/(n - 1))
                del_sim[j][k] = (size/sig**2) * ((np.mean(czs_loc) - np.mean(czs))**2 + (sig_loc - sig)**2)


        DELTA = sum(np.sqrt(delta[0:n]))

        DELSIM = sum(np.sqrt(del_sim[0:n]))

        p_value = len(np.where(DELSIM > DELTA)[0])/100.

        return p_value


def scrambled(orig):
    dest = orig[:]
    np.random.shuffle(dest)
    return dest #return np.random.permutation(orig)


if __name__=='__main__':
    df = pd.read_csv("galcat_eco_m337_82221_1.csv")  
    #df = df[(df.absrmag<=-17.33) & (df.grpcz<7000) & (df.grpcz>3000)]

     
    # works perfectly
    #x = AD_test(df.cz, df.boundID)
    #plt.figure()
    #plt.title("AD test")
    #plt.scatter(x,df.boundADalpha)
    #plt.show()
    #df['boundADalpha_zack']=x

    #x = group_color_gap(df.boundID, df.absrmag, df.modelu_r)
    #sel = np.where(np.logical_and(x!=0, np.array(df.boundURcolorgap)==0))
    #print(len(sel[0]))
    #plt.figure()
    #plt.title("color gap")
    #plt.scatter(x, df.boundURcolorgap)
    #plt.show()
    #df['boundURcolorgap_zack']=x
   
    #x = group_crossing_time(df.radeg, df.dedeg, df.cz, df.boundID)
    #plt.figure()
    #plt.title("crossing time")
   # tx=np.linspace(0,2000,100)
    #plt.plot(tx,tx)
    #plt.scatter(x, df.boundTCross)
    #plt.show()
    #df['boundTCross_zack']=x

    starttime=time.time()
    x = fast_DS_test(np.asarray(df.radeg), np.asarray(df.dedeg), np.asarray(df.cz), np.asarray(df.boundID), niter=2500)
    print("new", time.time()-starttime)

    starttime=time.time()
    #x = DS_test(np.asarray(df.radeg), np.asarray(df.dedeg), np.asarray(df.cz), np.asarray(df.boundID), niter=1000)
    #print("orig", time.time()-starttime)


    plt.figure()
    plt.title("DS test")
    plt.scatter(x, df.boundDSpval)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.show()
    df['boundDSpval_zack']=x

    plt.figure()
    plt.hist(df['boundDSpval'][df.boundDSpval>0], bins=10)
    plt.xlabel("boundDSpval")
    plt.show()
    #print('--------')
    #grp = df[df.boundID==15085]
    #y = DS_test_ella(grp.cz, grp.radeg, grp.dedeg)
   # # df.to_csv("withzackparameters.csv",index=False)
