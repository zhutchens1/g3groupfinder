import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from astropy.stats import biweight_location
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import binned_statistic
import foftools as fof

def giantaverage(galaxyra, galaxydec, galaxycz, galaxymag, galaxygrpid, divider=-19.5):
    """
    Compute average group centers using only giant galaxies.
    """
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxymag=np.array(galaxymag)
    assert (galaxymag<0).all(), "Some abs mags are >0."
    galaxygrpid=np.array(galaxygrpid)
    giantsel = np.where(galaxymag<divider)
    giantra = galaxyra[giantsel]
    giantdec = galaxydec[giantsel]
    giantcz = galaxycz[giantsel]
    giantgrpid = galaxygrpid[giantsel]
    giantgrpra, giantgrpde, giantgrpcz = fof.group_skycoords(giantra, giantdec, giantcz, giantgrpid)
    uniqgiantgrpid, uniqsel = np.unique(giantgrpid, return_index=True)
    giantgrpra = giantgrpra[uniqsel]
    giantgrpde = giantgrpde[uniqsel]
    giantgrpcz = giantgrpcz[uniqsel]
    grpra = np.zeros_like(galaxyra)
    grpde = np.zeros_like(galaxyra)
    grpcz = np.zeros_like(galaxyra)
    for ii,gg in enumerate(uniqgiantgrpid):
        sel = np.where(galaxygrpid==gg)
        grpra[sel] = giantgrpra[ii]
        grpde[sel] = giantgrpde[ii]
        grpcz[sel] = giantgrpcz[ii]
    return grpra,grpde,grpcz

def BCG_skycoords(galaxyra, galaxydec, galaxycz, galaxymag, galaxygrpid):
    """
    Compute the group center of each galaxy's group, defined as the BCG
    position.
    """
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxymag=np.array(galaxymag)
    assert (galaxymag<0).all(), "Some abs mags are >0."
    galaxygrpid=np.array(galaxygrpid)
    dtr=np.pi/180.
    galaxyphi = galaxyra * dtr
    galaxytheta = np.pi/2. - galaxydec*dtr
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    groupra = np.zeros_like(galaxyra)
    groupdec = np.zeros_like(galaxydec)
    groupcz = np.zeros_like(galaxycz)
    for gg in np.unique(galaxygrpid):
        sel = np.where(galaxygrpid==gg)
        censel = np.argmin(galaxymag[sel])
        xcen = galaxyx[sel][censel]*galaxycz[sel][censel]
        ycen = galaxyy[sel][censel]*galaxycz[sel][censel]
        zcen = galaxyz[sel][censel]*galaxycz[sel][censel]
        czcen = np.sqrt(xcen**2.+ycen**2.+zcen**2.)
        deccen = np.arcsin(zcen/czcen)*180./np.pi
        if (ycen >=0 and xcen >=0):
            phicor = 0.0
        elif (ycen < 0 and xcen < 0):
            phicor = 180.0
        elif (ycen >= 0 and xcen < 0):
            phicor = 180.0
        elif (ycen < 0 and xcen >=0):
            phicor = 360.0
        elif (xcen==0 and ycen==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = racen # in degrees
        groupdec[sel] = deccen # in degrees
        groupcz[sel] = czcen
    return groupra, groupdec, groupcz


def logistic_skycoords(galaxyra,galaxydec,galaxycz,galaxymag,galaxygrpid,kval=1,nstar=5):
    """
    Smoothly transition between an average group center and BCG position.

    Parameters
    ---------------------
    galaxyra : iterable
        RA of grouped galaxies in decimal degrees.
    galaxydec : iterable
        Dec of grouped galaxies in degrees.
    galaxycz : iterable 
        cz of grouped galaxies in km/s.
    galaxymag : iterable
        absolute magnitude of grouped galaxies, for selecting centrals.
    galaxygrpid : iterable
        Group ID numbers by galaxy (length matches `galaxyra`).
    kval : float
        steepness parameter for sigmoid, default 1.
    nstar : float
        horizontal offset for sigmoid, default 5.

    Returns
    -------------------------
    groupra, groupdec, groupcz : np.array
        RA, Dec, and cz of group centers in deg or km/s.
        Length matches `galaxyra`.
    """
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxymag=np.array(galaxymag)
    galaxygrpid=np.array(galaxygrpid)
    dtr=np.pi/180.
    galaxyphi = galaxyra * dtr
    galaxytheta = np.pi/2. - galaxydec*dtr
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    groupra = np.zeros_like(galaxyra)
    groupdec = np.zeros_like(galaxydec)
    groupcz = np.zeros_like(galaxycz)
    for gg in np.unique(galaxygrpid):
        sel = np.where(galaxygrpid==gg)
        nmembers = len(sel[0])
        xavg = np.sum(galaxyx[sel]*galaxycz[sel])/nmembers
        yavg = np.sum(galaxyy[sel]*galaxycz[sel])/nmembers
        zavg = np.sum(galaxyz[sel]*galaxycz[sel])/nmembers
        censel = np.argmin(galaxymag[sel])
        xcen = galaxyx[sel][censel]*galaxycz[sel][censel]
        ycen = galaxyy[sel][censel]*galaxycz[sel][censel]
        zcen = galaxyz[sel][censel]*galaxycz[sel][censel]
        
        denom = 1+np.exp(-kval*(nmembers-nstar))
        xada = (xcen-xavg)/denom + xavg
        yada = (ycen-yavg)/denom + yavg
        zada = (zcen-zavg)/denom + zavg

        czada = np.sqrt(xada**2.+yada**2.+zada**2.)
        decada = np.arcsin(zada/czada)*180./np.pi
        if (yada >=0 and xada >=0):
            phicor = 0.0
        elif (yada < 0 and xada < 0):
            phicor = 180.0
        elif (yada >= 0 and xada < 0):
            phicor = 180.0
        elif (yada < 0 and xada >=0):
            phicor = 360.0
        elif (xada==0 and yada==0):
            print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
        # set up phicorrection and return phicen.
        raada=np.arctan(yada/xada)*(180/np.pi)+phicor # in degrees
        # set values at each element in the array that belongs to the group under iteration
        groupra[sel] = raada # in degrees
        groupdec[sel] = decada # in degrees
        groupcz[sel] = czada
    return groupra, groupdec, groupcz

def biweight_group_center(galaxyra,galaxydec,galaxycz,galaxygrpid):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxygrpid=np.array(galaxygrpid)
    dtr=np.pi/180.
    galaxyphi = galaxyra * dtr
    galaxytheta = np.pi/2. - galaxydec*dtr
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    xcen, ycen, zcen = biweight_location(galaxyx*galaxycz), biweight_location(galaxyy*galaxycz), biweight_location(galaxyz*galaxycz)
    czcen = np.sqrt(xcen*xcen + ycen*ycen + zcen*zcen)
    deccen = np.arcsin(zcen/czcen)*(1./dtr)
    if (ycen >=0 and xcen >=0):
        phicor = 0.0
    elif (ycen < 0 and xcen < 0):
        phicor = 180.0
    elif (ycen >= 0 and xcen < 0):
        phicor = 180.0
    elif (ycen < 0 and xcen >=0):
        phicor = 360.0
    elif (xcen==0 and ycen==0):
        print("Warning: xcen=0 and ycen=0 for group {}".format(galaxygrpid[i]))
    # set up phicorrection and return phicen.
    racen=np.arctan(ycen/xcen)*(180/np.pi)+phicor # in degrees
    return racen,deccen,czcen


def kde_skycoords(galaxyra, galaxydec, galaxycz, galaxygrpid):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxygrpid=np.array(galaxygrpid)
    ngalaxies=len(galaxyra)
    galaxyphi = galaxyra * np.pi/180.
    galaxytheta = np.pi/2. - galaxydec*np.pi/180.
    galaxyx = np.sin(galaxytheta)*np.cos(galaxyphi)
    galaxyy = np.sin(galaxytheta)*np.sin(galaxyphi)
    galaxyz = np.cos(galaxytheta)
    # Prepare output arrays
    uniqidnumbers = np.unique(galaxygrpid)
    groupra = np.zeros(ngalaxies)
    groupdec = np.zeros(ngalaxies)
    groupcz = np.zeros(ngalaxies)
    for i,uid in enumerate(uniqidnumbers):
        sel=np.where(galaxygrpid==uid)
        if len(sel[0])<4:
            groupra[sel]=np.mean(galaxyra[sel])
            groupdec[sel]=np.mean(galaxydec[sel])
            groupcz[sel]=np.mean(galaxycz[sel])
        else:
            nmembers = len(galaxygrpid[sel])
            values = np.array([galaxyra[sel],galaxydec[sel],galaxycz[sel]]).T
            #print(galaxyra[sel],galaxydec[sel],galaxycz[sel])
            #print(values)
            #dens = gaussian_kde(values,'silverman')(values)
            dens = KernelDensity().fit(values).score_samples(values)
            maxdensloc = np.argmax(dens)
            groupra[sel] = galaxyra[sel][maxdensloc] # in degrees
            groupdec[sel] = galaxydec[sel][maxdensloc] # in degrees
            groupcz[sel] = galaxycz[sel][maxdensloc]
    return groupra, groupdec, groupcz

