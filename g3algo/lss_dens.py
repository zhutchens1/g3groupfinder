# lss_dens.py
# Compute the large-scale structure density around
# each galaxy group. There are options for both 3D
# and 2D projected densities. This code is transl-
# ated from an IDL version by David Stark.
#
# Zack Hutchens - Aug. 2021
#

import numpy as np
from scipy.spatial import cKDTree
from numba import njit

def lss_dens_by_galaxy(grpid, ra, dec, cz, weights, Nnn, rarange=None,\
                       decrange=None, czrange=None, H0=70., dv=500.):
    """
    Wrapper around function `lss_dens`. This wrapper computes the nndens 
    for all galaxies in group catalog and returns the densities in galaxy-
    wise arrays (e.g. duplicated information for every group member). This
    wrapper returns both 2D and 3D LSS densities.

    Parameters
    ----------------
    grpid : iterable
        List of galaxy group ID numbers (length = # of galaxies in all groups).
    ra : iterable
        List of galaxy RA values in decimal degrees; length must match grpid.
    dec : iterable
        List of galaxy declination values in degrees; length must match grpid.
    cz : iterable
        List of galaxy recessional velocities in km/s; length must match grpid.

    For other input parameters, see the documentation for `lss_dens`.

    Returns
    ----------------
    nndens : np.array
        3D group nearest-neighbor densities.
    edgeflag : np.array
        3D group survey edge flags.
    nndens_2d : np.array
        2D group nearest-neighbor densities.
    edgeflag_2d : np.array
        2D group survey edge flags.
    edgescale_2d : np.array
        Correction for `nndens_2d` based on overlap with survey sky area.
    """
    grpid=np.asarray(grpid)
    ra=np.asarray(ra)
    dec=np.asarray(dec)
    cz=np.asarray(cz)
    weights=np.asarray(weights)
    uniqgrpid, uniqsel = np.unique(grpid, return_index=True)
    ra = ra[uniqsel]
    dec = dec[uniqsel]
    cz = cz[uniqsel]
    weights = weights[uniqsel]
    grp_nndens, grp_flag = lss_dens(ra, dec, cz, weights, Nnn, False,rarange,\
            decrange,czrange,H0,dv)
    grp_nndens_2d, grp_flag_2d, grp_scale_2d = lss_dens(ra,dec,cz,weights,Nnn,True,rarange,\
            decrange,czrange,H0,dv)

    # remap onto output arrays
    outshape=len(grpid)
    nndens=np.zeros(outshape)
    edgeflag=np.zeros(outshape)
    nndens_2d=np.zeros(outshape)
    edgeflag_2d=np.zeros(outshape)
    edgescale_2d=np.zeros(outshape)
    for i,uid in enumerate(uniqgrpid):
        sel=np.where(grpid==uid)
        nndens[sel]=grp_nndens[i]
        edgeflag[sel]=grp_flag[i]
        nndens_2d[sel]=grp_nndens_2d[i]
        edgeflag_2d[sel]=grp_flag_2d[i]
        edgescale_2d[sel]=grp_scale_2d[i]
    return nndens, edgeflag, nndens_2d, edgeflag_2d, edgescale_2d


def lss_dens(ra, dec, cz, weights, Nnn, projected=False, rarange=None,\
             decrange=None, czrange=None, H0=70., dv=500.):
    """
    Compute the large-scale structure densities for a catalog of galaxy groups.
    
    Parameters
    ------------------
    ra : iterable
        Right-ascension of galaxy groups in decimal degrees.
    dec : iterable
        Declination of galaxy groups in decimal degrees.
    cz : iterable
        Recessional velocities of groups in km/s
    weights : iterable
        Array containing weights for groups, length should match `ra`.
        Typically this will be the group halo mass, but could also be,
        for example, the group total stellar mass or an array of 1's.
        These should be expressed in log10 space.
    Nnn : int
        Nth closest neighbor used to determine densities.
    projected : bool, default False
        If True, compute projected densities. Otherwise compute 3D densities.
    rarange : 2-element tuple, default None
        range of right-ascensions over which to calculate densities
    decrange : 2-element tuple, default None
        range of declinations over which to calculate densities
    czrange : 2-element tuple, default None
        range of velocities over which to calculate densities
    H0 : float
        Hubble constant, default 70 km/s.
    dv : 2-element tuple, default 500 km/s
        If projected=True, this is the velocity range over which to search for neighbors.

    Returns
    -------------------
    nndens : iterable
        Array containing nearest-neighbor density for each group, with units of log(Msun/Mpc^3)
        given the provided value of H0.
    edgeflag : iterable
        flag indicating whether the volume relevant for that group overlapped the survey boundary
        note: only returned if rarange, decrange, and czrange are not None
    edgescale : iterable
        if projected=True, fraction of nearest-neighbor volume contained with official survey 
    """
    if projected:
        return lss_dens_2D(ra,dec,cz,weights,Nnn,rarange,decrange,czrange,H0,dv)
    else:
        return lss_dens_3D(ra,dec,cz,weights,Nnn,rarange,decrange,czrange,H0,dv)

############ 2D functions ##################
def angular_separation(ra1,dec1,ra2,dec2):
    """
    Compute the angular separation bewteen two lists of galaxies using the Haversine formula.
    
    Parameters
    ------------
    ra1, dec1, ra2, dec2 : array-like
       Lists of right-ascension and declination values for input targets, in decimal degrees. 
    
    Returns
    ------------
    angle : np.array
       Array containing the angular separations between coordinates in list #1 and list #2, as above.
       Return value expressed in radians, NOT decimal degrees.
    """
    phi1 = ra1*np.pi/180.
    phi2 = ra2*np.pi/180.
    theta1 = np.pi/2. - dec1*np.pi/180.
    theta2 = np.pi/2. - dec2*np.pi/180.
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2.0)**2.0 + np.sin(theta1)*np.sin(theta2)*np.sin((phi2 - phi1)/2.0)**2.0))

def lss_dens_2D(ra,dec,cz,weights,Nnn,rarange,decrange,czrange,H0,dv):
    dtor=np.pi/180.
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    cz = np.asarray(cz)
    vsep = np.abs(cz-cz[:,None])
    angsep = angular_separation(ra, dec, ra[:,None], dec[:,None]) # radians
    angsep[np.where(vsep>dv)]=9e9
    index = np.argsort(angsep,axis=1)
    neighboridx = index[:,0:Nnn+1]
    nth_dist_index = neighboridx[:,-1]
    nth_dist = np.take_along_axis(angsep, nth_dist_index[:,None], axis=1).flatten()
    totalmass = np.sum(10**weights[neighboridx],axis=1)
    maxradius=nth_dist*cz/H0
    area = np.pi*maxradius*maxradius
    nndens = np.log10(totalmass/area) # log (Msun/Mpc^2)
    edgeflag, edgescale = flag_edge_effects_2D(ra,dec,cz,maxradius,rarange,decrange,czrange,H0)
    return nndens, edgeflag, edgescale
    

def flag_edge_effects_2D(ra,dec,cz,nthdist,rarange,decrange,czrange,H0):
    edgeflag=np.zeros_like(ra)
    dtor=np.pi/180.
    # flag for edge effects (left side)
    edgesep=angular_separation(ra,dec,rarange[0]+np.zeros_like(ra),dec)*cz/H0
    edgeflag[np.where(edgesep<nthdist)]=1.
    # flag (right side)
    edgesep=angular_separation(ra,dec,rarange[1]+np.zeros_like(ra),dec)*cz/H0
    edgeflag[np.where(edgesep<nthdist)]=1.
    # flag (top)
    edgesep=angular_separation(ra,dec,ra,decrange[1]+np.zeros_like(dec))*cz/H0
    edgeflag[np.where(edgesep<nthdist)]=1.
    # flag (bottom)
    edgesep=angular_separation(ra,dec,ra,decrange[0]+np.zeros_like(dec))*cz/H0
    edgeflag[np.where(edgesep<nthdist)]=1.
    print(1/dtor)
    print(1/cz)
    edgescale = edgecor(ra,dec,cz,nthdist*H0/(dtor*cz),edgeflag,rarange,decrange)
    return edgeflag, edgescale

    
    

################ 3D functions ###################
def lss_dens_3D(ra,dec,cz,weights,Nnn,rarange,decrange,czrange,H0,dv):
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    cz = np.asarray(cz)
    phi = ra*np.pi/180.
    theta = np.pi/2. - dec*np.pi/180.
    xx = (cz/H0)*np.sin(theta)*np.cos(phi)
    yy = (cz/H0)*np.sin(theta)*np.sin(phi)
    zz = (cz/H0)*np.cos(theta)
    coords = np.array([xx,yy,zz]).T
    tree = cKDTree(coords)
    nndist, nnind = tree.query(coords,k=Nnn+1) # Nnn+1 = self + Nnn neighbors
    nndist=nndist[:,0:] # keep self match
    nnind=nnind[:,0:] # keep self match

    totalmass = (np.sum(10**weights[nnind],axis=1))
    maxradius = np.amax(nndist,axis=1)
    volume = (4/3.)*np.pi*maxradius*maxradius*maxradius
    nndens = np.log10(totalmass/volume)
    if rarange is not None:
        edgeflag = flag_edge_effects_3D(ra,dec,cz,xx,yy,zz,maxradius,rarange,decrange,czrange,H0,dv)
        return nndens, edgeflag
    else:
        return nndens


def flag_edge_effects_3D(ra,dec,cz,groupx,groupy,groupz,nthdist,rarange,decrange,czrange,H0,dv):
    edgeflag = np.zeros_like(ra)
    edgescale = np.zeros_like(ra)
    # flag for edge effects (distance from top)
    dtor = np.pi/180.
    edgera = ra
    edgedec = np.zeros_like(edgera)+decrange[1]
    edgedist = cz/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    # flag for edge effects (distance from bottom)
    edgera = ra
    edgedec = np.zeros_like(edgera)+decrange[0]
    edgedist = cz/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    # flag for edge effects (distance from left)
    edgedec = dec
    edgera = np.zeros_like(dec)+rarange[0]
    edgedist = cz/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    # flag for edge effects (distance from right)
    edgedec = dec
    edgera = np.zeros_like(dec)+rarange[1]
    edgedist = cz/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    # flag for edge effects (distance from front)
    edgedec = dec
    edgera = ra
    edgedist = np.zeros_like(ra)+czrange[0]/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    # flag for edge effects (distance from back)
    edgedec = dec 
    edgera = ra
    edgedist = np.zeros_like(ra)+czrange[1]/H0
    edgex = edgedist * np.sin((90-edgedec)*dtor) * np.cos(edgera*dtor)
    edgey = edgedist * np.sin((90-edgedec)*dtor) * np.sin(edgera*dtor)
    edgez = edgedist * np.cos((90-edgedec)*dtor)
    edgedist = np.sqrt((groupx-edgex)*(groupx-edgex) + (groupy-edgey)*(groupy-edgey) + (groupz-edgez)*(groupz-edgez))
    edgeflag[np.where(edgedist<nthdist)]=1.

    return edgeflag

############ function to compute edgescale ########
@njit
def edgecor(ra,dec,cz,maxradiusdeg,edgeflag,rarange,decrange,nsimpoints=10000):
    output=np.zeros_like(ra)+0.
    minra = min(rarange)
    maxra = max(rarange)
    mindec = min(decrange)
    maxdec = max(decrange)
    for i in range(0,len(ra)):
        if True:
            randomra = np.random.rand(nsimpoints)*2*maxradiusdeg[i]-maxradiusdeg[i]+ra[i]
            randomdec = np.random.rand(nsimpoints)*2*maxradiusdeg[i]-maxradiusdeg[i]+dec[i]
            incircle = ((randomra-ra[i])**2.+(randomdec-dec[i])**2.)<(maxradiusdeg[i]**2.)
            insurvey = ((randomra>minra) & (randomra<maxra) & (randomdec>mindec) & (randomdec<maxdec))
            if np.sum(incircle)==0 or np.sum(insurvey)==0:
                output[i]=0.
            else:
                output[i] = np.sum(incircle)/(np.sum(np.logical_and(incircle,insurvey)))
    return output

