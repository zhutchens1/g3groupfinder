import numpy as np
from astropy.cosmology import LambdaCDM
SPEED_OF_LIGHT=3e5

def linking_association(galaxyra,galaxydec,galaxycz,galaxygrpid,faintra,faintdec,faintcz,perp_LL,los_LL,H0,Om0,Ode0):
    galaxyra=np.array(galaxyra)
    galaxydec=np.array(galaxydec)
    galaxycz=np.array(galaxycz)
    galaxygrpid=np.array(galaxygrpid)
    faintra=np.array(faintra)
    faintdec=np.array(faintdec)
    faintcz=np.array(faintcz)
    
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) # this puts everything in "per h" units.
    faintphi = (faintra * np.pi/180.)
    fainttheta = (np.pi/2. - faintdec*(np.pi/180.))
    faintcmvg_transv = (cosmo.comoving_transverse_distance(faintcz/SPEED_OF_LIGHT).value)
    faintcmvg_los = cosmo.comoving_distance(faintcz/SPEED_OF_LIGHT).value
    galaxyphi = (galaxyra * np.pi/180.)
    galaxytheta = (np.pi/2. - galaxydec*(np.pi/180.))
    galaxycmvg_transv = cosmo.comoving_transverse_distance(galaxycz/SPEED_OF_LIGHT).value
    galaxycmvg_los = cosmo.comoving_distance(galaxycz/SPEED_OF_LIGHT).value
    
    # start making 2d arrays
    Ng = len(galaxyra)
    Nf = len(faintra)
    dlos = np.abs(galaxycmvg_los[:,np.newaxis]-faintcmvg_los)
    half_angle = np.arcsin((np.sin((galaxytheta[:,np.newaxis]-fainttheta)/2.0)**2.0 + np.sin(galaxytheta[:,np.newaxis])*np.sin(fainttheta)*np.sin((galaxyphi[:,np.newaxis]-faintphi)/2.0)**2.0)**0.5)
    dperp = (galaxycmvg_transv[:,np.newaxis]+faintcmvg_transv)*half_angle
    dperp_per_LL = dperp/perp_LL # convert to per-LL units
    dlos_per_LL = dlos/los_LL # convert to per-LL units
    delta = np.sqrt(dperp_per_LL**2. + dlos_per_LL**2.).T # note switching matrix orientation
    sel = np.where(np.logical_or(dperp_per_LL.T>1, dlos_per_LL.T>1))
    inf=999999.
    delta[sel]=inf # set delta to inf anywhere LLs are violated
    mindelta = np.min(delta,axis=1)
    argmindelta = np.argmin(delta,axis=1)
    faintgrpID = np.arange(0,Nf)+1+max(galaxygrpid) # assume all unique
    sel = np.where(mindelta<inf)#np.sqrt(2))
    faintgrpID[sel] = galaxygrpid[argmindelta][sel]
    return faintgrpID