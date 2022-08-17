import numpy as np
import foftools as fof
import iterativecombination as ic
from astropy.cosmology import LambdaCDM

def g3groupfinder_luminosity(radeg,dedeg,cz,absrmag,dwarfgiantdivide,fof_bperp=0.07,fof_blos=1.1,fof_sep=None, volume=None, center_mode='average',\
                 iterative_giant_only_groups=False, n_bootstraps=10000, rproj_fit_guess=None, rproj_fit_params = None, rproj_fit_multiplier=None,\
                 vproj_fit_guess = None, vproj_fit_params = None, vproj_fit_multiplier=None, gd_rproj_fit_guess=None, gd_rproj_fit_params = None,\
                 gd_rproj_fit_multiplier=None, gd_vproj_fit_guess=None, gd_vproj_fit_params = None, gd_vproj_fit_multiplier=None, gd_fit_bins=None,\
                 ic_center_mode='arithmetic', ic_decision_mode='centers',H0=100., Om0=0.3, Ode0=0.7, showplots=False, saveplotspdf=False):
    """
    Identify galaxy groups in redshift space using the RESOLVE-G3 algorithm (Hutchens et al. 2022).

    Parameters
    -------------------
    radeg : array_like
        Right ascension of input galaxies in decimal degrees.
    dedeg : array_like
        Declination of input galaxies in decimal degrees.
    cz : array_like
        Recessional velocities of input galaxies in decimal degrees.
    absrmag : array_like
        Absolute magnitude for galaxies, used to select giants vs. dwarfs.
    dwarfgiantdivide : float
        Value that will divide giants and dwarfs.
    fof_bperp : float
        Perpendicular FoF linking length, default 0.07.
    fof_blos : float
        Line-of-sight FoF linking length, default 1.1.
    fof_sep : float
        Mean galaxy separation used for FoF. Should be expressed in units of (Mpc/h) with 
        h corresponding to the `H0` argument (i.e. use h=0.7 if setting H0=70.). If None
        (default), fof_sep will be determined using the number of galaxies and `volume`.
    volume : float
        Group finding volume in (Mpc/h)^3 with h corresponding to the `H0` argument, default
        None. This argument is unnecessary if fof_sep is provided. `fof_sep` and `volume`
        cannot both be `None`.
    center_mode : str
        Specifies how group centers for giant-hosting groups should be computed when iteratively
        combining giant-only FoF groups, or associating dwarfs to giant-only groups. 
        Can be 'average' or 'BCG'.
    iterative_giant_only_groups : bool 
        If False (default), giant-only groups are determined with a single run of FoF.
        If True, giant-only groups are determined iteratively, starting with FoF and refining
        based on iteratively-updated group boundaries.
    n_bootstraps : int
        Number of bootstraps to perform when computing errors on medians, default 10,000.
    rproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting rproj,gal vs. N_giants.
    rproj_fit_params : iterable
        Parameters to use when associating dwarfs and/or iteratively combining giant-only groups.
        If this parameter is passed, then the fit to rproj,gal vs. N_giants is not performed.
    rproj_fit_multiplier : float
        Scalar multiplier for rproj_fit.
    vproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting rproj,gal vs. N_giants.
    vproj_fit_params : iterable
        Parameters to use when associating dwarfs and/or iteratively combining giant-only groups.
        If this parameter is passed, then the fit to vproj,gal vs. N_giants is not performed.
    vproj_fit_multiplier : float
        Scalar multiplier for vproj_fit.
    gd_rproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting gdrproj,gal vs. Ltot.
    gd_rproj_fit_params : iterable
        Parameters to use when iteratively combining dwarf-only seed groups.
        If this parameter is passed, then the fit to gdrproj,gal vs. Ltot is not performed.
    gd_rproj_fit_multiplier : float
        Scalar multiplier of gd_rproj_fit for use in dwarf-only group finding.
    gd_vproj_fit_guess : iterable
        Guess supplied to scipy.optimize.curve_fit when fitting gd_vproj,gal vs. Ltot.
    gd_vproj_fit_params : iterable
        Parameters to use for iterative combination dwarf-only groups.
        If this parameter is passed, then the fit to gd_vproj,gal vs. N_giants is not performed.
    gd_vproj_fit_multiplier : float
        Scalar multiplier of gd_vproj_fit for use in dwarf-only group finding.
    gd_fit_bins : iterable
        Array of bin edges for binning and fitting properties of giant+dwarf groups prior to
        dwarf-only group finding. 
    ic_center_mode : str
        Mode of computing group center in dwarf-only group finding mode (default `arithmetic`).
    ic_decision_mode : str
        Mode of deciding whether to merge giant-only or dwarf-only seed groups. Default `centers`, which
        evaluates whether seed group centers are close enough. If `allgalaxies`, all galaxies
        must within the specified boundaries.
    H0 : float
        z=0 Hubble constant in units of (km/s)/Mpc, default 100.0. Return parameters will
        be consistent with this choice.
    showplots : False
        If True, plots are rendered using matplotlib at each group finding step.
    saveplotspdf : False
        If True, rendered plots will be saved in a ./figures/ subfolder.

    Returns
    -------------------------
    g3grpid : np.array
        Group ID number for each galaxy from G3 algorithm.
    g3ssid : np.array
        Group substructure ID number for each galaxy. Equals `g3grpid` if 
        `iterative_giant_only_groups` is set to `False`.
    fof_sep : float
        Mean galaxy separation in FoF, if volume was provided, in units of Mpc/h
        with h matching H0 parameter. 
    rproj_bestfit : np.array
        Best-fitting values to rproj,gal vs. Ngiants, matches `rproj_fit_params` if provided.
    rproj_bestfit_err : np.array
        Errors on best-fitting values to rproj,gal vs. Ngiants, None if `rproj_fit_params` was provided.
    vproj_bestfit : np.array
        Best-fitting values to vproj,gal vs. Ngiants, matches `vproj_fit_params` if provided.
    vproj_bestfit_err : np.array
        Errors on best-fitting values to vproj,gal vs. Ngiants, None if `vproj_fit_params` was provided.
    gd_rproj_bestfit : np.array
        Best-fitting values to rproj,gal vs. Ltot, matches `gd_rproj_fit_params if provided.
    gd_rproj_bestfit_err : np.array
        Best-fitting values to rproj,gal vs. Ltot, None if `gd_rproj_fit_params was provided.
    gd_vproj_bestfit : np.array
        Best-fitting values to vproj,gal vs. Ltot, matches `gd_vproj_fit_params if provided.
    gd_vproj_bestfit_err : np.array
        Best-fitting values to vproj,gal vs. Ltot, None if `gd_vproj_fit_params was provided.
    """
    ### prepare arrays ---------------------------- #
    radeg=np.array(radeg)
    dedeg=np.array(dedeg)
    cz=np.array(cz)
    absrmag=np.array(absrmag)
    g3grpid = np.zeros_like(radeg)-99.
    g3ssid = np.zeros_like(radeg)-99.
    cosmo = LambdaCDM(H0=H0,Om0=Om0,Ode0=Ode0)
    SPEED_OF_LIGHT=2.998e+5
    ### giant-only FoF ----------------- # 
    giantsel = (absrmag<dwarfgiantdivide)
    if fof_sep is not None:
        giantfofid = fof.fast_fof(radeg[giantsel],dedeg[giantsel],cz[giantsel],fof_bperp,fof_blos,fof_sep)
    else:
        fof_sep = (volume/np.sum(giantsel))**(1/3.)
        giantfofid = fof.fast_fof(radeg[giantsel],dedeg[giantsel],cz[giantsel],fof_bperp,fof_blos,fof_sep)
    g3grpid[giantsel] = giantfofid

    ### if values not passed, fit rproj and vproj vs. N_giants
    if (rproj_fit_params is None) or (vproj_fit_params is None):
        if center_mode='average':
            giantgrpra, giantgrpdec, giantgrpcz = fof.group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], giantfofid)
        elif center_mode='BCG':
            giantgrpra, giantgrpdec, giantgrpcz = fof.BCG_center(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel], giantfofid)
        relvel = np.abs(giantgrpcz - cz[giantsel])
        #relprojdist = (giantgrpcz + cz[giantsel])/H0 * np.sin(ic.angular_separation(giantgrpra, giantgrpdec, radeg[giantsel], dedeg[giantsel])/2.0)
        grp_ctd = cosmo.comoving_transverse_distance(giantgrpcz/SPEED_OF_LIGHT)
        gia_ctd = cosmo.comoving_transverse_distance(cz[giantsel]/SPEED_OF_LIGHT)
        relprojdist = (grp_ctd+gia_ctd)*np.sin(ic.angular_separation(giantgrpra, giantgrpdec, radeg[giantsel], dedeg[giantsel])/2.0)
        giantgrpn = fof.multiplicity_function(giantfofid, return_by_galaxy=True)
        uniqgiantgrpn, uniqindex = np.unique(giantgrpn, return_index=True)
        keepcalsel = np.where(uniqgiantgrpn>1)
        median_relprojdist = np.array([np.median(relprojdist[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        median_relvel = np.array([np.median(relvel[np.where(giantgrpn==sz)]) for sz in uniqgiantgrpn[keepcalsel]])
        rproj_median_error = np.std(np.array([sbs(relprojdist[np.where(giantgrpn==sz)], n_bootstraps, np.median, kwargs=dict({'axis':1 })) for sz in uniqgiantgrpn[keepcalsel]]), axis=1)
        dvproj_median_error = np.std(np.array([sbs(relvel[np.where(giantgrpn==sz)], n_bootstraps, np.median, kwargs=dict({'axis':1})) for sz in uniqgiantgrpn[keepcalsel]]), axis=1)
    if rproj_fit_params is None:    
        rproj_bestfit, rproj_bestfit_cov = curve_fit(giantmodel, uniqgiantgrpn[keepcalsel], median_relprojdist, sigma=rproj_median_error, p0=rproj_fit_guess)
        rproj_bestfit_err = np.sqrt(np.diag(rproj_bestfit_cov))
    else:
        rproj_bestfit = np.array(rproj_fit_params)
        rproj_bestfit_err = np.zeros(2)*1.
    if vproj_fit_params is None:
        vproj_bestfit, vproj_bestfit_cov  = curve_fit(giantmodel, uniqecogiantgrpn[keepcalsel], median_relvel, sigma=dvproj_median_error, p0=vproj_fit_guess)
        vproj_bestfit_err = np.sqrt(np.diag(vproj_bestfit_cov))
    else:
        vproj_bestfit = np.array(vproj_fit_params)
        vproj_bestfit_err = np.zeros(2)*1.
    
    rproj_boundary = lambda Ngiants: rproj_fit_multiplier*giantmodel(Ngiants, *rproj_bestfit)
    vproj_boundary = lambda Ngiants: vproj_fit_multiplier*giantmodel(Ngiants, *vproj_bestfit)
    
    ### if requested, merge giant-only FoF groups through iterative combination
    if iterative_giant_only_groups:
        revisedgiantgrpid = iterative_combination_giants(radeg[giantsel],dedeg[giantsel],cz[giantsel],giantfofid,rproj_boundary,vproj_boundary,ic_decision_mode,H0)
        g3grpid[giantsel] = revisedgiantgrpid
    else:
        pass

    ### associate dwarfs to giant-only groups
    dwarfsel = (absrmag<dwarfgiantdivide)
    if center_mode='average':
        giantgrpra, giantgrpdec, giantgrpcz = fof.group_skycoords(radeg[giantsel], dedeg[giantsel], cz[giantsel], giantfofid)
    elif center_mode='BCG':
        giantgrpra, giantgrpdec, giantgrpcz = fof.BCG_center(radeg[giantsel], dedeg[giantsel], cz[giantsel], absrmag[giantsel], giantfofid)
    giantgrpn = fof.multiplcitiy_function(g3grpid[giantsel],return_by_galaxy=True)
    dwarfassocid, _ = fof.fast_faint_assoc(radeg[dwarfsel],dedeg[dwarfsel],cz[dwarfsel],giantgrpra,giantgrpdec,giantgrpcz,g3grpid[giantsel],\
        rproj_boundary(giantgrpn),vproj_boundary(giantgrpn))
    g3grpid[dwarfsel]=dwarfassocid

    ### if values not passed, fit rproj and vproj for giants+dwarfs vs. Ltot
    if (gd_rproj_fit_params is None) or (gd_vproj_fit_params is None):
        gdgrpn = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
        gdsel = np.logical_not(np.logical_or(g3grpid==-99., ((gdgrpn==1) & (absrmag>dwargiantdivide)))) 
        gdgrpra, gdgrpdec, gdgrpcz = fof.group_skycoords(radeg[gdsel], dedeg[gdsel], cz[gdsel], g3grpid[gdsel])
        gdrelvel = np.abs(gdgrpcz - cz[gdsel])
        ctd1 = cosmo.comoving_transverse_distance(gdgrpcz/SPEED_OF_LIGHT)
        ctd2 = cosmo.comoving_transverse_distance(cz[gdsel]/SPEED_OF_LIGHT)
        gdrelprojdist = (ctd1 + ctd2) * np.sin(ic.angular_separation(gdgrpra, gdgrpdec, radeg[gdsel], dedeg[gdsel])/2.0)
        #gdrelprojdist = (gdgrpcz + cz[gdsel])/H0 * np.sin(ic.angular_separation(gdgrpra, gdgrpdec, radeg[gdsel], dedeg[gdsel])/2.0)
        gdn = gdgrpn[gdsel]
        gdtotalmag = ic.get_int_mag(absrmag[gdsel], g3grpid[gdsel])
        binsel = np.where(np.logical_and(gdn>1, gdtotalmag>np.min(gd_fit_bins))) # test here
        gdmedianrproj, magbincenters, agbinedges, jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], np.median, bins=gd_fit_bins)
        gdmedianrproj_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelprojdist[binsel], sigmarange, bins=gd_fit_bins)
        gdmedianrelvel, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], np.median, bins=gd_fit_bins)
        gdmedianrelvel_err, jk, jk, jk = center_binned_stats(gdtotalmag[binsel], gdrelvel[binsel], sigmarange, bins=gd_fit_bins)
        nansel = np.isnan(gdmedianrproj)
    if (gd_rproj_fit_params is None):
        gd_rproj_bestfit, gd_rproj_cov=curve_fit(decayexp, magbincenters[~nansel], gdmedianrproj[~nansel], p0=gd_rproj_fit_guess)
        gd_rproj_bestfit_err = np.sqrt(np.diag(gd_rproj_cov))
    else:
        gd_rproj_bestfit = np.array(gd_rproj_fit_params)
        gd_rproj_bestfit_err = np.zeros(len(gd_rproj_fit_params))*1.
    if (gd_vproj_fit_params is None):
        gd_vproj_bestfit, gd_vproj_cov=curve_fit(decayexp, magbincenters[~nansel], gdmedianvproj[~nansel], p0=gd_vproj_fit_guess)
        gd_vproj_bestfit_err = np.sqrt(np.diag(gd_vproj_cov))
    else:
        gd_vproj_bestfit = np.array(gd_vproj_fit_params)
        gd_vproj_bestfit_err = np.zeros(len(gd_vproj_fit_params))*1.
    rproj_for_iteration = lambda M: gd_rproj_fit_multiplier*decayexp(M, *gd_rproj_bestfit)
    vproj_for_iteration = lambda M: gd_vproj_fit_multiplier**decayexp(M, *gd_vproj_bestfit)

    ### --------- iterative combination to make dwarf-only groups
    assert (g3grpid[(absrmag<=dwarfgiantdivide)]!=-99.).all(), "Not all giants are grouped." 
    grpnafterassoc = fof.multiplicity_function(g3grpid, return_by_galaxy=True)
    _ungroupeddwarf_sel = (absrmag>dwarfgiantdivde) & (grpnafterassoc==1)    
    itassocid = ic.iterative_combination(radeg[_ungroupeddwarf_sel], dedeg[_ungroupeddwarf_sel], cz[_ungroupeddwarf_sel], absrmag[_ungroupeddwarf_sel],\
                                           rproj_for_iteration, vproj_for_iteration, starting_id=np.max(ecog3grpid)+1, centermethod=ic_center_mode, decisionmode=ic_decision_mode)
    g3grpid[_ungroupeddwarf_sel]=itassocid
    ### ------------  return quantities
    return g3grpid, g3ssid, fof_sep, rproj_bestfit, rproj_bestfit_err, vproj_bestfit, vproj_bestfit_err 
