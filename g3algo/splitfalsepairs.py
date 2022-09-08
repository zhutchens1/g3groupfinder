import numpy as np
import foftools as fof

def split_false_pairs(galra, galde, galcz, galgroupid, groupcoords=None):
    """
    Split false-pairs of FoF groups following the algorithm
    of Eckert et al. (2017), Appendix A.
    https://ui.adsabs.harvard.edu/abs/2017ApJ...849...20E/abstract

    Parameters
    ---------------------
    galra : array_like
        Array containing galaxy RA.
        Units: decimal degrees.
    galde : array_like
        Array containing containing galaxy DEC.
        Units: degrees.
    galcz : array_like
        Array containing cz of galaxies.
        Units: km/s
    galid : array_like
        Array containing group ID number for each galaxy.
    
    Returns
    ---------------------
    newgroupid : np.array
        Updated group ID numbers.

    """
    galra = np.array(galra)
    galde = np.array(galde)
    galcz = np.array(galcz)
    galgroupid = np.array(galgroupid)
    if groupcoords is None:
        groupra,groupde,groupcz=fof.group_skycoords(galra,galde,galcz,galgroupid)
    else:
        (groupra, groupde, groupcz) = groupcoords
    groupn = fof.multiplicity_function(galgroupid, return_by_galaxy=True)
    newgroupid = np.copy(galgroupid)
    brokenupids = np.arange(len(newgroupid))+np.max(galgroupid)+100
    r75func = lambda r1,r2: 0.75*(r2-r1)+r1
    n2grps = np.unique(galgroupid[np.where(groupn==2)])
    bb=360.
    mm = (bb-0.0)/(0.0-0.12)
    for ii,gg in enumerate(n2grps):
        galsel = np.where(galgroupid==gg)
        deltacz = np.abs(np.diff(galcz[galsel])) 
        theta = fof.angular_separation(galra[galsel],galde[galsel],groupra[galsel],\
            groupde[galsel])
        rproj = theta*groupcz[galsel][0]/70.
        rproj = theta*groupcz[galsel]/70.
        grprproj = r75func(np.min(rproj),np.max(rproj))
        keepN2 = bool((deltacz<(mm*grprproj+bb)))
        if (not keepN2):
            newgroupid[galsel]=brokenupids[galsel]
        else:
            pass
    return newgroupid 
    
if __name__=='__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    eco = pd.read_csv("ECODR2.csv")
    eco = eco[(eco.absrmag<=-17.)]
    
    print(eco[['radeg','dedeg','cz']])

    psgrpid = split_false_pairs(np.array(eco.radeg), np.array(eco.dedeg),\
        np.array(eco.cz), np.array(eco.grp_e16),\
        groupcoords=None)#(np.array(eco.grpra_e16), np.array(eco.grpdec_e16), np.array(eco.grpcz_e16)))

    e16mult = np.array(fof.multiplicity_function(np.array(eco.grp_e16), return_by_galaxy=False))
    e17mult = np.array(fof.multiplicity_function(np.array(eco.grp_e17), return_by_galaxy=False))
    mymult = np.array(fof.multiplicity_function(np.array(psgrpid), return_by_galaxy=False))

    print("# N=2 groups in E16 catalog = {}".format(len(e16mult[e16mult==2.])))
    print("# N=2 groups in E17 catalog = {}".format(len(e17mult[e17mult==2])))
    print("# N=2 groups in my catalog = {}".format(len(mymult[mymult==2])))

    eco['zackN'] = np.array(fof.multiplicity_function(np.array(psgrpid), return_by_galaxy=True))
    eco['katieN'] = np.array(fof.multiplicity_function(np.array(eco.grp_e17), return_by_galaxy=True))
    sel = (eco.katieN==2) & (eco.zackN==1)
    print("groups that Katie kept but Zack split up: ")
    print(eco[sel][['name','radeg','dedeg','cz','absrmag','grp_e17']].sort_values(by='grp_e17'))

    sel = (eco.katieN==1) & (eco.zackN==2)
    print("groups that Zack kept but Katie split up: ")
    print(eco[sel][['name','radeg','dedeg','cz','absrmag','grp_e17']].sort_values(by='grp_e17'))

    if True:    
        fig, (ax,ax1)=plt.subplots(ncols=2,figsize=(12,5))
        binv = np.arange(0.5,300.5,1)
        ax.hist(e16mult,log=True, bins=binv)
        ax.set_xlim(0,5.5)
        ax1.hist(e17mult,log=True, bins=binv)
        ax1.hist(mymult,log=True, bins=binv, histtype='step', linewidth=3, color='orange')
        ax1.set_xlim(0,5.5)
        plt.show()
