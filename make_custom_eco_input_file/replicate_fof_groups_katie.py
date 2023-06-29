import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
import iterativecombination as ic
import time
import os
from splitfalsepairs import split_false_pairs

def hubble_fof(ra,dec,cz,bperp,blos,s):
    Ngalaxies = len(ra)
    ra = np.float64(ra)
    dec = np.float64(dec)
    cz = np.float64(cz)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."
    phi = (ra * np.pi/180.)
    theta =(np.pi/2. - dec*(np.pi/180.))
    dist = cz/100.
    friendship = np.zeros((Ngalaxies, Ngalaxies))

    # Compute on-sky and line-of-sight distance between galaxy pairs
    column_phi = phi[:, None]
    column_theta = theta[:, None]
    half_angle = np.arcsin((np.sin((column_theta-theta)/2.0)**2.0 + np.sin(column_theta)*np.sin(theta)*np.sin((column_phi-phi)/2.0)**2.0)**0.5)
    
    # Compute on-sky perpendicular distance
    column_dist = dist[:, None]
    dperp = (column_dist + dist) * np.sin(half_angle) # In Mpc/h
    
    # Compute line-of-sight distances
    dlos = np.abs(dist - column_dist)
    index = np.where(np.logical_and(dlos<=blos*s, dperp<=bperp*s))
    friendship[index]=1

    assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."
    return fof.collapse_friendship_matrix(friendship)

def do_katie_HAM(galra, galdec, galcz, galmag, galgrpid, volume,  inputfilename=None, outputfilename=None):
    """
    Perform halo abundance matching on a galaxy group catalog (wrapper around the C code of A.A. Berlind).

    Parameters
    -------------
    galra, galdec, galcz : iterable
        Input coordinates of galaxies (in decimal degrees, and km/s).
    galmag : iterable
        Input r-band absolute magnitudes of galaxies, or their stellar or baryonic masses. The code
        will distinguish between mags/masses, albeit variables in the code refer to 'mag' throughout.
    galgrpid : iterable
        Group ID number for every input galaxy.
        value to true if matching abundance on mass (like group-int stellar mass), for which the values are >0.
    volume : float, units in (Mpc/h)^3
        Survey volume. 
    inputfilename : string, default None
        Filename to save input HAM file for the C executable. If None, the file is removed during execution.
    outputfilename : string, default None
        Filename to save output HAM file from the C executable. If None, the file is removed during execution.

    Returns
    -------------
    haloid : np.float64 array
        ID of abundance-matched halos (matches number of unique values in `galgrpid`).
    halologmass : np.float64 array
        Log(halo mass per h) of each halo, length matches `haloid`.
    halorvir : np.float64 array
         Virial radii of each halo.
    halosigma : np.float64 array
         Theoretical velocity dispersion of each halo.
    """
    galra=np.array(galra)
    galdec=np.array(galdec)
    galcz=np.array(galcz)
    galmag=np.array(galmag)
    galgrpid=np.array(galgrpid)
    deloutfile=(outputfilename==None)
    delinfile=(inputfilename==None)
    # Prepare inputs
    grpra, grpdec, grpcz = fof.group_skycoords(galra, galdec, galcz, galgrpid)
    if (galmag<0).all():
        grpmag = ic.get_int_mag(galmag, galgrpid)
    elif (galmag>0).all():
        grpmag = -1*ic.get_int_mass(galmag, galgrpid) # need -1 to trick Andreas' HAM code into using masses.
    grprproj, grpsigma = fof.get_rproj_czdisp(galra, galdec, galcz, galgrpid)
    # Reshape them to match len grps
    uniqgrpid, uniqind = np.unique(galgrpid, return_index=True)
    grpra=grpra[uniqind]
    grpdec=grpdec[uniqind]
    grpcz=grpcz[uniqind]
    grprproj=grprproj[uniqind]
    grpsigma=grpsigma[uniqind]
    grpmag = grpmag[uniqind]
    grpN=[]
    for uid in uniqgrpid: grpN.append(len(galgrpid[np.where(galgrpid==uid)]))
    grpN=np.asarray(grpN)
    # Create input file and write to it
    if inputfilename is  None:
       inputfilename = "temporaryhaminput"+str(time.time())+".txt"
    f = open(inputfilename, 'w')
    for i in range(0,len(grpra)):
        f.write("G\t{a}\t{b}\t{c}\t{d}\t{e}\t{f}\t{g}\t{h}\n".format(a=int(uniqgrpid[i]),b=grpra[i],c=grpdec[i],d=grpcz[i],e=grpN[i],f=grpsigma[i],g=grprproj[i],h=grpmag[i]))
    f.close()
    # try to do the HAM
    if outputfilename is None: outputfilename='temporaryhamoutput'+str(time.time())+".txt"
    #try:
    hamcommand = "./massmatch Warren_200_MF.dat {} < ".format(volume)+inputfilename+" > "+outputfilename
    try:
        os.system(hamcommand)
    except:
        raise ValueError("OS call to HAM C executable failed; check input data type")
    hamfile=np.genfromtxt(outputfilename)
    haloid = np.float64(hamfile[:,0])
    halologmass = np.float64(hamfile[:,1])
    halorvir = np.float64(hamfile[:,2])
    halosigma = np.float64(hamfile[:,3])
    if deloutfile: os.remove(outputfilename)
    if delinfile: os.remove(inputfilename)
    return haloid, halologmass, halorvir, halosigma



if __name__=='__main__':
    # get data needed for group finding
    dr2 = pd.read_csv("ECODR2_050922.csv").sort_values(by='name')
    liv = pd.read_csv("ECOliving_070822.csv").sort_values(by='name')
    corr = pd.read_csv("eco_ra_dec_051922.csv").sort_values(by='name')


    radeg = np.float64(corr.radeg)
    dedeg = np.float64(corr.dedeg)    
    cz = np.float64(dr2.cz)
    absrmag = np.array(dr2.absrmag)
    logmstar = np.array(dr2.logmstar)
    logmbary = np.array(dr2.logmbary)

    grpid = np.zeros_like(radeg)-99.
    grpra = np.zeros_like(grpid)-99.
    grpde = np.zeros_like(grpid)-99.
    grpcz = np.zeros_like(grpid)-99.
    grpn = np.zeros_like(grpid)-99.
    logmh = np.zeros_like(grpid)-99.   

    # run FoF + compare to grp*_e16 cols
    grpsel = (absrmag<-17.33)
    print(np.sum(grpsel))
    ecovol = 192351.36 #191958.08 # (Mpc/h)^3
    sep = (ecovol/(np.sum(grpsel)))**(1/3.)
    tmp_ = hubble_fof(radeg[grpsel],dedeg[grpsel],cz[grpsel],0.07,1.1,sep)
    grpid[grpsel] = tmp_
    grpra[grpsel], grpde[grpsel], grpcz[grpsel] = fof.group_skycoords(radeg[grpsel],dedeg[grpsel],cz[grpsel],tmp_)

    if True:
        plt.figure()
        checksel = (grpcz>3000)&(grpcz<7000)&(absrmag<-17.33)#&(grpid>-99)
        my_grpn = fof.multiplicity_function(grpid[checksel], return_by_galaxy=False)
        checksel = (dr2.grpcz_e16>3000)&(dr2.grpcz_e16<7000)&(dr2.absrmag<-17.33)#&(dr2.grp_e16>-99)
        kt_grpn = fof.multiplicity_function(np.array(dr2.grp_e16[checksel]), return_by_galaxy=False)
        binv=np.arange(0.5,280.5,1)
        plt.hist(my_grpn,bins=binv,log=True, label='FoF groups w/ cosmology')
        plt.hist(kt_grpn,bins=binv,log=True,histtype='step',linewidth=2,label="ECO DR2 groups (as published)")
        plt.xlim(0,45)
        plt.xlabel("Number of Group Members")
        plt.ylabel("Number of Groups")
        plt.legend(loc='best')
        plt.show()

    tmp_ = split_false_pairs(radeg[grpsel], dedeg[grpsel], cz[grpsel],grpid[grpsel])
    grpid[grpsel] = tmp_ 
    grpra[grpsel], grpde[grpsel], grpcz[grpsel] = fof.group_skycoords(radeg[grpsel],dedeg[grpsel],cz[grpsel],tmp_)

    if True:
        plt.figure()
        checksel = (grpcz>3000)&(grpcz<7000)&(absrmag<-17.33)#&(grpid>-99)
        my_grpn = fof.multiplicity_function(grpid[checksel], return_by_galaxy=False)
        checksel = (dr2.grpcz_e17>3000)&(dr2.grpcz_e17<7000)&(dr2.absrmag<-17.33)#&(dr2.grp_e17>-99)
        kt_grpn = fof.multiplicity_function(np.array(dr2.grp_e17[checksel]), return_by_galaxy=False)
        binv=np.arange(0.5,280.5,1)
        plt.hist(my_grpn,bins=binv,log=True, label='pair split FOF groups (mine)')
        plt.hist(kt_grpn,bins=binv,log=True,histtype='step',linewidth=2,label="as published")
        plt.xlim(0,45)
        plt.xlabel("Number of Group Members")
        plt.ylabel("Number of Groups")
        plt.legend(loc='best')
        plt.show()

    haloid, tmpmass_, _, _ = do_katie_HAM(radeg[grpsel], dedeg[grpsel], cz[grpsel], absrmag[grpsel], grpid[grpsel], ecovol)
    tmpmass_ = tmpmass_ - np.log10(0.7)
    #tmpmass_ = np.log10(10**tmpmass_ / fof.getmhoffset(200,280,1,1,6))
    for kk,hh in enumerate(haloid):
        logmh[np.where(grpid==hh)] = tmpmass_[kk]
        
    if True:
        plt.figure()
        plt.scatter(logmh[checksel],dr2.logmh_e17[checksel],s=3,color='k')
        plt.plot([10,16],[10,16],color='red',zorder=0)
        plt.show()
        
    print('---- test group assoc -----')
    print(np.min(grpid)) 
