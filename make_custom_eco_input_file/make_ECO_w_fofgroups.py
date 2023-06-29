import sys
sys.path.insert(0,'../g3algo/')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import foftools as fof
import iterativecombination as ic
import time
import os
from splitfalsepairs import split_false_pairs
from replicate_fof_groups_katie import do_katie_HAM
from linking_association import linking_association
import sys


inputfile = '.ECODR3_for_groups.csv'

def get_int_mag_giants(galmags, grpid, divider=-19.5):
    """
    Given a list of galaxy absolute magnitudes and group ID numbers,
    compute group-integrated total magnitudes. This function only
    considers giant galaxies (default <=-19.5) when integrating
    group luminosity. Dwarf-only groups return -999.

    Parameters
    ------------
    galmags : iterable
       List of absolute magnitudes for every galaxy (SDSS r-band).
    grpid : iterable
       List of group ID numbers for every galaxy.
    divider : float
        Giant galaxy - dwarf galaxy divide in magnitude units.

    Returns
    ------------
    grpmags : np array
       Array containing group-integrated magnitudes for each galaxy. Length matches `galmags`.
    """
    galmags=np.asarray(galmags)
    grpid=np.asarray(grpid)
    grpmags = np.zeros(len(galmags))
    uniqgrpid=np.unique(grpid)
    for uid in uniqgrpid:
        sel=np.where(grpid==uid)
        intsel = np.where(np.logical_and(grpid==uid, galmags<=divider))
        if len(intsel[0])==0:
            grpmags[sel]=-999. # dwarf-only group
        else:
            mags_to_int = galmags[intsel]
            totalmag = -2.5*np.log10(np.sum(10**(-0.4*mags_to_int)))
            grpmags[sel]=totalmag
    return grpmags




# get data needed for group finding
#dr2 = pd.read_csv("ECODR2_050922.csv").sort_values(by='name')
liv = pd.read_csv(inputfile).sort_values(by='name')
#corr = pd.read_csv("eco_ra_dec_051922.csv").sort_values(by='name')

# drop rs1492
#dr2 = dr2[dr2.name!='ECO13860']
liv = liv[liv.resname!='rs1492']

name = np.array(liv.name, dtype=object)
resname = np.array(liv.resname, dtype=object)
radeg = np.float64(liv.radeg)
dedeg = np.float64(liv.dedeg)    
cz = np.float64(liv.cz)
absrmag = np.array(liv.absrmag)
logmstar = np.array(liv.logmstar)
logmhi = np.log10(np.array(10**liv.logmgas/1.4))
logmbary = np.log10(10**liv.logmgas.to_numpy()/1.4 + 10**liv.logmstar.to_numpy())    

grpid = np.zeros_like(radeg)-999.
grpra = np.zeros_like(grpid)-999.
grpde = np.zeros_like(grpid)-999.
grpcz = np.zeros_like(grpid)-999.
grpn = np.zeros_like(grpid)-999.
grpabsrmag = np.zeros_like(grpid)-999.
grpgiantabsrmag = np.zeros_like(grpid)-999.
logmhvir = np.zeros_like(grpid)-999.   
logmh200 = np.zeros_like(grpid)-999.   
logmhdyn = np.zeros_like(grpid)-999.   
cenflag = np.zeros_like(grpid)-999.
logmhigrp=np.zeros_like(grpn)-999.
logmstargrp=np.zeros_like(grpn)-999.
logmbarygrp=np.zeros_like(grpn)-999.

# run FoF + compare to grp*_e16 cols
grpsel = (absrmag<=-17.33)
print(np.sum(grpsel))
ecovol = 191958.08 # (Mpc/h)^3
sep = (ecovol/(np.sum(grpsel)))**(1/3.)
tmp_ = fof.fast_fof(radeg[grpsel],dedeg[grpsel],cz[grpsel],0.07,1.1,sep)
grpid[grpsel] = tmp_
grpra[grpsel], grpde[grpsel], grpcz[grpsel] = fof.group_skycoords(radeg[grpsel],dedeg[grpsel],cz[grpsel],tmp_)

if False:
    plt.figure()
    checksel = (grpcz>3000)&(grpcz<7000)&(absrmag<=-17.33)#&(grpid>-999)
    my_grpn = fof.multiplicity_function(grpid[checksel], return_by_galaxy=False)
    checksel = (dr2.grpcz_e16>3000)&(dr2.grpcz_e16<7000)&(dr2.absrmag<=-17.33)#&(dr2.grp_e16>-999)
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

if False:
    plt.figure()
    checksel = (grpcz>3000)&(grpcz<7000)&(absrmag<=-17.33)#&(grpid>-999)
    my_grpn = fof.multiplicity_function(grpid[checksel], return_by_galaxy=False)
    checksel = (dr2.grpcz_e17>3000)&(dr2.grpcz_e17<7000)&(dr2.absrmag<=-17.33)#&(dr2.grp_e17>-999)
    kt_grpn = fof.multiplicity_function(np.array(dr2.grp_e17[checksel]), return_by_galaxy=False)
    binv=np.arange(0.5,280.5,1)
    plt.hist(my_grpn,bins=binv,log=True, label='pair split FOF groups (mine)')
    plt.hist(kt_grpn,bins=binv,log=True,histtype='step',linewidth=2,label="as published")
    plt.xlim(0,45)
    plt.xlabel("Number of Group Members")
    plt.ylabel("Number of Groups")
    plt.legend(loc='best')
    plt.show()

haloid, tmpmass_, _, _ = ic.HAMwrapper(radeg[grpsel], dedeg[grpsel], cz[grpsel], absrmag[grpsel], grpid[grpsel], ecovol)
tmpmass_ = tmpmass_ - np.log10(0.7)
for kk,hh in enumerate(haloid):
    logmhvir[np.where(grpid==hh)] = tmpmass_[kk]

haloid, tmpmass_, _, _ = do_katie_HAM(radeg[grpsel], dedeg[grpsel], cz[grpsel], absrmag[grpsel], grpid[grpsel], ecovol)
tmpmass_ = tmpmass_ - np.log10(0.7)
for kk,hh in enumerate(haloid):
    logmh200[np.where(grpid==hh)] = tmpmass_[kk]        

if False:
    plt.figure()
    plt.scatter(logmh200[checksel],dr2.logmh_e17[checksel],s=3,color='k')
    plt.plot([10,16],[10,16],color='red',zorder=0)
    plt.show()
    plt.clf()
    plt.scatter(logmh200[checksel],logmhvir[checksel],s=3,color='k')
    plt.xlabel("m200")
    plt.ylabel("m337")
    plt.show()

grpn[grpsel] = fof.multiplicity_function(grpid[grpsel],return_by_galaxy=True) 
cenflag[grpsel] = fof.get_central_flag(absrmag[grpsel],grpid[grpsel])
logmhigrp[grpsel]=ic.get_int_mass(logmhi[grpsel],grpid[grpsel]) 
logmstargrp[grpsel]=ic.get_int_mass(logmstar[grpsel],grpid[grpsel]) 
logmbarygrp[grpsel]=ic.get_int_mass(logmbary[grpsel],grpid[grpsel])
grpabsrmag[grpsel] = ic.get_int_mag(absrmag[grpsel],grpid[grpsel])
grpgiantabsrmag[grpsel] = get_int_mag_giants(absrmag[grpsel],grpid[grpsel])
logmhdyn[grpsel] = fof.dynmass(radeg[grpsel],dedeg[grpsel],cz[grpsel],grpid[grpsel],Aval=9.9,h=0.7)
logmhdyn[grpn<8]=-999.

###############################################################
# do assoc of faint dwarfs with new linking association method
faintsel = np.where(absrmag>-17.33)
grpid[faintsel] = linking_association(radeg[grpsel],dedeg[grpsel],cz[grpsel],grpid[grpsel],radeg[faintsel],dedeg[faintsel],cz[faintsel],\
                        0.07*sep, 1.1*sep, 100., 0.3, 0.7)

###############################################################
# map group metrics onto subfloor dwarfs
# need extrapolating function for halo masses of isolated sub-floor objects
# so must construct RESB-analogue and do FoF, HAM.
resbana = pd.read_csv("/srv/one/zhutchen/g3groupfinder/make_custom_eco_input_file/eco_custom_input_052523.csv")
resbana = resbana[resbana.absrmag<=-17.0]
resbanasep = (ecovol/len(resbana))**(1/3.)
resbanafofid = fof.fast_fof(resbana.radeg,resbana.dedeg,resbana.cz,0.07,1.1,resbanasep)
resbanalogmh200=np.zeros(len(resbana))
resbanalogmh337=np.zeros(len(resbana))
rba_haloid, tmpmass200, _, _ = do_katie_HAM(resbana.radeg,resbana.dedeg,resbana.cz,resbana.absrmag,resbanafofid,ecovol)
rba_haloid, tmpmass337, _, _ = ic.HAMwrapper(resbana.radeg,resbana.dedeg,resbana.cz,resbana.absrmag,resbanafofid,ecovol)
tmpmass200 = tmpmass200-np.log10(0.7)
tmpmass337 = tmpmass337-np.log10(0.7)
for kk,hh in enumerate(rba_haloid):
    resbanalogmh200[np.where(resbanafofid==hh)]=tmpmass200[kk]
    resbanalogmh337[np.where(resbanafofid==hh)]=tmpmass337[kk]
resbanagrpabsrmag = ic.get_int_mag(resbana.absrmag.to_numpy(),resbanafofid)
extrapdf = pd.DataFrame(np.array([resbanafofid,resbanagrpabsrmag,resbanalogmh200,resbanalogmh337]).T,columns=['fofid','grpMr','mh200','mhvir']).groupby('fofid').first()
extrapdf = extrapdf.sort_values(by='grpMr').tail(100)
paramsvir = np.polyfit(extrapdf.grpMr,extrapdf.mhvir,1)
params200 = np.polyfit(extrapdf.grpMr,extrapdf.mh200,1)
extrap_mhvir = lambda Mrtot: np.polyval(paramsvir,Mrtot)
extrap_mh200 = lambda Mrtot: np.polyval(params200,Mrtot)


#tmp = pd.DataFrame(np.array([grpabsrmag[grpsel],logmhvir[grpsel],logmh200[grpsel]]).T,columns=['grprmag','mhvir','mh200'])
#tmp = tmp.sort_values(by='grprmag').tail(100)
#paramsvir = np.polyfit(tmp.grprmag,tmp.mhvir,1)
#params200 = np.polyfit(tmp.grprmag,tmp.mh200,1)
#extrap_mhvir = lambda Mrtot: np.polyval(paramsvir,Mrtot)
#extrap_mh200 = lambda Mrtot: np.polyval(params200,Mrtot)

#plt.figure()
#tx=np.linspace(-25,-15,1000)
#plt.plot(tx,extrap_mhvir(tx),'k-')
#plt.plot(tmp.grprmag,tmp.mhvir,'rx')
#plt.show()

#################################
# also get skycutoffflag
fofscflag = fof.get_skycutoff_flag_RESA(radeg,dedeg,grpid)

subfloordwnames = name[faintsel]
for nn in subfloordwnames:
    sel = (name==nn)
    grpidvalue = grpid[sel]
    grpsel = (grpid==grpidvalue)&(absrmag<=-17.33)
    if np.sum(grpsel)>0:
        grpra[sel] = grpra[grpsel][0]
        grpde[sel] = grpde[grpsel][0]
        grpcz[sel] = grpcz[grpsel][0]
        grpn[sel] = grpn[grpsel][0]
        grpabsrmag[sel] = grpabsrmag[grpsel][0]
        grpgiantabsrmag[sel] = grpgiantabsrmag[grpsel][0]
        logmhvir[sel] = logmhvir[grpsel][0]
        logmh200[sel] = logmh200[grpsel][0]
        logmhdyn[sel] = logmhdyn[grpsel][0]
        cenflag[sel] = 0
        logmhigrp[sel] = logmhigrp[grpsel][0]
        logmstargrp[sel] = logmstargrp[grpsel][0]
        logmbarygrp[sel] = logmbarygrp[grpsel][0]
        fofscflag[sel] = fofscflag[grpsel][0]
    else:
        # group metrics are just galaxy properties (isolated galx below floor)
        grpra[sel] = radeg[sel]
        grpde[sel] = dedeg[sel]
        grpcz[sel] = cz[sel]
        grpn[sel] = 1
        grpabsrmag[sel] = absrmag[sel]
        grpgiantabsrmag[sel] = -999.
        logmhvir[sel] = extrap_mhvir(absrmag[sel])
        logmh200[sel] = extrap_mh200(absrmag[sel]) 
        logmhdyn[sel] = -999.
        cenflag[sel] = 1
        logmhigrp[sel] = logmhi[sel]
        logmstargrp[sel] = logmstar[sel]
        logmbarygrp[sel] = logmbary[sel]
        fofscflag[sel] = 0.

###############################
# output to file 
###############################
############################### 
outdata = np.array([name,grpid,grpra,grpde,grpcz,grpn,grpabsrmag,grpgiantabsrmag,logmhvir,logmh200,logmhdyn,cenflag,logmhigrp,logmstargrp,logmbarygrp,fofscflag,resname]).T
pd.DataFrame(outdata, columns=['name','fofgrp','fofgrpradeg','fofgrpdedeg','fofgrpcz','fofgrpn','fofgrpabsrmag','fofgrpgiantabsrmag','foflogmhvir','foflogmh200','foflogmhdyn','foffc','fofgrpmhi','logmstarfofgrp','logmbaryfofgrp','fofskycutoffflag','resname']).to_csv(".ECO_DR3_groups_only.csv",index=False)
print('success :-)')
