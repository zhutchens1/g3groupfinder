import sys
sys.path.insert(0,'../g3algo/')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import foftools as fof
from splitfalsepairs import split_false_pairs
import iterativecombination as ic
from replicate_fof_groups_katie import do_katie_HAM
from scipy.interpolate import interp1d
from linking_association import linking_association


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

resinputfile = '.RESOLVEDR4_for_groups.csv'
ecoinputfile = '.ECO_csv_for_RESOLVE_FoF.csv'

res = pd.read_csv(resinputfile).set_index('name')
res = res[res.index!='rs1492']
#res = res.set_index('name').drop(['grp','grpradeg','grpdedeg','grpcz','grpn','grpabsrmag'],axis=1)

# copy ECO DR3 group info into RESOLVE-A
eco = pd.read_csv(ecoinputfile)
maxECOID = np.max(eco.fofgrp)
#eco = eco[eco.absrmag<=-17.0] # commented out and moved below 4/24/23 - need subfloor galax for RESOLVE
eco_volume = 191958.08 # (Mpc/h)^3
resa_in_eco = eco[(eco.resname!='notinresolve')].set_index('resname')
resa_in_eco = resa_in_eco[['fofgrp','fofgrpradeg','fofgrpdedeg','fofgrpcz','fofgrpn','fofgrpabsrmag','fofgrpgiantabsrmag','foflogmhvir','foflogmh200','fofgrpmhi','logmstarfofgrp','logmbaryfofgrp','foflogmhdyn','fofskycutoffflag']]
#print(resa_in_eco)
#print('======================')
res = res.join(resa_in_eco)
#print(res[['absrmag','fofgrp','fofgrpn','fofgrpabsrmag','fofgrpgiantabsrmag']])
#print('--------------------')

# do group finding for RESOLVE-B and fill in the rest of the arrays
resb_groupsel = (res.f_b==1)&(res.absrmag<=-17.0)
resb = res[resb_groupsel]

sep = (eco_volume/len(eco[eco.absrmag<-17.]))**(1/3.)
resbfofid = fof.fast_fof(resb.radeg, resb.dedeg, resb.cz, 0.07, 1.1, sep)
resbfofid = split_false_pairs(resb.radeg,resb.dedeg,resb.cz,resbfofid)
resbgrpra, resbgrpde, resbgrpcz = fof.group_skycoords(np.array(resb.radeg),\
        np.array(resb.dedeg), np.array(resb.cz), np.array(resbfofid))
resbgrpn = fof.multiplicity_function(resbfofid, return_by_galaxy=True)
resbgrpmhi = ic.get_int_mass(np.array(np.log10(10**resb.logmgas/1.4)), resbfofid)
resblogmstargrp = ic.get_int_mass(resb.logmstar.to_numpy(), resbfofid)
resblogmbarygrp = ic.get_int_mass(np.log10(10**resb.logmstar.to_numpy() + 10**resb.logmgas.to_numpy()/1.4), resbfofid)
 
# do ECO group finding to Mr = -17.0 and get HAM curve
eco = eco[eco.absrmag<=-17.0] #
ecofofid = fof.fast_fof(eco.radeg,eco.dedeg,eco.cz,0.07,1.1,sep)
ecologmh200=np.zeros_like(ecofofid)
haloid, tmpmass, _, _ = do_katie_HAM(eco.radeg,eco.dedeg,eco.cz,eco.absrmag,ecofofid,eco_volume)
tmpmass = tmpmass - np.log10(0.7)
for kk,hh in enumerate(haloid):
    ecologmh200[np.where(ecofofid==hh)]=tmpmass[kk]

ecologmhvir=np.zeros_like(ecofofid)
haloid, tmpmass, _, _ = ic.HAMwrapper(eco.radeg,eco.dedeg,eco.cz,eco.absrmag,ecofofid,eco_volume)
tmpmass = tmpmass - np.log10(0.7)
for kk,hh in enumerate(haloid):
    ecologmhvir[np.where(ecofofid==hh)]=tmpmass[kk]

ecogrpabsrmag = ic.get_int_mag(eco.absrmag,ecofofid)
_, uniqsel = np.unique(ecofofid, return_index=True)
logmh200func = interp1d(ecogrpabsrmag[uniqsel],ecologmh200[uniqsel],fill_value='extrapolate') 
logmhvirfunc = interp1d(ecogrpabsrmag[uniqsel],ecologmhvir[uniqsel],fill_value='extrapolate')
tmp = pd.DataFrame({'grprmag':ecogrpabsrmag[uniqsel],'mhvir':ecologmhvir[uniqsel], 'mh200':ecologmh200[uniqsel]})
tmp = tmp.sort_values(by='grprmag').tail(100)
paramsvir = np.polyfit(tmp.grprmag,tmp.mhvir,1) 
params200 = np.polyfit(tmp.grprmag,tmp.mh200,1)
extrap_func_vir = lambda Mrtot: np.polyval(paramsvir,Mrtot) 
extrap_func_200 = lambda Mrtot: np.polyval(params200,Mrtot) 

plt.figure()
plt.plot(tmp.grprmag,tmp.mhvir)
tx=np.linspace(-25,-15,100)
plt.plot(tx,extrap_func_vir(tx),'k-')
plt.show()

# get RES-B ham masses from splines
#print('MAX ',max(resb.absrmag))
resbgrpabsrmag = ic.get_int_mag(resb.absrmag,resbfofid)
resbgrpgiantabsrmag = get_int_mag_giants(resb.absrmag.to_numpy(),resbfofid)
resblogmh200 = logmh200func(resbgrpabsrmag)
resblogmhvir = logmhvirfunc(resbgrpabsrmag)
resbdynmass = fof.dynmass(resb.radeg.to_numpy(),resb.dedeg.to_numpy(),resb.cz.to_numpy(),resbfofid,Aval=9.9,h=0.7)
resbdynmass[np.where(resbgrpn<=7)]=-999.
resbscflag = np.zeros(len(resbfofid))
#plt.figure()
#plt.scatter(ecogrpabsrmag,ecologmhvir,color='lightgreen',s=4)
#plt.scatter(resbgrpabsrmag,resblogmhvir,color='k',s=2)
#plt.show()

# populate and output dataframe
res.loc[resb_groupsel,'fofgrp']=resbfofid+np.max(res.fofgrp)+1+maxECOID # this overkill, just making sure not re-using group IDs

faintsel = (res.f_b==1)&(res.absrmag>-17.0)
faintassocid  = linking_association(res[resb_groupsel].radeg,res[resb_groupsel].dedeg,res[resb_groupsel].cz,res[resb_groupsel].fofgrp,res[faintsel].radeg,\
            res[faintsel].dedeg,res[faintsel].cz, 0.07*sep, 1.1*sep, 100., 0.3, 0.7)
res.loc[faintsel,'fofgrp']=faintassocid

res.loc[resb_groupsel,'fofgrpn']=resbgrpn
res.loc[resb_groupsel,'fofgrpradeg']=resbgrpra
res.loc[resb_groupsel,'fofgrpdedeg']=resbgrpde
res.loc[resb_groupsel,'fofgrpcz']=resbgrpcz
res.loc[resb_groupsel,'fofgrpabsrmag']=resbgrpabsrmag
res.loc[resb_groupsel,'fofgrpgiantabsrmag']=resbgrpgiantabsrmag
res.loc[resb_groupsel,'foflogmh200']=resblogmh200
res.loc[resb_groupsel,'foflogmhvir']=resblogmhvir
res.loc[resb_groupsel,'fofgrpmhi']=resbgrpmhi
res.loc[resb_groupsel,'logmstarfofgrp']=resblogmstargrp
res.loc[resb_groupsel,'logmbaryfofgrp']=resblogmbarygrp
res.loc[resb_groupsel,'foflogmhdyn']=resbdynmass
res.loc[resb_groupsel,'fofskycutoffflag']=resbscflag
res.fillna({'fofgrp':-999.,'fofgrpn':-999.,'fofgrpradeg':-999.,'fofgrpdedeg':-999.,'fofgrpcz':-999.,'fofgrpabsrmag':-999.,\
            'foflogmh200':-999., 'foflogmhvir':-999, 'fofgrpmhi':-999., 'logmstarfofgrp':-999., 'logmbaryfofgrp':-999.},inplace=True)

### need to add group metrics for sub-floor dwarfs
subfloordwarfsB = res[faintsel]
for subfloorname,grpidvalue in zip(subfloordwarfsB.index.to_numpy(), subfloordwarfsB.fofgrp.to_numpy()):
    fillsel = (res.index==subfloorname)
    grpsel = (res.fofgrp==grpidvalue)&(res.absrmag<=-17.0)
    if np.sum(grpsel)>0:
        res.loc[fillsel,'fofgrpn'] = res.loc[grpsel,'fofgrpn'][0]
        res.loc[fillsel,'fofgrpradeg'] = res.loc[grpsel,'fofgrpradeg'][0]
        res.loc[fillsel,'fofgrpdedeg'] = res.loc[grpsel,'fofgrpdedeg'][0]
        res.loc[fillsel,'fofgrpcz'] = res.loc[grpsel,'fofgrpcz'][0]
        res.loc[fillsel,'fofgrpabsrmag'] = res.loc[grpsel,'fofgrpabsrmag'][0]
        res.loc[fillsel,'fofgrpgiantabsrmag'] = res.loc[grpsel,'fofgrpgiantabsrmag'][0]
        res.loc[fillsel,'foflogmhvir'] = res.loc[grpsel,'foflogmhvir'][0]
        res.loc[fillsel,'foflogmh200'] = res.loc[grpsel,'foflogmh200'][0]
        res.loc[fillsel,'fofgrpmhi'] = res.loc[grpsel,'fofgrpmhi'][0]
        res.loc[fillsel,'logmstarfofgrp'] = res.loc[grpsel,'logmstarfofgrp'][0]
        res.loc[fillsel,'logmbaryfofgrp'] = res.loc[grpsel,'logmbaryfofgrp'][0]
        res.loc[fillsel,'foflogmhdyn'] = res.loc[grpsel,'foflogmhdyn'][0]
        res.loc[fillsel,'fofskycutoffflag'] = res.loc[grpsel,'fofskycutoffflag'][0]
    else:
        res.loc[fillsel,'fofgrpn'] = 1.0
        res.loc[fillsel,'fofgrpradeg'] = res.loc[fillsel,'radeg']
        res.loc[fillsel,'fofgrpdedeg'] = res.loc[fillsel,'dedeg']
        res.loc[fillsel,'fofgrpcz'] = res.loc[fillsel,'cz']
        res.loc[fillsel,'fofgrpabsrmag'] = res.loc[fillsel, 'absrmag']
        res.loc[fillsel,'fofgrpgiantabsrmag'] = -999.
        res.loc[fillsel,'foflogmhvir'] = extrap_func_vir(res.loc[fillsel,'absrmag'])
        res.loc[fillsel,'foflogmh200'] = extrap_func_200(res.loc[fillsel,'absrmag'])
        res.loc[fillsel,'fofgrpmhi'] = np.log10(10**res.loc[fillsel,'logmgas'] / 1.4)
        res.loc[fillsel,'logmstarfofgrp'] = res.loc[fillsel,'logmstar']
        res.loc[fillsel,'logmbaryfofgrp'] = np.log10(10**res.loc[fillsel,'logmstar'] + 10**res.loc[fillsel,'logmgas'])
        res.loc[fillsel,'foflogmhdyn'] = -999.
        res.loc[fillsel,'fofskycutoffflag'] = 0.

#print(res[['absrmag','fofgrp','fofgrpn','fofgrpradeg','fofgrpdedeg','fofgrpcz','fofgrpabsrmag','fofgrpgiantabsrmag','foflogmh200','foflogmhvir','fofgrpmhi', 'logmstarfofgrp', 'logmbaryfofgrp','fofskycutoffflag']])

#print(res[['absrmag','fofgrpabsrmag','fofgrpgiantabsrmag']])
res.to_csv(".RESOLVE_DR4_groups_only.csv")
