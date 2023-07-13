startingfile = "ECOdata_051123.csv"
ecodr2file = "ECODR2_052423.csv"
photinputfile = '/srv/two/sheila/pops/ECO_all_massinputfilefix.txt'
outputfile = 'eco_dr3_base_071223.csv'
sedsuppdata = '/srv/two/sheila/pops/ecodr3fixsuppdata.dat'
sedmasses = '/srv/two/sheila/pops/ecodr3fixbmasses.dat'
sedextdists = '/srv/two/sheila/pops/ecodr3fixbextdists.dat'
sedkcorrs = '/srv/two/sheila/pops/ecodr3fixbkcorrs.dat'

resinputfile = '/srv/two/sheila/pops/resolve_fuvnuvuvot_ugriz_YJHKall.txt'
rs0106inputfile = '/srv/two/sheila/pops/rs0106_fuvnuvuvot_ugriz_YJHKall.txt'
ressedsuppdata = '/srv/two/sheila/pops/resdr4suppdata.dat'
ressedbmasses = '/srv/two/sheila/pops/resdr4bmasses.dat'
ressedbextdists = '/srv/two/sheila/pops/resdr4bextdists.dat'
ressedbkcorrs = '/srv/two/sheila/pops/resdr4bkcorrs.dat'
resoutputfile = 'res_dr4_base_062823.csv'



import pandas as pd
import numpy as np
from calculate_logmgas import calculate_logmgas_hutchens23
from scipy.io import readsav
import subprocess 
import pdb
from astropy.cosmology import LambdaCDM
hubble_const = 70.
omega_m = 0.3
omega_de0 = 0.7
cspeed=3e5
cosmo = LambdaCDM(hubble_const, Om0=omega_m, Ode0=omega_de0)

if __name__=='__main__':
    ecoliv = pd.read_csv(startingfile).set_index('name').sort_index()
    
    ## first, need to drop all photometry + SED stuff. Going to retrieve it
    ## directly from raw files on disk
    ecoliv = ecoliv.drop(labels=['logmstar'],axis=1)
    ecoliv = ecoliv.drop(labels=[kk for kk in ecoliv.keys() if (('logmstar' in kk) or ('fsmgr' in kk) or ('sfr' in kk) or ('dgr' in kk) or ('mu_delta' in kk))], axis=1)
    ecoliv = ecoliv.drop(labels=[kk for kk in ecoliv.keys() if 'mag' in kk],axis=1) # drop mags and errors 
    ecoliv = ecoliv.drop(labels=[kk for kk in ecoliv.keys() if (('_' in kk) and ('model' in kk))],axis=1) # drop modelx_y, modelx_ycorr
    ecoliv = ecoliv.drop(labels=[kk for kk in ecoliv.keys() if 'ext' in kk],axis=1) # drop extinctions
    ecoliv = ecoliv.drop(labels=[kk for kk in ecoliv.keys() if (('grp' in kk) or ('logmh' in kk) or ('den1mpc' in kk))],axis=1)

    # get phot from input file
    inputfile_columns = ["name","ra","dec","vhel","nuv","nuverr","mag_u","magerr_u","mag_g","magerr_g","mag_r","magerr_r","mag_i"\
        ,"magerr_i","mag_z","magerr_z","yukmag","eyukmag","hukmag","ehukmag","kukmag","ekukmag","jmag","ejmag"\
        ,"hmag","ehmag","kmag","ekmag","extinction_u","extinction_g","extinction_r","extinction_i","extinction_z"\
        ,"ayuk","aj","ahuk","akuk","vlg","uvo t","uvoterr","fuv","fuverr","extnuvccm","extnuvwyder","extuvotm2ccm"\
        ,"extfuv"]
    photinput = pd.read_csv(photinputfile,sep='\s+', names=inputfile_columns).set_index('name')
    photinput = photinput[photinput.index != 'SJUSHR9999']
    photinput = photinput.loc[~photinput.index.duplicated(),:]
    photinput = photinput.loc[ecoliv.index,:].sort_index()
    assert len(photinput)==len(ecoliv)
    assert (photinput.index==ecoliv.index).all()

    # fuv not published for RESOLVE, leaving out
    ecoliv.loc[:,'uvm2mag'] = photinput['uvo t']
    ecoliv.loc[:,'nuvmag'] = photinput['nuv'] # filter mag values
    ecoliv.loc[:,'umag'] = photinput['mag_u']
    ecoliv.loc[:,'gmag'] = photinput['mag_g']
    ecoliv.loc[:,'rmag'] = photinput['mag_r']
    ecoliv.loc[:,'imag'] = photinput['mag_i']
    ecoliv.loc[:,'zmag'] = photinput['mag_z']
    ecoliv.loc[:,'uymag'] = photinput['yukmag']
    ecoliv.loc[:,'uhmag'] = photinput['hukmag']
    ecoliv.loc[:,'ukmag'] = photinput['kukmag']
    ecoliv.loc[:,'2jmag'] = photinput['jmag']
    ecoliv.loc[:,'2hmag'] = photinput['hmag']
    ecoliv.loc[:,'2kmag'] = photinput['kmag']

    ecoliv.loc[:,'e_uvm2mag'] = photinput['uvoterr']
    ecoliv.loc[:,'e_nuvmag'] = photinput['nuverr'] # filter mag errors
    ecoliv.loc[:,'e_umag'] = photinput['magerr_u']
    ecoliv.loc[:,'e_gmag'] = photinput['magerr_g']
    ecoliv.loc[:,'e_rmag'] = photinput['magerr_r']
    ecoliv.loc[:,'e_imag'] = photinput['magerr_i']
    ecoliv.loc[:,'e_zmag'] = photinput['magerr_z']
    ecoliv.loc[:,'e_uymag'] = photinput['eyukmag']
    ecoliv.loc[:,'e_uhmag'] = photinput['ehukmag']
    ecoliv.loc[:,'e_ukmag'] = photinput['ekukmag']
    ecoliv.loc[:,'e_2jmag'] = photinput['ejmag']
    ecoliv.loc[:,'e_2hmag'] = photinput['ehmag']
    ecoliv.loc[:,'e_2kmag'] = photinput['ekmag']

    ecoliv.loc[:,'extuvm2'] = photinput['extuvotm2ccm']
    ecoliv.loc[:,'extnuv'] = photinput['extnuvccm'] # extinction values
    ecoliv.loc[:,'extu'] = photinput['extinction_u']
    ecoliv.loc[:,'extg'] = photinput['extinction_g']
    ecoliv.loc[:,'extr'] = photinput['extinction_r']
    ecoliv.loc[:,'exti'] = photinput['extinction_i']
    ecoliv.loc[:,'extz'] = photinput['extinction_z']
    ecoliv.loc[:,'exty'] = photinput['ayuk']
    ecoliv.loc[:,'exth'] = photinput['ahuk']
    ecoliv.loc[:,'extk'] = photinput['akuk']
    ecoliv.loc[:,'extj'] = photinput['aj']
    #ecoliv.loc[:,'exth'] = photinput['ahuk']
    #ecoliv.loc[:,'extk'] = photinput['akuk']

    # Fix ECO NUV where had small positive values
    sel = (ecoliv.nuvmag<1)
    ecoliv.loc[sel,'nuvmag'] = 0.0

    # Put parenthesis around mag with the catatrosphic errors
    ecoliv.loc[:,'absrmag'] = (ecoliv.rmag + 5 - 5*np.log10(cosmo.luminosity_distance(np.array(ecoliv.cz)/cspeed).to_value() * 1e6) - ecoliv.extr)
    ecoliv.loc[:,'absgmag'] = (ecoliv.gmag + 5 - 5*np.log10(cosmo.luminosity_distance(np.array(ecoliv.cz)/cspeed).to_value() * 1e6) - ecoliv.extg)
    ecoliv = ecoliv.round({kk:3 for kk in ecoliv.keys() if 'mag' in kk})
    for kk in ecoliv.keys():
        if 'mag' in kk:
            ecoliv[kk+'_withoutparens'] = ecoliv[kk].copy()
    ecoliv = ecoliv.astype({kk:str for kk in ecoliv.keys() if (('mag' in kk) and ('withoutparens' not in kk))})
    photerrcut={'nuv':3.6, 'u':3, 'g':1.7, 'r':1.7, 'i':1.7, 'z':1.7, 'uy':2.3, '2j':2.3, '2h':2.3, '2k':2.3, 'uh':2.3, 'uk':2.3}
    filterbands=['uvm2','nuv','u','g','r','i','z','uy','uh','uk','2j','2k','2h']
    extbands=['uvm2','nuv','u','g','r','i','z','y','h','k','j','k','h']
    for ii,nn in enumerate(ecoliv.index):
        for bb in filterbands:
            if bb!='uvm2':
                if (ecoliv.loc[nn,'e_'+bb+'mag_withoutparens']>photerrcut[bb]):
                    ecoliv.loc[nn,bb+'mag']='('+str(ecoliv.loc[nn,bb+'mag'])+')'
                    ecoliv.loc[nn,'e_'+bb+'mag']='('+str(ecoliv.loc[nn,'e_'+bb+'mag'])+')'
                    if 'abs'+bb+'mag' in ecoliv.keys():
                        ecoliv.loc[nn,'abs'+bb+'mag']='('+str(ecoliv.loc[nn,'abs'+bb+'mag'])+')'
        hasnuvur = ecoliv.loc[nn,'nuvmag_withoutparens']>0 and ecoliv.loc[nn,'umag_withoutparens']>0 and ecoliv.loc[nn,'rmag_withoutparens']>0
        badnuvcolor = (ecoliv.loc[nn,'nuvmag_withoutparens']-ecoliv.loc[nn,'umag_withoutparens'])<-4
        inurrange = ((ecoliv.loc[nn,'umag_withoutparens']-ecoliv.loc[nn,'rmag_withoutparens'])>-2.5) and ((ecoliv.loc[nn,'umag_withoutparens']-ecoliv.loc[nn,'rmag_withoutparens'])<4.5)
        if hasnuvur and badnuvcolor and inurrange:
            ecoliv.loc[nn,'nuvmag']='('+str(ecoliv.loc[nn,'nuv'+'mag'])+')'
            ecoliv.loc[nn,'e_nuvmag']='('+str(ecoliv.loc[nn,'e_'+'nuv'+'mag'])+')'
        elif hasnuvur and badnuvcolor and (not inurrange):
            ecoliv.loc[nn,'umag']='('+str(ecoliv.loc[nn,'u'+'mag'])+')'
            ecoliv.loc[nn,'e_umag']='('+str(ecoliv.loc[nn,'e_'+'u'+'mag'])+')'

    # get SED info from Sheila's code outputs, and construct into a single dataframe
    sednames = readsav(sedsuppdata)['name']
    sednames = np.array([nn.decode('utf-8') for nn in sednames])
    masses = readsav(sedmasses)['medmass']
    sedoutputs = pd.DataFrame({'name':sednames,'logmstar':masses})
    sheilamap = {'u':0,'g':1,'r':2,'i':3,'z':4,'j':5,'h':6,'k':7,'nuv':8,'uy':9,'uh':10,'uk':11,'uvotm2':12}#makeresolvecatalog 1448
    colors_need_to_get=['nuv_r','u_r','u_i','u_j','u_k','g_r','g_i','g_j','g_k','nuv_k']
    colorfile = readsav(sedextdists)
    for cc in colors_need_to_get:
        f1,f2 = cc.split('_')
        sedoutputs.loc[:,'model'+cc] = colorfile['smoothrestmag'][:,sheilamap[f1]] - colorfile['smoothrestmag'][:,sheilamap[f2]]
        sedoutputs.loc[:,'model'+cc+'corr'] = colorfile['deextrestmag'][:,sheilamap[f1]] - colorfile['deextrestmag'][:,sheilamap[f2]]
    sedoutputs.loc[:,'kcorrr'] = readsav(sedkcorrs)['meankcorr'][:,sheilamap['r']] # can do others, just doing r for now
    sedoutputs = sedoutputs[sedoutputs.name!='SJUSHR9999'].set_index('name').loc[ecoliv.index,:].sort_index()
    sedoutputs = sedoutputs.loc[~sedoutputs.index.duplicated(),:] # remove duplicate objects 
    assert len(sedoutputs)==len(ecoliv)
    assert (sedoutputs.index==ecoliv.index).all()
    ecoliv = pd.concat([ecoliv,sedoutputs],axis=1)

    
    ##########################################################
    ##########################################################
    # Replace using RESOLVE photometry + SED modeling outputs
    # Take care to treat rs0106 individually
    resinput = pd.read_csv(resinputfile,sep='\s+',names=inputfile_columns).set_index('name')
    rs0106input = pd.read_csv(rs0106inputfile,sep='\s+',names=inputfile_columns).set_index('name')
    resinput.update(rs0106input)

    officialresname=resinput.index.to_numpy() # below is taken from Sheila's code, see slack post
    officialresname[np.where(officialresname == 'rfnew1')] ='rf0798'
    officialresname[np.where(officialresname == 'rfnew2')] ='rf0799'
    officialresname[np.where(officialresname == 'rfnew3')] ='rf0800'
    officialresname[np.where(officialresname == 'rfnew4')] ='rf0801'
    officialresname[np.where(officialresname == 'rfnew5')] ='rf0802'
    officialresname[np.where(officialresname == 'rfnew6')] ='rf0803'

    officialresname[np.where(officialresname == 'rsnew1')] ='rs1410'
    officialresname[np.where(officialresname == 'rsnew2')] ='rs1411'
    officialresname[np.where(officialresname == 'rsnew3')] ='rs1412'
    officialresname[np.where(officialresname == 'rsnew4')] ='rs1413'
    officialresname[np.where(officialresname == 'rsnew5')] ='rs1414'
    officialresname[np.where(officialresname == 'rsnew6')] ='rs1415'
    officialresname[np.where(officialresname == 'rsnew7')] ='rs1416'
    officialresname[np.where(officialresname == 'rsnew8')] ='rs1417'
    officialresname[np.where(officialresname == 'rsnew9')] ='rs1418'

    officialresname[np.where(officialresname == 'HI085755.3+013003')] ='rs1419'
    officialresname[np.where(officialresname == 'HI091756.0+003300')] ='rs1420'
    officialresname[np.where(officialresname == 'HI092520.1+030700')] ='rs1421'
    officialresname[np.where(officialresname == 'HI092926.2+015242')] ='rs1422'
    officialresname[np.where(officialresname == 'HI093519.6+025923')] ='rs1423'
    officialresname[np.where(officialresname == 'HI093739.5+024645')] ='rs1424'
    officialresname[np.where(officialresname == 'HI093811.6+025617')] ='rs1425'
    officialresname[np.where(officialresname == 'HI094549.2+034007')] ='rs1426'
    officialresname[np.where(officialresname == 'HI094617.2+032452')] ='rs1427'
    officialresname[np.where(officialresname == 'HI094731.6+035913')] ='rs1428'
    officialresname[np.where(officialresname == 'HI095200.1+041424')] ='rs1429'
    officialresname[np.where(officialresname == 'HI095414.2+022835')] ='rs1430'
    officialresname[np.where(officialresname == 'HI100545.1+013620')] ='rs1431'
    officialresname[np.where(officialresname == 'HI101059.1+033421')] ='rs1432'
    officialresname[np.where(officialresname == 'HI102558.9+003044')] ='rs1433'
    officialresname[np.where(officialresname == 'HI102750.4+023010')] ='rs1434'
    officialresname[np.where(officialresname == 'HI103828.9+000441')] ='rs1435'
    officialresname[np.where(officialresname == 'HI103844.6+035356')] ='rs1436'
    officialresname[np.where(officialresname == 'HI111301.0+040330')] ='rs1437'
    officialresname[np.where(officialresname == 'HI111553.8+042233')] ='rs1438'
    officialresname[np.where(officialresname == 'HI111608.2+033207')] ='rs1439'
    officialresname[np.where(officialresname == 'HI112016.2+004147')] ='rs1440'
    officialresname[np.where(officialresname == 'HI113341.7+021510')] ='rs1441'
    officialresname[np.where(officialresname == 'HI113355.6+010142')] ='rs1442'
    officialresname[np.where(officialresname == 'HI113701.4+032630')] ='rs1443'
    officialresname[np.where(officialresname == 'HI115134.1+022437')] ='rs1444'
    officialresname[np.where(officialresname == 'HI120247.9+022337')] ='rs1445'
    officialresname[np.where(officialresname == 'HI121530.9+024554')] ='rs1446'
    officialresname[np.where(officialresname == 'HI122120.1+034730')] ='rs1447'
    officialresname[np.where(officialresname == 'HI122426.0+014600')] ='rs1448'
    officialresname[np.where(officialresname == 'HI123106.0+001527')] ='rs1449'
    officialresname[np.where(officialresname == 'HI123518.9+012239')] ='rs1450'
    officialresname[np.where(officialresname == 'HI124808.3+033800')] ='rs1451'
    officialresname[np.where(officialresname == 'HI125819.3+040755')] ='rs1452'
    officialresname[np.where(officialresname == 'HI132248.4+034653')] ='rs1453'
    officialresname[np.where(officialresname == 'HI133812.3+041400')] ='rs1454'
    officialresname[np.where(officialresname == 'HI133817.2+035618')] ='rs1455'
    officialresname[np.where(officialresname == 'HI134046.8+024901')] ='rs1456'
    officialresname[np.where(officialresname == 'HI134117.3+025817')] ='rs1457'
    officialresname[np.where(officialresname == 'HI134440.9+030230')] ='rs1458'
    officialresname[np.where(officialresname == 'HI134530.3+011521')] ='rs1459'
    officialresname[np.where(officialresname == 'HI135359.8+031447')] ='rs1460'
    officialresname[np.where(officialresname == 'HI135500.6+032242')] ='rs1461'
    officialresname[np.where(officialresname == 'HI135907.7+033008')] ='rs1462'
    officialresname[np.where(officialresname == 'HI140335.0+015107')] ='rs1463'
    officialresname[np.where(officialresname == 'HI142034.6+044043')] ='rs1464'
    officialresname[np.where(officialresname == 'gama1')] ='rs1465'
    officialresname[np.where(officialresname == 'gama2')] ='rs1466'
    officialresname[np.where(officialresname == 'gama3')] ='rs1467'
    officialresname[np.where(officialresname == 'gama4')] ='rs1468'
    officialresname[np.where(officialresname == 'gama5')] ='rs1469'
    officialresname[np.where(officialresname == 'gama6')] ='rs1470'
    officialresname[np.where(officialresname == 'gama7')] ='rs1471'
    officialresname[np.where(officialresname == 'gama8')] ='rs1472'
    officialresname[np.where(officialresname == 'gama9')] ='rs1473'
    officialresname[np.where(officialresname == 'gama10')] ='rs1474'
    officialresname[np.where(officialresname == 'gama11')] ='rs1475'
    officialresname[np.where(officialresname == 'gama12')] ='rs1476'
    officialresname[np.where(officialresname == 'gama13')] ='rs1477'
    officialresname[np.where(officialresname == 'gama14')] ='rs1478'
    officialresname[np.where(officialresname == 'gama15')] ='rs1479'
    officialresname[np.where(officialresname == 'gama16')] ='rs1480'
    officialresname[np.where(officialresname == 'gama17')] ='rs1481'
    officialresname[np.where(officialresname == 'gama19')] ='rs1482'
    officialresname[np.where(officialresname == 'gama20')] ='rs1483'
    officialresname[np.where(officialresname == 'gama21')] ='rs1484'
    officialresname[np.where(officialresname == 'gama22')] ='rs1485'
    officialresname[np.where(officialresname == 'gama23')] ='rs1486'
    officialresname[np.where(officialresname == 'gama24')] ='rs1487'
    officialresname[np.where(officialresname == 'gama25')] ='rs1488'
    officialresname[np.where(officialresname == 'gama26')] ='rs1489'
    officialresname[np.where(officialresname == 'gama27')] ='rs1490'
    officialresname[np.where(officialresname == 'gama28')] ='rs1491'
    officialresname[np.where(officialresname == 'gama29')] ='rs1492'
    officialresname[np.where(officialresname == 'gama30')] ='rs1493'
    officialresname[np.where(officialresname == 'gama31')] ='rs1494'
    officialresname[np.where(officialresname == 'gama32')] ='rs1495'
    officialresname[np.where(officialresname == 'gama33')] ='rs1496'
    officialresname[np.where(officialresname == 'gama34')] ='rs1497'
    officialresname[np.where(officialresname == 'gama35')] ='rs1498'
    officialresname[np.where(officialresname == 'gama36')] ='rs1499'
    officialresname[np.where(officialresname == 'gama37')] ='rs1500'
    officialresname[np.where(officialresname == 'gama38')] ='rs1501'
    officialresname[np.where(officialresname == 'gama39')] ='rs1502'
    officialresname[np.where(officialresname == 'gama40')] ='rs1503'
    officialresname[np.where(officialresname == 'gama41')] ='rs1504'
    officialresname[np.where(officialresname == 'gama42')] ='rs1505'
    officialresname[np.where(officialresname == 'gama43')] ='rs1506'
    officialresname[np.where(officialresname == 'gama44')] ='rs1507'
    officialresname[np.where(officialresname == 'gama45')] ='rs1508'
    officialresname[np.where(officialresname == 'gama46')] ='rs1509'
    officialresname[np.where(officialresname == 'gama47')] ='rs1510'
    officialresname[np.where(officialresname == 'gama48')] ='rs1511'
    officialresname[np.where(officialresname == 'gama49')] ='rs1512'
    officialresname[np.where(officialresname == 'gama50')] ='rs1513'
    officialresname[np.where(officialresname == 'gama51')] ='rs1514'
    officialresname[np.where(officialresname == 'gama52')] ='rs1515'
    officialresname[np.where(officialresname == 'gama53')] ='rs1516'
    officialresname[np.where(officialresname == 'gama54')] ='rs1517'
    officialresname[np.where(officialresname == 'optcHI091958.4+005152')] ='rs1518'
    resinput = resinput.set_index(officialresname)#.sort_index()
  
    cnamemapping = {'nuv':'nuvmag', 
           'nuverr':'e_nuvmag', 
           'mag_u':'umag', 
           'magerr_u':'e_umag', 
           'mag_g':'gmag',
           'magerr_g':'e_gmag', 
           'mag_r':'rmag', 
           'magerr_r':'e_rmag', 
           'mag_i':'imag', 
           'magerr_i':'e_imag', 
           'mag_z':'zmag',
           'magerr_z':'e_zmag', 
           'yukmag':'uymag', 
           'eyukmag':'e_uymag', 
           'hukmag':'uhmag', 
           'ehukmag':'e_uhmag', 
           'kukmag':'ukmag',
           'ekukmag':'e_ukmag', 
           'jmag':'2jmag', 
           'ejmag':'e_2jmag', 
           'hmag':'2hmag', 
           'ehmag':'e_2hmag', 
           'kmag':'2kmag', 
           'ekmag':'e_2kmag',
           'extinction_u':'extu', 
           'extinction_g':'extg',
           'extinction_r':'extr', 
           'extinction_i':'exti',
           'extinction_z':'extz', 
           'ayuk':'exty', 
           'aj':'extj', 
           'ahuk':'exth', 
           'akuk':'extk', 
           'uvo t':'uvm2mag', 
           'uvoterr':'e_uvm2mag',
           #'fuv':None, 
           #'fuverr':None, 
           'extnuvccm':'extnuv', 
           #'extnuvwyder':None, 
           'extuvotm2ccm':'extuvm2', 
           #'extfuv':None
          }
    resinput=resinput.rename(cnamemapping,inplace=False,axis=1)

    masses = readsav(ressedbmasses)['medmass']
    ressedboutputs = pd.DataFrame({'name':officialresname,'logmstar':masses})
    colorfile = readsav(ressedbextdists)
    for cc in colors_need_to_get:
        f1,f2 = cc.split('_')
        ressedboutputs.loc[:,'model'+cc] = colorfile['smoothrestmag'][:,sheilamap[f1]] - colorfile['smoothrestmag'][:,sheilamap[f2]]
        ressedboutputs.loc[:,'model'+cc+'corr'] = colorfile['deextrestmag'][:,sheilamap[f1]] - colorfile['deextrestmag'][:,sheilamap[f2]]
    ressedboutputs.loc[:,'kcorrr'] = readsav(ressedbkcorrs)['meankcorr'][:,sheilamap['r']] # can do others, just doing r for now
    ressedboutputs = ressedboutputs[ressedboutputs.name!='SJUSHR9999'].set_index('name').loc[resinput.index,:]
    ressedboutputs = ressedboutputs.loc[~ressedboutputs.index.duplicated(),:] # remove duplicate objects 
    assert len(ressedboutputs)==len(resinput)
    assert (ressedboutputs.index==resinput.index).all()
    resphot = pd.concat([resinput,ressedboutputs],axis=1)
    
    # Trim file to just the previously published RESOLVE names
    # also get cz and compute absrmag for substitution into ECO catalog 
    resliv = pd.read_csv("RESOLVEliving_061223.csv")[['name','f_a','f_b','cz','econame','r50','r90','b_a','mhidet','emhidet','mhilim','confused','mhi_corr','emhi_corr_rand','emhi_corr_sys']].set_index('name')
    resphot = resphot.loc[resliv.index,:]
    resphot = pd.concat([resphot,resliv],axis=1)
    resphot.loc[:,'absrmag'] = (resphot.rmag + 5 - 5*np.log10(cosmo.luminosity_distance(np.array(resphot.cz)/cspeed).to_value() * 1e6) - resphot.extr)
    resphot.loc[:,'absgmag'] = (resphot.gmag + 5 - 5*np.log10(cosmo.luminosity_distance(np.array(resphot.cz)/cspeed).to_value() * 1e6) - resphot.extg)
    resphot.rename({'ra':'radeg','dec':'dedeg'},inplace=True,axis=1) 
    resphot.drop(['vlg','fuv','fuverr','extfuv','extnuvwyder'],axis=1,inplace=True)


    # Put parenthesis around mag with the catatrosphic errors
    resphot = resphot.round({kk:3 for kk in resphot.keys() if 'mag' in kk})
    for kk in resphot.keys():
        if 'mag' in kk:
            resphot[kk+'_withoutparens'] = resphot[kk].copy()
    resphot = resphot.astype({kk:str for kk in resphot.keys() if (('mag' in kk) and ('withoutparens' not in kk))})
    photerrcut={'nuv':3.6, 'u':3, 'g':1.7, 'r':1.7, 'i':1.7, 'z':1.7, 'uy':2.3, '2j':2.3, '2h':2.3, '2k':2.3, 'uh':2.3, 'uk':2.3}
    filterbands=['uvm2','nuv','u','g','r','i','z','uy','uh','uk','2j','2k','2h']
    extbands=['uvm2','nuv','u','g','r','i','z','y','h','k','j','k','h']
    for ii,nn in enumerate(resphot.index):
        for bb in filterbands:
            if bb!='uvm2':
                if (resphot.loc[nn,'e_'+bb+'mag_withoutparens']>photerrcut[bb]):
                    resphot.loc[nn,bb+'mag']='('+str(resphot.loc[nn,bb+'mag'])+')'
                    resphot.loc[nn,'e_'+bb+'mag']='('+str(resphot.loc[nn,'e_'+bb+'mag'])+')'
                    if 'abs'+bb+'mag' in resphot.keys():
                        resphot.loc[nn,'abs'+bb+'mag']='('+str(resphot.loc[nn,'abs'+bb+'mag'])+')'
        hasnuvur = resphot.loc[nn,'nuvmag_withoutparens']>0 and resphot.loc[nn,'umag_withoutparens']>0 and resphot.loc[nn,'rmag_withoutparens']>0
        badnuvcolor = (resphot.loc[nn,'nuvmag_withoutparens']-resphot.loc[nn,'umag_withoutparens'])<-4
        inurrange = ((resphot.loc[nn,'umag_withoutparens']-resphot.loc[nn,'rmag_withoutparens'])>-2.5) and ((resphot.loc[nn,'umag_withoutparens']-resphot.loc[nn,'rmag_withoutparens'])<4.5)
        if hasnuvur and badnuvcolor and inurrange:
            resphot.loc[nn,'nuvmag']='('+str(resphot.loc[nn,'nuv'+'mag'])+')'
            resphot.loc[nn,'e_nuvmag']='('+str(resphot.loc[nn,'e_'+'nuv'+'mag'])+')'
        elif hasnuvur and badnuvcolor and (not inurrange):
            resphot.loc[nn,'umag']='('+str(resphot.loc[nn,'u'+'mag'])+')'
            resphot.loc[nn,'e_umag']='('+str(resphot.loc[nn,'e_'+'u'+'mag'])+')'

    # Substitute quantities into ECO
    ecoliv = ecoliv.reset_index().set_index('resname').sort_index()
    resatmp = resphot.loc[[ii for ii in resphot.index if 'rs' in ii],:].sort_index()
    for kk in ecoliv.keys():
        if ('mag' in kk) or (('model' in kk) and ('_' in kk)) or ('ext' in kk) or ('logmstar' in kk) or ('sfr' in kk):
            ecoliv.loc[resatmp.index,kk] = resatmp.loc[resatmp.index,kk]
    ecoliv = ecoliv.reset_index().set_index('name')

    # Deal with ECO13546 // rs1519 (needs to be added to RESOLVE file)
    # also add rs1520/ECO01169, rs1521/ECO01356, and rs1522/ECO12853
    ecoliv.loc['ECO01169','resname'] = 'rs1520'
    ecoliv.loc['ECO01356','resname'] = 'rs1521'
    ecoliv.loc['ECO12853','resname'] = 'rs1522'
    resphot = pd.concat([resphot,ecoliv.loc[['ECO13546','ECO01169','ECO01356','ECO12853'],['resname']+[kk for kk in resphot.keys() if kk not in ('econame','f_a','f_b')]].set_index('resname')],axis=0)
    resphot.loc['rs1519','econame'] = 'ECO13546'
    resphot.loc['rs1520','econame'] = 'ECO01169'
    resphot.loc['rs1521','econame'] = 'ECO01356'
    resphot.loc['rs1522','econame'] = 'ECO12853'
  
    assert len(ecoliv[[('rs' in nn) for nn in ecoliv.resname]])==len(resphot[[('rs' in nn) for nn in resphot.index]])
 
    # Calculate remaining photometry outputs
    calc_mu_delta = lambda mstar, r50, r90: np.log10(0.9*mstar/(np.pi*r90*r90)) + 1.7*(np.log10(0.5*mstar/(np.pi*r50*r50)) - np.log10(0.4*mstar/(np.pi*r90*r90 - np.pi*r50*r50))) # K13 eq. 1-3
    ecoliv.loc[:,'r50_in_kpc'] = ecoliv.r50/206265 * cosmo.angular_diameter_distance(np.array(ecoliv.cz)/cspeed).value * 1e3 # kpc/Mpc
    ecoliv.loc[:,'r90_in_kpc'] = ecoliv.r90/206265 * cosmo.angular_diameter_distance(np.array(ecoliv.cz)/cspeed).value * 1e3 # kpc/Mpc
    ecoliv.loc[:,'mu_delta'] = calc_mu_delta(10**ecoliv.logmstar, ecoliv.r50_in_kpc, ecoliv.r90_in_kpc)
    ecoliv.loc[ecoliv[(ecoliv.r50==ecoliv.r90)|(ecoliv.r50<0)].index,['mu_delta','r50','r90']] = -999.
    resphot.loc[:,'mu_delta'] = calc_mu_delta(10**resphot.logmstar, resphot.cz/70.*1e3*resphot.r50/206265, resphot.cz/70.*1e3*resphot.r90/206265)

    print(ecoliv)
    print(resphot)
    ##########################################################
    ##########################################################
    ##########################################################
    # Re-scale quantities using new cz's, recompute logmgas.
    ecodr2 = pd.read_csv(ecodr2file).set_index('name').sort_index()
    assert len(ecoliv)==len(ecodr2)
    assert [xx==yy for (xx,yy) in zip(ecodr2.index,ecoliv.index)]
    ecoliv.loc[:,'cz-e16'] = ecodr2.loc[:,'cz']#ecodr2.cz.to_numpy()
    ecoliv.loc[:,'grpcz-e17'] = ecodr2.loc[:,'grpcz_e17']#ecodr2.grpcz.to_numpy()
    ecoliv.loc[:,'absrmag-e16'] = ecodr2.loc[:,'absrmag']#ecodr2.absrmag.to_numpy()
    ecoliv.loc[:,'fm15'] = ecodr2.loc[:,'fm15']
    
    cznew_over_czold = (ecoliv.cz / ecoliv['cz-e16'])
    ecoliv.loc[:,'logmstar'] = np.log10(10**ecoliv.logmstar * (cosmo.luminosity_distance(np.array(ecoliv.cz)/cspeed).to_value())**2./(ecoliv['cz-e16']/hubble_const)**2. ) # like L = 4 pi d^2 f
    # note: don't need to do absrmag, calculated above using up-to-date cz

    # scale mhidet_a100
    sel = (ecoliv.mhidet_a100>0)# this means has a detection from ALFALFA-100
    cznew_over_czalfa = np.zeros(np.sum(sel)) + 1.
    eco_agcnr = ecoliv[sel].agcnr_a100.to_numpy()
    eco_cz = ecoliv[sel].cz.to_numpy()
    a100 = pd.read_csv("a100.csv")
    a100vhel = a100.Vhelio.to_numpy()
    a100agcnr = a100.AGCNr.to_numpy() 
    for ii,agcnumber in enumerate(eco_agcnr):
        alfasel = np.where(a100agcnr == agcnumber)
        alfa_v_value = a100vhel[alfasel]
        cznew_over_czalfa[ii] = eco_cz[ii] / alfa_v_value # cz_new / cz_alpha100
    
    ecoliv.loc[sel,'mhidet_a100'] = ecoliv.loc[sel,'mhidet_a100'] * (cznew_over_czalfa)**2. * (cosmo.luminosity_distance(np.array(ecoliv.loc[sel,'cz']/cspeed)).to_value()**2./(ecoliv.loc[sel,'cz']/hubble_const)**2.) / (1+(ecoliv.loc[sel,'cz']/cspeed))**2.
    print(np.max(cznew_over_czalfa),np.min(cznew_over_czalfa))

    # scale mhilim_a100
    ecoliv.loc[:,'mhilim_a100'] = ecoliv.loc[:,'mhilim_a100'] * (cznew_over_czold)**2. * (cosmo.luminosity_distance(np.array(ecoliv.loc[:,'cz']/cspeed)).to_value()**2./(ecoliv.loc[:,'cz']/hubble_const)**2.) / (1+(ecoliv.loc[:,'cz']/cspeed))**2

    # now update mhidet and mhilim and limsigma
    # NOTE: don't need to change mhidet or mhilim b/c these inherited from RESOLVE (so consistent with existing RESOLVE cz's
    # which are inherited into ECO DR3)
    sel = (ecoliv.resname=='notinresolve')
    ecoliv.loc[sel,'mhidet'] = ecoliv.loc[sel,'mhidet_a100']
    ecoliv.loc[sel,'mhilim'] = ecoliv.loc[sel,'mhilim_a100']
    ecoliv.loc[sel,'limsigma'] = ecoliv.loc[sel,'mhilim'] / ecoliv.loc[sel,'limmult']
   
    # rescale mhidet and mhilim for RESOLVE, re-sub into ECO
    resphot.loc[:,'mhidet'] = (resphot.mhidet / (1+resphot.cz/cspeed)**2.) * (cosmo.luminosity_distance(np.array(resphot.cz)/cspeed).to_value()**2. / (resphot.cz/hubble_const)**2.)
    resphot.loc[:,'mhilim'] = (resphot.mhilim / (1+resphot.cz/cspeed)**2.) * (cosmo.luminosity_distance(np.array(resphot.cz)/cspeed).to_value()**2. / (resphot.cz/hubble_const)**2.)
    ecoliv = ecoliv.reset_index().set_index('resname').sort_index()
    resatmp = resphot.loc[[ii for ii in resphot.index if 'rs' in ii],:].sort_index()
    ecoliv.loc[resatmp.index,'mhidet'] = resatmp.loc[resatmp.index,'mhidet']
    ecoliv.loc[resatmp.index,'mhilim'] = resatmp.loc[resatmp.index,'mhilim']
    ecoliv = ecoliv.reset_index().set_index('name').sort_index()
 
    # re-calculate logmgas and type flags
    out1,out2 = calculate_logmgas_hutchens23(ecoliv.mhidet,ecoliv.mhilim,ecoliv.mhi_corr,ecoliv.emhi_corr_sys,ecoliv.emhi_corr_rand,ecoliv.confused,ecoliv.modelu_j,ecoliv.b_a,ecoliv.logmstar)
    ecoliv.loc[:,'logmgas'] = out1
    ecoliv.loc[:,'logmgastype'] = out2

    logmgas_a100, typeflag_a100 = calculate_logmgas_hutchens23(ecoliv.mhidet_a100, ecoliv.mhilim_a100, np.zeros(len(ecoliv)), np.zeros(len(ecoliv)), np.zeros(len(ecoliv)), ecoliv.confused_a100,\
            ecoliv.modelu_j, ecoliv.b_a, ecoliv.logmstar)
    ecoliv.loc[:,'logmgas_a100'] = logmgas_a100
    ecoliv.loc[:,'logmgastype_a100'] = typeflag_a100
    
    resout1,resout2 = calculate_logmgas_hutchens23(resphot.mhidet,resphot.mhilim,resphot.mhi_corr,resphot.emhi_corr_sys,resphot.emhi_corr_rand,resphot.confused,resphot.modelu_j,resphot.b_a,resphot.logmstar)
    resphot.loc[:,'logmgas'] = resout1
    resphot.loc[:,'logmgastype'] = resout2

    #####################################################################
    #####################################################################
    #####################################################################
    # Now need to deal with the ~113 galaxies that had r50==r90 or r50<0.
    
    bad_radius_sel = (ecoliv.r50==ecoliv.r90) | (ecoliv.r50<0)
    ecoliv.loc[:,'badsdssphot'] = bad_radius_sel.astype(int)
    linefn = lambda x,a,b: a*x+b
    params0=(1,-0.95-0.5)
    params1=(-0.5,-0.54-0.5)
    outlier_in_logmstar = ((ecoliv.logmstar < linefn(ecoliv.absgmag_withoutparens,*params1)) | (ecoliv.absgmag_withoutparens==0)) & (ecoliv.badsdssphot>0)
    outlier_in_absrmag_withoutparens = ((ecoliv.absrmag_withoutparens < linefn(ecoliv.absgmag_withoutparens,*params0)) | (ecoliv.absgmag_withoutparens==0) | (ecoliv.absrmag_withoutparens==0)) & (ecoliv.badsdssphot>0)
    # <--plot to check

    # if outlier in absrmag but not in logmstar, then estimate from forward fit to RESOLVE M_r vs. M* -- need to change in absrmag and absrmag_withoutparens
    forwardfitresolve=np.poly1d(np.polyfit(resphot.logmstar, resphot.absrmag_withoutparens, 1))
    ecoliv.loc[(outlier_in_absrmag_withoutparens)&(~outlier_in_logmstar),'absrmag_withoutparens'] = forwardfitresolve(ecoliv.loc[(outlier_in_absrmag_withoutparens)&(~outlier_in_logmstar),'logmstar'])
    ecoliv.loc[(outlier_in_absrmag_withoutparens)&(~outlier_in_logmstar),'absrmag'] = ecoliv.loc[(outlier_in_absrmag_withoutparens)&(~outlier_in_logmstar),'absrmag_withoutparens'].astype(str)

    # for the rest (outlier in logmstar but not in absrmag, or outlier in both, set nominal values in parens)
    ecoliv['logmstar_withoutparens'] = ecoliv.logmstar.copy()
    ecoliv = ecoliv.astype({'logmstar':str})
    for nn in ecoliv.loc[(outlier_in_logmstar)&(~outlier_in_absrmag_withoutparens),:].index:
        ecoliv.loc[nn,'logmstar'] = '('+str(ecoliv.loc[nn,'logmstar'])+')'
        ecoliv.loc[nn,'absrmag'] = '('+str(ecoliv.loc[nn,'absrmag'])+')'
    for nn in ecoliv.loc[(outlier_in_logmstar)&(outlier_in_absrmag_withoutparens),:].index:
        ecoliv.loc[nn,'logmstar'] = '('+str(ecoliv.loc[nn,'logmstar'])+')'
        ecoliv.loc[nn,'absrmag'] = '('+str(ecoliv.loc[nn,'absrmag'])+')'

    #######################################################
    #######################################################
    #######################################################
    # Remove rs1492, duplicates before group finding
    # also remove 
    resphot = resphot[resphot.index!='rs1492'] 
    ecoliv = ecoliv[ecoliv.resname!='rs1492'] # get rid of rs1492
    ecoliv = ecoliv[ecoliv.index!='ECO13245'] # get rid of ECO13245, it's a star
    ecoliv = ecoliv[ecoliv.dup<1] # get rid of duplicates 

    ecoliv.rename(columns={'dup':'vif'},inplace=True)

    #######################################################
    #######################################################
    # write comoving distances
    ecoliv.loc[:,'loscmvgdist'] = cosmo.comoving_distance(np.array(ecoliv.loc[:,'cz'])/cspeed).value # in Mpc
    resphot.loc[:,'loscmvgdist'] = cosmo.comoving_distance(np.array(resphot.loc[:,'cz'])/cspeed).value # in Mpc
    
    #######################################################
    #######################################################
    #######################################################
    # Group finding
    ecoliv.to_csv('.ECODR3_for_groups.csv')
    subprocess.run(["python3","make_ECO_w_fofgroups.py"])
    ecodr3groups = pd.read_csv(".ECO_DR3_groups_only.csv").set_index('name')
    ecoliv = pd.concat([ecoliv,ecodr3groups],axis=1)

    tmp = resphot.copy()
    tmp.loc[:,'name'] = tmp.index
    tmp = tmp.reset_index()
    tmp.to_csv('.RESOLVEDR4_for_groups.csv',index=False)
    ecoliv.to_csv('.ECO_csv_for_RESOLVE_FoF.csv')
    subprocess.run(['python3','make_RESOLVE_w_fofgroups.py'])
    resdr4groups = pd.read_csv(".RESOLVE_DR4_groups_only.csv").set_index('name')
    resphot = pd.concat([resphot,resdr4groups],axis=1)

    ########################################################
    # Output files
    print(resphot)
    print(ecoliv)
    resphot.loc[:,'name'] = resphot.index
    resphot.to_csv(resoutputfile,index=False)
    ecoliv.to_csv(outputfile)
