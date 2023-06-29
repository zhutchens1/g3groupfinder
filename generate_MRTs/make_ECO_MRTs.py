import numpy as np
import pandas as pd
import sys
papertitle='The RESOLVE and ECO Gas in Galaxy Groups Initiative: The Group Finder and the Group HI-Halo Mass Relation'
authorlist='Hutchens Z.L., Kannappan S.J., Berlind A.A., Asad M., Eckert K.D., Stark D.V., Carr D.S., Castelloe E.R., Baker A.J., Hess K.M., Moffett A.J., Norris M.A., Croton D.'

dr3path = '../resolve_and_eco/ECOdata_G3catalog_luminosity.csv'##'/srv/two/cielo/zhutchen/database/make_ECO_DR3/ECODR3_052523.csv'





#########################################################################
############# ECO DR3 Photometry Table
#########################################################################

def print_for_online(dct):
    print("### PRINTING OUTPUT FOR ONLINE AAS MRT MAKER ###")
    for k in dct.keys():
        print(dct[k][1])
    print('-----')
    for k in dct.keys():
        print(k)
    print('-----')
    for k in dct.keys():
        print(dct[k][2])

### PHOT TABLE ###
print('### PHOT TABLE ###')
photdf = pd.read_csv(dr3path)
photdf = photdf[photdf.vif<1]
photdf.replace(-99,-999,inplace=True)

photdf.loc[:,'modelu-r']=photdf.modelu_r
photdf.loc[:,'modelu-j']=photdf.modelu_j
photdf.loc[:,'modelu-k']=photdf.modelu_k
photdf.loc[:,'modelg-r']=photdf.modelg_r
photdf.loc[:,'modelu-rcorr']=photdf.modelu_rcorr
photdf.loc[:,'b/a']=photdf.b_a
photdf.loc[:,'mu-delta']=photdf.mu_delta
photcolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    'resname':[None,'---','RESOLVE Galaxy Identifier'],\
    'radeg':[5,'deg','RA J2000'],\
    'dedeg':[5,'deg','Dec J2000'],\
    'cz':[1,'km/s','Galaxy Local Group-corrected Recession Velocity of Galaxy'],\
    'cz-e16':[1,'km/s','Galaxy Local Group-corrected Recession Velocity of Galaxy from E16'],\
    #'czupdateflag':[None,'---','Update Flag for cz (1)'],\
    'vif':[None,'---','Visual Inspection Flag (1)'],\
    'fm15':[None,'---','ECO DR1 Moffett+15 Membership Flag'],\
    #'morphel':[None,'---','Galaxy Morphology Flag (3)'],\
    #'fmorphel':[None,'---','Morphology Source Flag (4)'],\
    'umag':[3,'mag','Apparent Magnitude in SDSS u Band (2)'],\
    'e_umag':[3,'mag','Error in umag'],
    'gmag':[3,'mag','Apparent Magnitude in SDSS g Band (2)'],\
    'e_gmag':[3,'mag','Error in gmag'],\
    'rmag':[3,'mag','Apparent Magnitude in SDSS r Band (2)'],\
    'e_rmag':[3,'mag','Error in rmag'],\
    'imag':[3,'mag','Apparent Magnitude in SDSS i Band (2)'],\
    'e_imag':[3,'mag','Error in imag'],\
    'zmag':[3,'mag','Apparent Magnitude in SDSS z Band (2)'],\
    'e_zmag':[3,'mag','Error in zmag'],\
    'nuvmag':[3,'mag','Apparent Magnitude in GALEX NUV Band (2)'],\
    'e_nuvmag':[3,'mag','Error in NUVmag'],\
    '2jmag':[3,'mag','Apparent Magnitude in 2MASS J Band (2)'],\
    'e_2jmag':[3,'mag','Error in 2jmag'],\
    '2hmag':[3,'mag','Apparent Magnitude in 2MASS H Band (2)'],\
    'e_2hmag':[3,'mag','Error in 2hmag'],\
    '2kmag':[3,'mag','Apparent Magnitude in 2MASS K Band (2)'],\
    'e_2kmag':[3,'mag','Error in 2kmag'],\
    'extu':[3,'mag','Foreground Extinction in SDSS u Band'],\
    'extg':[3,'mag','Foreground Extinction in SDSS g Band'],\
    'extr':[3,'mag','Foreground Extinction in SDSS r Band'],\
    'exti':[3,'mag','Foreground Extinction in SDSS i Band'],\
    'extz':[3,'mag','Foreground Extinction in SDSS z Band'],\
    'extnuv':[3,'mag','Foreground Extinction in GALEX NUV Band'],\
    'extj':[3,'mag','Foreground Extinction in 2MASS J Band'],\
    'exth':[3,'mag','Foreground Extinction in 2MASS H Band'],\
    'extk':[3,'mag','Foreground Extinction in 2MASS K Band'],\
    'rkcorr':[3,'mag','k-correction in SDSS r Band'],\
    'absrmag':[3,'mag','Abs. Mag. in SDSS r Band (3)'],\
    'absrmag-e16':[3,'mag','Abs. Mag. in SDSS r Band from E16 (3)'],\
    'modelu-r':[3,'mag','Rest Frame SED Modeled (u-r) Color (4)'],\
    'modelu-j':[3,'mag','Rest Frame SED Modeled (u-J) Color (4)'],\
    'modelu-k':[3,'mag','Rest Frame SED Modeled (u-K) Color (4)'],\
    'modelg-r':[3,'mag','Rest Frame SED Modeled (g-r) Color (4)'],\
    'modelu-rcorr':[3,'mag','Rest Frame SED Modeled (u-r) Color Corrected for Internal Extinction (5)'],\
    'r50':[3,'arcsec','Half-light Radius in r Band'],\
    'r90':[3,'arcsec','90%-light Radius in r Band'],\
    'b/a':[3,'---','Axial Ratio'],\
    'logmstar':[2,'solMass','log Galaxy Stellar Mass (6)'],\
    'mu-delta':[2,'solMass/kpc2','Morphological metric (7)'],\
})
photdf = photdf[[k for k in photcolumns.keys()]]
photdf = photdf.round(dict((k,photcolumns[k][0]) for k in photcolumns.keys() if photcolumns[k][0] is not None))
photdf.to_csv("tab4_for_mrtmaker.csv",index=False, header=False)
print_for_online(photcolumns)

photMRTnotes=['Notes.','All N/A values indicated by -999.',\
    '(1): Value of 1 indicates that the ECO DR3 cz differs from ECO DR2 cz by >35 km/s.',\
    '(2): No duplicate galaxies are included in this table. Values <0 indicate the galaxy was visually confirmed to not be duplicate. The reason for the manual inspection is given from the following key:\n-1: Galaxy is the primary target within a merger remnant representing one catalog object.\n-2: Galaxy was duplicated in ECO DR2 and only this entry is kept in ECO DR3.\n-3: ECO DR2 included a second entry for this galaxy that was centered on a spiral arm or other feature.\n-4: Galaxy has a nearby object at similar redshift, but no obvious connecting material or morphological features indicate a merger in progress.\n-5: Galaxy is within the effective radius of a second galaxy or vice versa.\n-6: Galaxy is undergoing a merger but both cores are visually identifiable.',\
    '(3) Value `E`=early Type and `L`=late type as originally calculated by Moffett+2015. If galaxy was not part of the M15 sample, the value is calculated using the quantitative cut provided by M15.',\
    '(4) Value of 1 indicates morphology classified using quantitative cut from M15. Value of 0 indicates by-eye classification from Lintott+2011.',
    '(5) Custom remeasured value from Eckert+2016, provided without foreground extinction correction.',\
    '(6) Custom remeasured value from Eckert+2015, foreground extinction correction included.',\
    '(7) Foreground extinction corrections and k-corrections implicitly included.',\
    '(8) Extinction corrections performed without deep GALEX NUV data are not consistent with those having deep NUV data (see Appendix A). We recommend modelu_r for ECO analyses.',\
    '(9) Proxy for galaxy morphology (see Kannappan+2013, Eckert+2016). Values calculated as difference between stellar mass surface density within r50 and stellar mass surface density between r50 and r90.']
with open('photlatexnotes.txt','w') as ff:
    ff.write(' '.join(photMRTnotes))
ff.close()


latexdescriptions = []
for ii,k in enumerate(photcolumns.keys()):
    if False: #photcolumns[k][2].endswith(')'):
        latexdescriptions.append(photcolumns[k][2][0:-3])
    else:
        latexdescriptions.append(photcolumns[k][2])
photlatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
if __name__=='__main__':
    photlatexsample.to_latex("photlatex.txt",index=False)
#########################################################################
############# ECO DR3 21cm Catalog
#########################################################################
hidf = pd.read_csv(dr3path)
hidf = hidf[hidf.vif<1]
hidf.replace(-99,-999,inplace=True)
for k in hidf.keys():
    if k.endswith('_a100'):
        hidf.loc[:,k.replace('_','-')]=hidf[k]
hicolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    #'resname':[None,'---','RESOLVE Galaxy Identifier'],\
    #'radeg':[5,'deg','RA J2000'],\
    #'dedeg':[5,'deg','Dec J2000'],\
    #'cz':[1,'km/s','Galaxy Local Group-corrected Recession Velocity of Galaxy'],\
    'mhidet':[2,'solMass','Galaxy HI Mass from ALFALFA-100 or RESOLVE Detections (1)'],\
    'emhidet':[2,'solMass','Statistical Uncertainty on Galaxy HI Mass (even if confused)'],\
    'confused':[None,'---','Confusion Flag for `mhidet` (2)'],\
    'mhilim':[2,'solMass','Upper Limit on HI mass from ALFALFA-100 or RESOLVE (3)'],\
    #'limflag':[None,'---','Upper Limit Flag (4)'],\
    'limsigma':[2,'solMass','Integrated RMS Noise for Non-detections (4)'],\
    'limmult':[1,'---','Upper Limit Level, 3 or 5 (5)'],\
    'hirms':[4,'Jy','RMS Noise Level of HI Spectrum (6)'],\
    'hitelescope':[None,'---','Source of HI Data (7)'],\
    'peaksnhi':[1,'---','Peak SNR of 21cm Line Detection'],\
    'logmgas':[2,'solMass','Best Estimate of Atomic Gas Mass (8)'],\
    'logmgastype':[None,'---','Type Flag for `logmgas` (9)'],\
    'mhidet-a100':[2,'solMass','Galaxy HI Mass from ALFALFA-100 21cm Detection (1)'],\
    'emhidet-a100':[2,'solMass','Statistical Uncertainty on ALFALFA-100 HI Mass (even if confused)'],\
    'confused-a100':[None,'---','Confusion Flag for `mhidet-a100` (2)'],\
    'agcnr-a100':[None,'---','Arecibo General Catalog Number'],\
    'mhilim-a100':[None,'solMass','Upper Limit on ALFALFA-100 HI Mass (3)'],\
    'hirms-a100':[4,'Jy','RMS Noise Level of HI Spectrum (6)'],\
    'peaksnhi-a100':[1,'---','Peak SNR of 21cm Line Detection'],\
    'logmgas-a100':[2,'solMass','Best Estimate of Atomic Gas Mass from ALFALFA-100 Data (8)'],\
    'logmgastype-a100':[None,'---','Type Flag for `logmgas-a100` (9)'],\
})

hidf = hidf[[k for k in hicolumns.keys()]]
hidf = hidf.round(dict((k,hicolumns[k][0]) for k in hicolumns.keys() if hicolumns[k][0] is not None))
hidf.to_csv('tab7_for_mrtmaker.csv',index=False,header=False)
print_for_online(hicolumns)

hiMRTnotes=[
    'All columns with `a100` suffix use same conventions and methods as the main 21cm data columns, but\nthey exclude deeper RESOLVE-A region 21cm observations, providing a uniform ALFALFA-100-based dataset for ECO.',\
    '(1) 0 if not detected or not observed. If observation is confused, this column reports the total HI mass.',\
    '(2) 1 = observation likely confused; 0 = not confused',\
    '(3) 0 if detected in HI or not observed',\
    '(4) Based on estimated RMS noise level and estimated W20 (see Appendix A); 0 for detections.\nUpper limits computed at 3 or 5 times this value.',\
    '(5) Level of upper limit, 3 for RESOLVE sources and 5 for rest of ECO in ALFALFA-100 field \n(see Appendix A); 0 for detections.',\
    '(6) Measured from 21cm spectra for RESOLVE-A galaxies, drawn from ALFALFA-100 catalog for \ndetected ECO galaxies, or estimated for non-detections.',\
    '(7) Possible values: `ALFALFA`=ALFALFA HI survey, `GBT`=Green Bank Telescope, `AO`=Arecibo\nObservatory, `S05`=Springob+2005 catalog.',\
    '(8) Atomic gas mass = 1.4 * HI mass (He correction).',\
    '(9) 0 = real HI measurement, 1 = estimate from photometric gas fractions',\
]

with open('hilatexnotes.txt','w') as ff:
    ff.write(' '.join(hiMRTnotes))
ff.close()

latexdescriptions = []
for ii,k in enumerate(hicolumns.keys()):
    if False:#hicolumns[k][2].endswith(')'):
        latexdescriptions.append(hicolumns[k][2][0:-4])
    else:
        latexdescriptions.append(hicolumns[k][2])
hilatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
if __name__=='__main__':
    hilatexsample.to_latex("hilatex.txt",index=False)

#########################################################################
############# ECO DR3 Groups Table
#########################################################################

grpdf = pd.read_csv(dr3path)
grpdf = grpdf[grpdf.vif<1]
grpdf.replace(-99,-999,inplace=True)
grpcolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    #'resname':[None,'---','RESOLVE Galaxy Identifier'],\
    #'radeg':[5,'deg','RA J2000'],\
    #'dedeg':[5,'deg','Dec J2000'],\
    #'cz':[1,'km/s','Galaxy Local Group-corrected Recession Velocity of Galaxy'],\
    'invole17':[None,'---','In-Volume Flag for E17 Groups (1)'],\
    'g3grp':[None,'---','G3 Group Identifier'],\
    'g3grpradeg':[5,'deg','G3 Group Center RA J2000'],\
    'g3grpdedeg':[5,'deg','G3 Group Center Dec J2000'],\
    'g3grpcz':[1,'km/s','G3 Group Center Velocity'],\
    'involg3':[None,'---','In-Volume Flag for G3 Groups'],\
    'g3grpngi':[None,'---','Number of Giants in G3 Group'],\
    'g3grpndw':[None,'---','Number of Dwarfs in G3 Group'],\
    'g3grpabsrmag':[2,'mag','G3 Group-Integrated r-Band Abs. Mag.'],\
    'g3grpgiantabsrmag':[2,'mag','G3 Group-Integrated Giant r-Band Abs. Mag.'],\
    'g3logmhvir':[2,'solMass','G3 HAM log Group Mass (337b) (2)'],\
    'g3logmh200':[2,'solMass','G3 HAM log Group Mass (200b) (3)'],\
    'g3grpmhi':[2,'solMass','G3 log Group-Integrated HI Mass'],\
    'ccb-remapped':[None,'---','Boundary Completeness Correction Factor (4)'],\
    'fofgrp':[None,'---','FoF Group Identifier'],\
    'fofgrpradeg':[5,'deg','FoF Group Center RA J2000'],\
    'fofgrpdedeg':[5,'deg','FoF Group Center Dec J2000'],\
    'fofgrpcz':[1,'km/s','FoF Group Center Velocity'],\
    'involfof':[None,'---','In-Volume Flag for FoF Groups'],\
    'fofgrpn':[None,'---','Number of Galaxies in FoF Group'],\
    'fofgrpabsrmag':[2,'mag','FoF Group-Integrated r-Band Abs. Mag.'],\
    'fofgrpgiantabsrmag':[2,'mag','FoF Group-Integrated Giant r-Band Abs. Mag.'],\
    'foflogmhvir':[2,'solMass','FoF HAM log Group Mass (337b) (2)'],\
    'foflogmh200':[2,'solMass','FoF HAM log Group Mass (200b) (3)'],\
    'fofgrpmhi':[2,'solMass','FoF log Group-Integrated HI Mass']
})
grpdf = grpdf[[k for k in grpcolumns.keys()]]
grpdf = grpdf.round(dict((k,grpcolumns[k][0]) for k in grpcolumns.keys() if grpcolumns[k][0] is not None))
grpdf.to_csv("tab5_for_mrtmaker.csv",index=False,header=False)
print_for_online(grpcolumns)

grpMRTnotes=[\
    'Note (1): HAM performed using Tinker et al. 2008 halo mass function.',\
    'Note (2): HAM performed using Warren et al. 2006 halo mass function.',\
    'Note (3): Correction factors for redshift incompleteness due to group peculiar velocities extending\nbeyond the RESOLVE/ECO volumes.\
    Affects only Coma and two massive groups.\nValues as computed by Eckert+2016, mapped to new G3 group catalog.'
]
with open('grplatexnotes.txt','w') as ff:
    ff.write(' '.join(grpMRTnotes))
ff.close()

latexdescriptions = []
for ii,k in enumerate(grpcolumns.keys()):
    if False:#grpcolumns[k][2].endswith(')'):
        latexdescriptions.append(grpcolumns[k][2][0:-3])
    else:
        latexdescriptions.append(grpcolumns[k][2])
grplatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
if __name__=='__main__':
    grplatexsample.to_latex("grplatex.txt",index=False)


############################################################
############################################################
# Duplicate Table
dupdf = pd.read_csv(dr3path)
dupcolumns = dict({
    'dr2name':[None,'---','ECO DR2 Identifier (1)'],\
    'match':[None,'---','ECO DR3 Identifier (2)'],\
    'vif':[None,'---','Duplicate Classification (3)'],\
})
if __name__=='__main__':
    dupdf.to_latex('duplatex.txt',index=False)
