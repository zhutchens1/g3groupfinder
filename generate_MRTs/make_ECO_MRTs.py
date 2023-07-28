import numpy as np
import pandas as pd
import sys

def render_cols_latex(dictionary,filename):
    outfile=open(filename,'w')
    for ii,kk in enumerate(dictionary.keys()):
        desig = kk
        if '_' in desig:
            desig=desig.replace('_','\\_')
        outfile.write('{} & '.format(ii+1)+' \\texttt{'+desig+'} & '+dictionary[kk][2]+'\\\\ \n')
        #if dictionary[kk][2].endswith(')'):
        #    tmp = dictionary[kk][2].split(' ')
        #    outfile.write('{} & '.format(ii+1)+' '.join(tmp[:-1])+' \\texttt{'+kk+'} '+tmp[-1]+' \\\\ \n')
        #else:

papertitle='The RESOLVE and ECO Gas in Galaxy Groups Initiative: The Group Finder and the Group HI-Halo Mass Relation'
authorlist='Hutchens Z.L., Kannappan S.J., Berlind A.A., Asad M., Eckert K.D., Stark D.V., Carr D.S., Castelloe E.R., Baker A.J., Hess K.M., Moffett A.J., Norris M.A., Croton D.'

dr3path = '../resolve_and_eco/ECOdata_G3catalog_luminosity.csv'##'/srv/two/cielo/zhutchen/database/make_ECO_DR3/ECODR3_052523.csv'
df = pd.read_csv(dr3path)
df = df[df.vif<1]
df.rename(columns={xx:xx[:-2] for xx in df.keys() if xx.startswith('g3') and xx.endswith('_l')},inplace=True)
df.rename(columns={'ccb_remapped':'ccbremapped', 'kcorrr':'rkcorr', 'g3logmh':'g3logmhvir', 'vhel':'czhel', 'cz-e16':'cze16'},inplace=True)
df.rename(columns={kk:kk.replace('-','') for kk in df.keys() if '-' in kk},inplace=True)
df.rename(columns={kk:kk.replace('_','') for kk in df.keys() if kk.startswith('model')},inplace=True)

df.loc[:,'involg3'] = np.logical_and(df.g3grpcz>3000,df.g3grpcz<7000).astype(int)
df.loc[:,'involfof'] = np.logical_and(df.fofgrpcz>3000,df.fofgrpcz<7000).astype(int) 
df.loc[:,'invole17'] = np.logical_and(df['grpcze17']>3000,df['grpcze17']<7000).astype(int) 

# round HI mass cols.
factor=1e7#2.36e5 * (3000/70.)**2. * 0.1 # S16 precision was 0.1 Jy km/s
df.loc[:,'mhidet'] = factor*(df.mhidet/factor).round(2)
df.loc[:,'emhidet'] = factor*(df.emhidet/factor).round(2)
df.loc[:,'mhilim'] = factor*(df.mhilim/factor).round(2)
df.loc[:,'limsigma'] = factor*(df.limsigma/factor).round(2)
df.loc[:,'mhidet_a100'] = factor*(df.mhidet_a100/factor).round(2)
df.loc[:,'emhidet_a100'] = factor*(df.emhidet_a100/factor).round(2)
df.loc[:,'mhilim_a100'] = factor*(df.mhilim_a100/factor).round(2)
df.rename(columns={'emhidet':'e_mhidet', 'emhidet_a100':'e_mhideta100'},inplace=True)


df.to_csv(".tmp_ecodr3_file.csv",index=False)
dr3path = ".tmp_ecodr3_file.csv"


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

#photdf.loc[:,'modelu-r']=photdf.modelu_r
#photdf.loc[:,'modelu-i']=photdf.modelu_i
#photdf.loc[:,'modelu-j']=photdf.modelu_j
#photdf.loc[:,'modelu-k']=photdf.modelu_k
#photdf.loc[:,'modelg-r']=photdf.modelg_r
#photdf.loc[:,'modelg-r']=photdf.modelg_r
#photdf.loc[:,'modelu-rcorr']=photdf.modelu_rcorr
#photdf.loc[:,'modelnuv-rcorr']=photdf.modelnuv_rcorr
photdf.rename(columns={kk:kk.replace('_','-') for kk in photdf.keys() if ('model' in kk) and ('_' in kk)},inplace=True)
photdf.loc[:,'axialratio']=photdf.b_a
photdf.loc[:,'mudelta']=photdf.mu_delta
photcolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    'resname':[None,'---','RESOLVE Galaxy Identifier'],\
    'radeg':[5,'deg','RA J2000'],\
    'dedeg':[5,'deg','Dec J2000'],\
    'cz':[1,'km/s','Local Group-corrected Recession Velocity of Galaxy'],\
    'cze16':[1,'km/s','Local Group-corrected Recession Velocity of Galaxy from E16'],\
    'czhel':[1,'km/s','Heliocentric Recession Velocity of Galaxy'],\
    'loscmvgdist':[1,'Mpc','Line-of-Sight Comoving Distance to Galaxy'],\
    'vif':[None,'---','Visual Inspection Flag (1)'],\
    'fm15':[None,'---','ECO DR1 Moffett+15 Membership Flag'],\
    'nuvmag':[3,'mag','Apparent Magnitude in GALEX NUV Band (2)'],\
    'e_nuvmag':[3,'mag','Error in NUVmag'],\
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
    '2jmag':[3,'mag','Apparent Magnitude in 2MASS J Band (2)'],\
    'e_2jmag':[3,'mag','Error in 2jmag'],\
    '2hmag':[3,'mag','Apparent Magnitude in 2MASS H Band (2)'],\
    'e_2hmag':[3,'mag','Error in 2hmag'],\
    '2kmag':[3,'mag','Apparent Magnitude in 2MASS K Band (2)'],\
    'e_2kmag':[3,'mag','Error in 2kmag'],\
    'uymag':[3,'mag','Apparent Magnitude in UKIDSS Y Band (2)'],\
    'e_uymag':[3,'mag','Error in uymag'],\
    'uhmag':[3,'mag','Apparent Magnitude in UKIDSS H Band (2)'],\
    'e_uhmag':[3,'mag','Error in uhmag'],\
    'ukmag':[3,'mag','Apparent Magnitude in UKIDSS K Band (2)'],\
    'e_ukmag':[3,'mag','Error in ukmag'],\
    'extnuv':[3,'mag','Foreground Extinction in GALEX NUV Band'],\
    'extu':[3,'mag','Foreground Extinction in SDSS u Band'],\
    'extg':[3,'mag','Foreground Extinction in SDSS g Band'],\
    'extr':[3,'mag','Foreground Extinction in SDSS r Band'],\
    'exti':[3,'mag','Foreground Extinction in SDSS i Band'],\
    'extz':[3,'mag','Foreground Extinction in SDSS z Band'],\
    'exty':[3,'mag','Foreground Extinction in UKIDSS Y Band'],\
    'extj':[3,'mag','Foreground Extinction in 2MASS J Band'],\
    'exth':[3,'mag','Foreground Extinction in 2MASS and UKIDSS H Bands'],\
    'extk':[3,'mag','Foreground Extinction in 2MASS and UKIDSS K Bands'],\
    #'rkcorr':[3,'mag','k-correction in SDSS r Band'],\
    'badrphot':[1,'---','Bad r-band Photometry Flag (3)'],\
    'logmstar':[2,'solMass','Log Galaxy Stellar Mass (4)'],\
    'absrmag':[3,'mag','Absolute Magnitude in SDSS r Band (5)'],\
    'absrmage16':[3,'mag','Absolute Magnitude in SDSS r Band from E16 (5)'],\
    'modelabsrmag':[3,'mag','Rest Frame SED Modeled Absolute Magnitude in SDSS r Band (6)'],\
    'modelnuvr':[3,'mag','Rest Frame SED Modeled (NUV-r) Color (6)'],\
    'modelur':[3,'mag','Rest Frame SED Modeled (u-r) Color (6)'],\
    'modelui':[3,'mag','Rest Frame SED Modeled (u-i) Color (6)'],\
    'modeluj':[3,'mag','Rest Frame SED Modeled (u-J) Color (6)'],\
    'modeluk':[3,'mag','Rest Frame SED Modeled (u-K) Color (6)'],\
    'modelgr':[3,'mag','Rest Frame SED Modeled (g-r) Color (6)'],\
    'modelgi':[3,'mag','Rest Frame SED Modeled (g-i) Color (6)'],\
    'modelgj':[3,'mag','Rest Frame SED Modeled (g-J) Color (6)'],\
    'modelgk':[3,'mag','Rest Frame SED Modeled (g-K) Color (6)'],\
    'modelurcorr':[3,'mag','Rest Frame SED Modeled (u-r) Color Corrected for Internal Extinction (7)'],\
    'r50':[3,'arcsec','Half-light Radius in r Band (8)'],\
    'r90':[3,'arcsec','90%-light Radius in r Band (8)'],\
    'axialratio':[3,'---','Axial Ratio'],\
    'mudelta':[2,'solMass/kpc2','Morphological metric (9)'],\
})
photdf = photdf[[k for k in photcolumns.keys()]]
photdf = photdf.round(dict((k,photcolumns[k][0]) for k in photcolumns.keys() if photcolumns[k][0] is not None))
photdf.to_csv("tab4_for_mrtmaker.csv",index=False, header=False)
print_for_online(photcolumns)

render_cols_latex(photcolumns,'photlatex.txt')

#photlatexfile=open('photlatex.txt','w')
#for ii,kk in enumerate(photcolumns.keys()):
#    desig = kk
#    if '_' in desig:
#        desig=desig.replace('_','\\_')
#    photlatexfile.write('{} & '.format(ii+1)+' \\texttt{'+desig+'} & '+photcolumns[kk][2]+'\\\\ \n')
    #if photcolumns[kk][2].endswith(')'):
    #    tmp = photcolumns[kk][2].split(' ')
    #    photlatexfile.write('{} & '.format(ii+1)+' '.join(tmp[:-1])+' \\texttt{'+kk+'} '+tmp[-1]+' \\\\ \n')
    #else:
    #    photlatexfile.write('{} & '.format(ii+1)+photcolumns[kk][2]+' \\texttt{'+kk+'} '+' \\\\ \n')
#photlatexfile.close()

#latexdescriptions = []
#for ii,k in enumerate(photcolumns.keys()):
#    if photcolumns[k][2].endswith(')'):
#        latexdescriptions.append(photcolumns[k][2][0:-3])
#    else:
#        latexdescriptions.append(repr(photcolumns[k][2]+repr('\texttt{'+k+'} ')))

#        print(photcolumns[k][2]+' \\texttt{'+k+'} ')
#photlatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
#if __name__=='__main__':
#    photlatexsample.to_latex("photlatex.txt",index=False)

#########################################################################
############# ECO DR3 21cm Catalog
#########################################################################
hidf = pd.read_csv(dr3path)
hidf = hidf[hidf.vif<1]
hidf.replace(-99,-999,inplace=True)
for k in hidf.keys():
    if k.endswith('_a100'):
        hidf.loc[:,k.replace('_','')]=hidf[k]
hicolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    'mhidet':[1,'solMass','Galaxy HI Mass from ALFALFA-100 or RESOLVE Detection (1)'],\
    'e_mhidet':[1,'solMass','Statistical Uncertainty on Galaxy HI Mass from 21cm Detection (even if confused)'],\
    'confused':[None,'---','Confusion Flag (2)'],\
    'mhilim':[1,'solMass','Upper Limit on HI Mass from ALFALFA-100 or RESOLVE (3)'],\
    'limsigma':[1,'solMass','Integrated RMS Noise for Non-detection (4)'],\
    'limmult':[1,'---','Upper Limit Level, 3 or 5 (5)'],\
    'hirms':[4,'Jy','RMS Noise Level of HI Spectrum (6)'],\
    'hitelescope':[None,'---','Source of HI Data (7)'],\
    'peaksnhi':[1,'---','Peak SNR of 21cm Line Detection'],\
    'logmgas':[2,'solMass','Best Estimate of Atomic Gas Mass (8)'],\
    'logmgastype':[None,'---','Type Flag for Atomic Gas Mass Estimate (9)'],\
    'mhideta100':[1,'solMass','Galaxy HI Mass from ALFALFA-100 21cm Detection (1)'],\
    'e_mhideta100':[1,'solMass','Statistical Uncertainty on ALFALFA-100 HI Mass (even if confused)'],\
    'confuseda100':[None,'---','Confusion Flag for ALFALFA-100 Detection (2)'],\
    'agcnra100':[None,'---','Arecibo General Catalog Number'],\
    'mhilima100':[1,'solMass','Upper Limit on ALFALFA-100 HI Mass (3)'],\
    'hirmsa100':[4,'Jy','RMS Noise Level of HI Spectrum (6)'],\
    'peaksnhia100':[1,'---','Peak SNR of ALFALFA-100 21cm Line Detection'],\
    'logmgasa100':[2,'solMass','Best Estimate of Atomic Gas Mass from ALFALFA-100 Data (8)'],\
    'logmgastypea100':[None,'---','Type Flag for Atomic Gas Mass Estimate from ALFALFA-100 Data (9)'],\
})

hidf = hidf[[k for k in hicolumns.keys()]]
hidf = hidf.round(dict((k,hicolumns[k][0]) for k in hicolumns.keys() if hicolumns[k][0] is not None))
hidf.to_csv('tab7_for_mrtmaker.csv',index=False,header=False)
print_for_online(hicolumns)
render_cols_latex(hicolumns,'hilatex.txt')

#latexdescriptions = []
#for ii,k in enumerate(hicolumns.keys()):
#    if False:#hicolumns[k][2].endswith(')'):
#        latexdescriptions.append(hicolumns[k][2][0:-4])
#    else:
#        latexdescriptions.append(hicolumns[k][2])
#hilatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
#if __name__=='__main__':
#    hilatexsample.to_latex("hilatex.txt",index=False)

#########################################################################
############# ECO DR3 Groups Table
#########################################################################

grpdf = pd.read_csv(dr3path)
grpdf = grpdf[grpdf.vif<1]
grpdf.replace(-99,-999,inplace=True)
grpcolumns=dict({
    'name':[None,'---','ECO Galaxy Identifier'],\
    'invole17':[None,'---','In-Volume Flag for E17 Groups (1)'],\
    'g3grp':[None,'---','G3 Group Identifier'],\
    'g3grpradeg':[5,'deg','G3 Group Center RA J2000'],\
    'g3grpdedeg':[5,'deg','G3 Group Center Dec J2000'],\
    'g3grpcz':[1,'km/s','G3 Group Center cz (2)'],\
    'involg3':[None,'---','In-Volume Flag for G3 Groups (1)'],\
    'g3grpngi':[None,'---','Number of Giants in G3 Group'],\
    'g3grpndw':[None,'---','Number of Dwarfs in G3 Group'],\
    'g3grpabsrmag':[2,'mag','G3 Group-Integrated r-Band Abs. Mag.'],\
    'g3grpgiantabsrmag':[2,'mag','G3 Group-Integrated Giant r-Band Abs. Mag.'],\
    'g3logmhvir':[2,'solMass','G3 HAM Log Group Mass (337b) (3)'],\
    'g3logmh200':[2,'solMass','G3 HAM Log Group Mass (200b) (4)'],\
    'g3grpmhi':[2,'solMass','G3 Log Group-Integrated HI Mass'],\
    'g3fc':[1,'---','G3 Central Galaxy Flag (5)'],\
    'ccbremapped':[None,'---','Boundary Completeness Correction Factor (6)'],\
    'fofgrp':[None,'---','FoF Group Identifier'],\
    'fofgrpradeg':[5,'deg','FoF Group Center RA J2000'],\
    'fofgrpdedeg':[5,'deg','FoF Group Center Dec J2000'],\
    'fofgrpcz':[1,'km/s','FoF Group Center cz (2)'],\
    'involfof':[None,'---','In-Volume Flag for FoF Groups (1)'],\
    'fofgrpn':[None,'---','Number of Galaxies in FoF Group'],\
    'fofgrpabsrmag':[2,'mag','FoF Group-Integrated r-Band Abs. Mag.'],\
    'foflogmhvir':[2,'solMass','FoF HAM Log Group Mass (337b) (3)'],\
    'foflogmh200':[2,'solMass','FoF HAM Log Group Mass (200b) (4)'],\
    'fofgrpmhi':[2,'solMass','FoF Log Group-Integrated HI Mass'],\
    'foffc':[1,'---','FoF Central Galaxy Flag (5)'],\
})
grpdf = grpdf[[k for k in grpcolumns.keys()]]
grpdf = grpdf.round(dict((k,grpcolumns[k][0]) for k in grpcolumns.keys() if grpcolumns[k][0] is not None))
grpdf.to_csv("tab5_for_mrtmaker.csv",index=False,header=False)
print_for_online(grpcolumns)
render_cols_latex(grpcolumns,'grplatex.txt')

#latexdescriptions = []
#for ii,k in enumerate(grpcolumns.keys()):
#    if False:#grpcolumns[k][2].endswith(')'):
#        latexdescriptions.append(grpcolumns[k][2][0:-3])
#    else:
#        latexdescriptions.append(grpcolumns[k][2])
#grplatexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T, columns=['Column','Description'])
#if __name__=='__main__':
#    grplatexsample.to_latex("grplatex.txt",index=False)


############################################################
############################################################
# Duplicate Table
#dupdf = pd.read_csv(dr3path)
#dupcolumns = dict({
#    'dr2name':[None,'---','ECO DR2 Identifier (1)'],\
#    'match':[None,'---','ECO DR3 Identifier (2)'],\
#    'vif':[None,'---','Duplicate Classification (3)'],\
#})
#if __name__=='__main__':
#    dupdf.to_latex('duplatex.txt',index=False)
