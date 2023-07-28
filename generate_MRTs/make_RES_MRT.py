import numpy as np
import pandas as pd

file_to_read = '../resolve_and_eco/RESOLVEdata_G3catalog_luminosity.csv'

resolvedr4 = pd.read_csv(file_to_read).sort_values(by='name')
for kk in resolvedr4.keys():
    if kk.endswith('_l'): #g3grp_l type columns
        resolvedr4.loc[:,kk[:-2]]=resolvedr4[kk].to_numpy()
resolvedr4.loc[:,'g3logmhvir']=resolvedr4.g3logmh_l.to_numpy()
resolvedr4.loc[:,'involg3'] = np.logical_and(resolvedr4.g3grpcz.to_numpy()>4500,resolvedr4.g3grpcz.to_numpy()<7000).astype(int)
resolvedr4.loc[:,'involfof'] = np.logical_and(resolvedr4.fofgrpcz.to_numpy()>4500,resolvedr4.fofgrpcz.to_numpy()<7000).astype(int)
resolvedr4.loc[:,'axialratio'] = resolvedr4.b_a
resolvedr4.loc[:,'rkcorr']=resolvedr4.kcorrr
resolvedr4.loc[:,'mudelta']=resolvedr4.mu_delta
resolvedr4.loc[:,'czhel']=resolvedr4.vhel
resolvedr4.rename(columns={xx:xx.replace('_','') for xx in resolvedr4.keys() if ('model' in xx) and ('_' in xx)},inplace=True)
resolvedr4.rename(columns={xx:xx.replace('-','') for xx in resolvedr4.keys() if '-' in xx},inplace=True)
resolvedr4.rename(columns={'emhidet':'e_mhidet'},inplace=True)
resolvedr4.to_csv(".tmp_resolvedr4_file.csv")

keys_to_publish=['name','radeg','dedeg','cz','czhel','loscmvgdist',\
        'nuvmag','e_nuvmag','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag',\
        '2jmag','e_2jmag','2hmag','e_2hmag','2kmag','e_2kmag','uymag','e_uymag','uhmag','e_uhmag','ukmag','e_ukmag',\
        'extnuv','extu','extg','extr','exti','extz','exty','extj','exth','extk','logmstar','absrmag',\
        'modelabsrmag','modelnuvr','modelur','modelui','modeluj','modeluk','modelgr','modelgi','modelgj','modelgk','modelurcorr',\
        'r50','r90','axialratio','mudelta',\
        'mhidet','e_mhidet','mhilim','logmgas','logmgastype',\
        'g3grp','g3grpradeg','g3grpdedeg','g3grpcz','g3grpngi','g3grpndw','g3grpabsrmag','g3logmhvir','g3logmh200','g3grpmhi','g3fc','g3skycutoffflag','involg3',\
        'fofgrp','fofgrpradeg','fofgrpdedeg','fofgrpcz','fofgrpn','fofgrpabsrmag','foflogmhvir','foflogmh200','fofgrpmhi','foffc','fofskycutoffflag','involfof']

resolvedr4 = resolvedr4[keys_to_publish]

factor=1e7#2.36e5 * (3000/70.)**2. * 0.01 # S16 precision was 0.01 Jy km/s
resolvedr4.loc[:,'mhidet'] = factor*(resolvedr4.mhidet/factor).round(2)
resolvedr4.loc[:,'e_mhidet'] = factor*(resolvedr4.e_mhidet/factor).round(2)
resolvedr4.loc[:,'mhilim'] = factor*(resolvedr4.mhilim/factor).round(2)

#resolvefof = pd.read_csv("RESOLVEliving_061223_updatedfofgroups.csv").sort_values(by='name')
#resolvefof = resolvefof[['fofgrp','fofgrpradeg','fofgrpdedeg','fofgrpcz','fofgrpn','fofgrpabsrmag','foflogmhvir','foflogmh200','fofgrpmhi','fofskycutoffflag']]
#resolvefof.loc[:,'involfof'] = np.logical_and(resolvefof.fofgrpcz.to_numpy()>4500,resolvefof.fofgrpcz.to_numpy()<7000).astype(int)

#resolvedr4 = pd.concat([resolveg3,resolvefof],axis=1)
resolvedr4.fillna(-999.,inplace=True)
print(resolvedr4)

############################## 
# now make the MRT
import sys
from make_ECO_MRTs import photcolumns, grpcolumns, hicolumns, print_for_online, render_cols_latex

resolvedr4.replace(-99,-999,inplace=True)
ecodr3masterdict = {**photcolumns, **grpcolumns, **hicolumns}

rescolumns=dict()
for kk in resolvedr4.keys():
    if kk=='g3skycutoffflag':
        rescolumns[kk]=[None,'---','RESOLVE-A Sky Cutoff Flag for G3 Groups']
    elif kk=='fofskycutoffflag':
        rescolumns[kk]=[None,'---','RESOLVE-A Sky Cutoff Flag for FoF Groups']
    else:
        try:
            rescolumns[kk]=ecodr3masterdict[kk]
        except:
            print("No ECO description for "+kk)
assert len(rescolumns)==len(resolvedr4.keys())
rescolumns['name']=[None,'---','RESOLVE Galaxy Identifier']

resolvedr4 = resolvedr4.round(dict((k,rescolumns[k][0]) for k in rescolumns.keys() if rescolumns[k][0] is not None))

print_for_online(rescolumns)
resolvedr4.to_csv("tab6_for_mrtmaker.csv",header=False,index=False)

#aptable = Table.from_pandas(resolvedr4)
#for kk in resolvedr4.keys():
#    tmp = aptable[kk]
#    tmp.unit = rescolumns[kk][1]
#    tmp.description = rescolumns[kk][2]
#tmaker.author = authorlist
#resMRT = tmaker.addTable(aptable,name='resdr4_mrt.txt',description='RESOLVE DR4')
#resMRT.notes = None
#tmaker.title=papertitle
#tmaker.toMRT()


render_cols_latex(rescolumns,'reslatex.txt')
#latexdescriptions=[]
#for ii,k in enumerate(rescolumns.keys()):
#    if False: #rescolumns[k][2].endswith(')'):
#        latexdescriptions.append(rescolumns[k][2][0:-3])
#    else:
#        latexdescriptions.append(rescolumns[k][2])

#if __name__=='__main__':
#    latexsample = pd.DataFrame(np.array([np.arange(0,len(latexdescriptions),1)+1,np.array(latexdescriptions)]).T,columns=['Column','Description'])
#    latexsample.to_latex('resdr4latex.txt',index=False)
