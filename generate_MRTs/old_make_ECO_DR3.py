import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_ra_dec_in_resolve(ra,dec):
    return ((ra>131.25) and (ra<236.25) and (dec < 5) and (dec > 0))

def is_ra_dec_out_resolve(ra,dec):
    return ((ra<=131.25) or (ra>=236.25) or (dec >= 5) or (dec <= 0))

def skycutoffflag(ravals,decvals):
    return is_ra_dec_in_resolve(ravals,decvals).any() and is_ra_dec_out_resolve(ravals,decvals).any()

dr2 = pd.read_csv("ECODR2_050922.csv")
liv = pd.read_csv("/srv/one/zhutchen/g3groupfinder/resolve_and_eco/ECOdata_G3catalog_luminosity.csv")
dr2 = dr2.sort_values(by='name').reset_index()
liv = liv.sort_values(by='name').reset_index()

dr2 = dr2[dr2.name!='ECO13860']
liv = liv[liv.name!='ECO13860']
# get czupdate flag and add to dr2 dataframe
czupdateflag = ((dr2.cz-liv.cz).abs()>35).astype(int)
print(max((dr2.cz-liv.cz).abs()))
print(np.sum(czupdateflag))
dr2.loc[:,'czupdateflag']=czupdateflag
"""
# get mhi type flag and add to liv
enoughzeros = np.zeros(len(liv))
logmgas_a100, flag_a100 = compute_logmgas_sheila(liv.mhidet_a100, liv.mhilim_a100, liv.confused_a100, enoughzeros, enoughzeros, enoughzeros, liv.modelu_j, liv.b_a, liv.logmstar)
badsel = np.where(np.array(liv.logmstar)==0.)
logmgas_a100[badsel]=0.
flag_a100[badsel] = 0.
liv['logmgastype_a100'] = flag_a100

plt.figure()
plt.plot(liv.logmgas_a100, logmgas_a100-np.array(liv.logmgas_a100), 'k.')
plt.title("a100: (recalc-liv) vs. liv")
plt.ylim(-1,1)
plt.show()

logmgas, flag = compute_logmgas_sheila(liv.mhidet, liv.mhilim, liv.confused, liv.mhi_corr, liv.emhi_corr_sys, liv.emhi_corr_rand, liv.modelu_j, liv.b_a, liv.logmstar)
logmgas[badsel]=0.0
liv['logmgastype'] = flag

plt.figure()
plt.plot(liv.logmgas, logmgas-np.array(liv.logmgas), 'k.')
plt.title("primary: (recalc-liv) vs. liv")
plt.ylim(-1,1)
plt.show()
"""

# extract necessary columns from living and add them to dr2 file.
dr2.loc[:,'cz-Eckert16'] = dr2.loc[:,'cz'] # calling `Eckert` here so doesn't get wiped out in drop function below, will edit back to e16
dr2.loc[:,'absrmag-Eckert16'] = dr2.loc[:,'absrmag']
dr2 = dr2.drop('cz',axis=1)
dr2 = dr2.drop('absrmag',axis=1)
dr2 = dr2.drop('logmstar',axis=1)
dr2 = dr2.set_index('name').sort_index()
liv = liv.set_index('name').sort_index()

#'hirms-a100', 'peaksnhi-a100'
dr3columns = ['cz', 'absrmag','dup','resname','logmstar',\
              'mhidet', 'emhidet', 'confused', 'mhilim','limflag', 'limsigma', 'limmult', 'hirms', 'hitelescope', 'peaksnhi', 'logmgas', 'logmgastype','hirms_a100','peaksnhi_a100',\
              'mhidet_a100', 'emhidet_a100', 'confused_a100', 'mhilim_a100', 'limflag_a100', 'logmgas_a100', 'agcnr_a100', 'logmgastype_a100',\
            'g3grp_l','g3grpradeg_l','g3grpdedeg_l','g3grpcz_l','g3grpngi_l','g3grpndw_l','g3grpabsrmag_l','g3grpgiantabsrmag_l','g3logmh_l','g3logmh200_l',\
            'g3grpmhi_l','ccb_remapped','g3skycutoffflag_l']
dr3columns += ['umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'nuvmag', 'e_nuvmag', '2jmag', 'e_2jmag', '2hmag', 'e_2hmag', '2kmag', 'e_2kmag', 'extu', 'extg', 'extr', 'exti', 'extz', 'extnuv', 'extj', 'exth', 'extk']

liv = liv[dr3columns]
liv.rename(columns={"g3grp_l":"g3grp"},inplace=True)
liv.rename(columns={"g3grpradeg_l":"g3grpradeg"},inplace=True)
liv.rename(columns={"g3grpdedeg_l":"g3grpdedeg"},inplace=True)
liv.rename(columns={"g3grpcz_l":"g3grpcz"},inplace=True)
liv.rename(columns={"g3grpngi_l":"g3grpngi"},inplace=True)
liv.rename(columns={"g3grpndw_l":"g3grpndw"},inplace=True)
liv.rename(columns={"g3grpabsrmag_l":"g3grpabsrmag"},inplace=True)
liv.rename(columns={"g3grpgiantabsrmag_l":"g3grpgiantabsrmag"},inplace=True)
liv.rename(columns={"g3logmh_l":'g3logmhvir'},inplace=True)
liv.rename(columns={"g3logmh200_l":'g3logmh200'},inplace=True)
liv.rename(columns={'g3grpmhi_l':'g3grpmhi'},inplace=True)
liv.rename(columns={'ccb_remapped':'ccb-remapped'},inplace=True)
liv.rename(columns={'g3skycutoffflag_l':'g3skycutoffflag'},inplace=True)
liv.loc[:,'involg3'] = np.logical_and(liv.g3grpcz>3000,liv.g3grpcz<7000).astype(int)

# drop group info from dr2, add in revised fof groups (but need to have vol flag first)
xx = np.copy(dr2.grpcz_e17.to_numpy())
dr2=dr2.drop(labels=[lab for lab in dr2.keys() if ((('e16' in lab)|('e17' in lab))&(lab!='hitype_e16'))],axis=1)
dr2.loc[:,'invole17'] = np.logical_and(xx>3000, xx<7000).astype(int) # if grpcz_e17 in survey volume 

dr2.rename(columns={'cz-Eckert16':'cz-e16'},inplace=True)
dr2.rename(columns={'absrmag-Eckert16':'absrmag-e16'},inplace=True)

grps = pd.read_csv("revised_fof_groups.csv").set_index('name').sort_index()
grps.loc[:,'involfof'] = np.logical_and(grps['fofgrpcz'].to_numpy()>3000, grps['fofgrpcz']<7000).astype(int)

# get kcorr info
from scipy.io import readsav
kcorrdf = readsav('/srv/one/keckert/currentcat/eco_wresa_032918.dat')
#print(kcorrdf['econames'])
#ktnames = np.array([x.decode('utf-8') for x in kcorrdf['econames']])
ktnames = np.array([x.decode('utf-8') for x in kcorrdf['econames']]).byteswap().newbyteorder()
ktkcorr = np.array(kcorrdf['meankcorrr']).byteswap().newbyteorder()
kcorrdf = pd.DataFrame({'name':ktnames,'rkcorr':ktkcorr}).set_index('name').sort_index()
kcorrdf = kcorrdf[kcorrdf.index!='ECO13860']




# write out

dr3 = pd.concat([liv,dr2,grps,kcorrdf],axis=1)
dr3 = dr3.drop('index',axis=1)
dr3 = dr3[dr3.dup < 1]
dr3.rename(columns={'dup':'vif'},inplace=True)
dr3.to_csv("ECODR3_061223.csv",index=True)

if True:
    fig,axs=plt.subplots(ncols=2, figsize=(9,4))
    mybary = np.log10(10**dr3.logmstar + 10**dr3.logmgas)
    axs[0].scatter(mybary[dr3.logmgastype==0], dr3.logmbary[dr3.logmgastype==0], alpha=0.7, s=2, color='lightgreen', label='HI measurement')
    axs[1].scatter(mybary[dr3.logmgastype==1], dr3.logmbary[dr3.logmgastype==1], alpha=0.7, s=2, color='darkorange', label='PGF')
    tx=np.linspace(7,12,20)
    axs[0].plot(tx,tx,color='k', label='1:1 Line')
    axs[0].plot(tx,tx+0.25,color='k',linestyle='-.', label='+/- 0.25 dex')
    axs[0].plot(tx,tx-0.25,color='k',linestyle='-.')
    axs[1].plot(tx,tx,color='k', label='1:1 Line')
    axs[1].plot(tx,tx+0.25,color='k',linestyle='-.', label='+/- 0.25 dex')
    axs[1].plot(tx,tx-0.25,color='k',linestyle='-.')
    axs[0].set_xlim(8.5,11.5)
    axs[0].set_ylim(8.5,11.5)
    axs[1].set_xlim(8.5,11.5)
    axs[1].set_ylim(8.5,11.5)
    axs[0].set_xlabel("log10(10^logmstar + 10^logmgas)")
    axs[1].set_xlabel("log10(10^logmstar + 10^logmgas)")
    axs[0].set_ylabel("logmbary (ECO DR2)")
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='best')
    plt.show()
    
    diff = np.abs(dr3.logmbary - mybary)
    print(np.percentile(diff, [50,90,95,99]))
