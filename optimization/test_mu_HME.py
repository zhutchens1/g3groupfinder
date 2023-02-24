import numpy as np
import pandas as pd
from robustats import weighted_median
import matplotlib.pyplot as plt

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    #cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    cdf = (np.cumsum(weights)) / np.sum(weights)
    return np.interp(perc, cdf, data)

df = pd.read_csv("../halobiasgroupcats/fiducial/ECO_cat_0_Planck_memb_cat.csv")
df = df[df.g3grp_l>0]

Ei = np.abs(df.g3logmh_l - df.loghalom)
wi = 1/df.g3grpn_l

# simple median
print("simple median: ", np.median(Ei))

# weighted median - as used in draft but improperly explained
print("weighted median - `weighted_percentile` function: ", weighted_percentile(Ei,wi,0.5))
print('weighted median - robustats: ',weighted_median(Ei,wi))

diff=[]
xx=[]
for ii in range(5,len(Ei)):
    diff.append(weighted_median(Ei[0:ii],wi[0:ii])-weighted_percentile(Ei[0:ii],wi[0:ii],0.5))
    xx.append(len(Ei[0:ii]))

plt.figure()
plt.plot(xx,diff,'k')
plt.xlabel("# galaxies included in calculation")
plt.ylabel("robustats weighted_median - original weighted_percentile function")
plt.xscale('log')
#plt.yscale('log')
plt.show()

# group-averaged as Sheila suggested
df.loc[:,'Ei']=Ei
tmp = df.groupby('g3grp_l').mean()
print("group-averaged median: ", np.median(tmp.Ei))



x = np.array([1.1, 5.3, 3.7, 2.1, 7.0, 9.9])
weights = np.array([1.1, 0.4, 2.1, 3.5, 1.2, 0.8])
print(weighted_percentile(x,weights,0.5))

print(weighted_median(x,weights))
