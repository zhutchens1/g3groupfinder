import pandas as pd
import matplotlib.pyplot as plt

eco = pd.read_csv("ECOdata_G3catalog_luminosity.csv")
res = pd.read_csv("RESOLVEdata_G3catalog_luminosity.csv")

plt.figure()
plt.plot(eco.g3grpabsrmag_l,eco.g3logmh_l,'k.')
plt.plot(res.g3grpabsrmag_l,res.g3logmh_l,'r.',markersize=2)
plt.gca().invert_xaxis()
plt.show()


plt.figure()
plt.plot(eco.g3grpabsrmag_l,eco.g3logmh200_l,'k.')
plt.plot(res.g3grpabsrmag_l,res.g3logmh200_l,'r.',markersize=2)
plt.gca().invert_xaxis()
plt.show()
