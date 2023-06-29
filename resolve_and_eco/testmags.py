import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("ECOdata_080822.csv")
#df.loc[:,'myabsrmag']=np.trunc(100*(df.rmag + 5 - 5*np.log10(df.cz/70. * 1e6) - df.extr))/100
df.loc[:,'myabsrmag']=(df.rmag + 5 - 5*np.log10(df.cz/70. * 1e6) - df.extr)#.round(2)

plt.figure()
plt.title("ECO catalog -- current.")
sel = (df.resname=='notinresolve')
plt.plot(df[sel].absrmag,df[sel].myabsrmag-df[sel].absrmag,'.',label='Not in RESOLVE-A', markersize=2, color='orange')
sel = (df.resname!='notinresolve')
plt.plot(df[sel].absrmag,df[sel].myabsrmag-df[sel].absrmag,'kx',label='In RESOLVE-A')
plt.legend(loc='best')
plt.xlabel("catalog absrmag")
plt.ylabel("re-calculated absrmag - catalog absrmag")
plt.show()
