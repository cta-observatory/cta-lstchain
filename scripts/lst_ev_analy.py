import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import os
import sys

date=sys.argv[1]
sizecut=int(sys.argv[2])
path = "/fefs/aswg/data/real/DL1/"+date+"/v0.4.3_v00/"
x=np.array([])
y=np.array([])
howmany=0
max_intensity=80000

for r, d, f in os.walk(path):
    for infile in f:
        if '.fits.h5' in infile:

            df = pd.read_hdf(path+infile, key="dl1/event/telescope/parameters/LST_LSTCam")
            df = df[df.intensity>sizecut]
            df = df[df.intensity<sizecut]
            x = np.append(x, df.x.to_numpy())
            y = np.append(y, df.y.to_numpy())
            howmany=howmany+1

#print("%d files" % howmany)
plt.hist2d(x, y, norm = mpl.colors.LogNorm(), bins=100)
plt.title(date)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar()
plt.savefig(date+"_sizecut%d.pdf" % sizecut)

#plt.show()
