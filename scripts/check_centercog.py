import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dl1file="/fefs/aswg/data/real/DL1/20200213/v0.4.3_v00/dl1_LST-1.1.Run01971.0023.fits.h5"

df = pd.read_hdf(dl1file, key="dl1/event/telescope/parameters/LST_LSTCam")
plt.hist(df.x, bins=100)
plt.show()
