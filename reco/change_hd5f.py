import numpy as np
import pandas as pd

df = pd.read_hdf("/scratch/bernardos/LST1/Events/proton_events.hdf5",key="proton_events")
print(df.keys())
features = ['obs_id',
            'event_id',
            'mc_energy',
            'mc_alt',
            'mc_az',
            'mc_core_x',
            'mc_core_y',
            'mc_h_first_int',
            'mc_type',
            'gps_time',
            'width',
            'length',
            'wl',
            'phi',
            'psi',
            'r',
            'x',
            'y',
            'intensity',
            'skewness',
            'kurtosis',
            'mc_alt_tel',
            'mc_az_tel',
            'impact',
            'mc_x_max',
            'time_gradient',
            'intercept',
            'src_x',
            'src_y',
            'disp',
            'hadroness',]
newdf = pd.DataFrame(columns=features)
print(newdf.keys())
newdf['obs_id'] = df['ObsID']
newdf['event_id'] = df['EvID']
newdf['mc_energy'] = df['mcEnergy']
newdf['mc_alt'] = df['mcAlt']
newdf['mc_az'] = df['mcAz']
newdf['mc_core_x'] = df['mcCore_x']
newdf['mc_core_y'] = df['mcCore_y']
newdf['mc_h_first_int'] = df['mcHfirst']
newdf['mc_type'] = df['mcType']
newdf['gps_time'] = df['GPStime']
newdf['width'] = df['width']
newdf['length'] = df['length']
newdf['wl'] = df['w/l']
newdf['phi'] = df['phi']
newdf['psi'] = df['psi']
newdf['r'] = df['r']
newdf['x'] = df['x']
newdf['y'] = df['y']
newdf['intensity'] = df['intensity']
newdf['skewness'] = df['skewness']
newdf['kurtosis'] = df['kurtosis']
newdf['mc_alt_tel'] = df['mcAlttel']
newdf['mc_az_tel'] = df['mcAztel']
newdf['impact'] = df['impact']
newdf['mc_x_max'] = df['mcXmax']
newdf['time_gradient'] = df['time_gradient']
newdf['intercept'] = df['intercept']
newdf['src_x'] = df['SrcX']
newdf['src_y'] = df['SrcY']
newdf['disp'] = df['disp']
newdf['hadroness'] = df['hadroness']

newdf.to_hdf("proton_events_point.h5",key="proton_events")
