import pandas as pd
import numpy as np
from CARS_model import CARS
import os

R = 500
df = pd.read_csv('../spectral_data.csv')

wavelengths_to_select = df[(df['Wavelength'] <= 950) & (df['Wavelength'] >= 450)]['Wavelength'].unique()
date_to_select = ['24-01-29', '24-02-01', '24-02-02']
date_to_select = ['24-01-29', '24-01-30', '24-01-31', '24-02-01', '24-02-02']

n_df = df[df['Wavelength'].isin(wavelengths_to_select) & df['Date'].isin(date_to_select) & df['Acquisition'].isin([1])].reset_index(drop=True)

n_df.loc[n_df.Date == '24-02-02', 'Class'] = 'Unhealty'  
n_df.loc[n_df.Date == '24-02-01', 'Class'] = 'Unhealty'  
n_df.loc[n_df.Date == '24-01-31', 'Class'] = 'Unhealty'  
n_df.loc[n_df.Date == '24-01-30', 'Class'] = 'Healty' 
n_df.loc[n_df.Date == '24-01-29', 'Class'] = 'Healty' 

n_df['Reflectance_avg_log'] = np.log10(n_df['Reflectance'])

# Apply a lambda function to normalize 'Reflectance' values between 0 and 100, using SNV normalization
col_group = ['Class', 'Date', 'Position', 'Acquisition', 'N']

grouped = n_df.groupby(col_group)
n_df['Reflectance_SNV_norm'] = grouped['Reflectance_avg_log'].transform(lambda x: (x - x.mean()) / x.std())
# n_df['Reflectance_SNV_norm'] = grouped['Reflectance'].transform(lambda x: (x - x.mean()) / x.std())

X_df = n_df.pivot_table(index=col_group, columns='Wavelength', values='Reflectance_SNV_norm')

# count healthy and unhealthy in X_df
print(X_df.index.get_level_values('Class').value_counts())

# Extract the name of the file
file_name = os.path.basename(__file__).split('.')[0]
path = os.path.join(os.path.abspath(os.getcwd()), file_name)
c = CARS(path, col_group, X_df, MAX_COMPONENTS=4, CV_FOLD=5, calibration=True, test_percentage=0.2)
c.perform_pca()
c.cars_model(R=R, N=100, rmsecv=True, ars=True , MC_SAMPLES=0.8, start=0)
c.save_results()
