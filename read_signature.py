#%%
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import plotly.express as px

EMPTY_LINES = 14

starting_folder = "Signature"

spectral_data = {}

# For pandas dataframe
spectral_data_app = {}
index = 0

# Read all folder inside the starting folder
for date_acquisition in os.listdir(starting_folder):
    # Check if the folder is a directory
    if os.path.isdir(os.path.join(starting_folder, date_acquisition)):
        # Read all acquisition folder inside the data acquisition folder
        for n_acquisition in os.listdir(os.path.join(starting_folder, date_acquisition)):
            # Check if folder is a directory
            if os.path.isdir(os.path.join(starting_folder, date_acquisition, n_acquisition)):
                # find the first match of one or more digits (\d+) in the input string
                number = int(re.findall(r'\d+', n_acquisition)[0])
                # Read all files inside the acquisition folder
                for file in os.listdir(os.path.join(starting_folder, date_acquisition, n_acquisition)):
                    # Check if file is a .txt file
                    if file.endswith(".txt"):
                        # print(os.path.join(starting_folder, date_acquisition, n_acquisition, file))
                        # Read the position of the lattuce and the acquisition number
                        match = re.search(r'([A-Za-z])_?(\d+).*Relative_*([0-9]*)', file)
                        position = match.group(1) + match.group(2)
                        n = int(match.group(3))
                        # Check if the position is already in the dictionary
                        if position not in spectral_data:
                            spectral_data[position] = {}
                        if date_acquisition not in spectral_data[position]:
                            spectral_data[position][date_acquisition] = {}
                        spectral_data[position][date_acquisition][n] = {}
                        # Read the file
                        with open(os.path.join(starting_folder, date_acquisition, n_acquisition, file), "r") as f:
                            # avoid the first 14 lines of the file
                            for _ in range(EMPTY_LINES):
                                f.readline()
                            # start reading the file
                            for line in f:
                                # from line read only the reflectance value
                                s = line.split()
                                wavelength = float(s[0].replace(",","."))
                                reflectance = float(s[1].replace(",","."))
                                spectral_data[position][date_acquisition][n][wavelength] = reflectance
                                
                                spectral_data_app[index] = {"Date": date_acquisition, "Acquisition": number, "N": n, "Position": position, "Wavelength": wavelength, "Reflectance": reflectance}
                                index += 1


# Read the wavelength, which will be the x axis                      
x = []
with open(os.path.join(starting_folder, date_acquisition, n_acquisition, file), "r") as f:
    # avoid the first 14 lines of the file
    for _ in range(EMPTY_LINES):
        f.readline()
    # start reading the file
    for line in f:
        # from line read only the reflectance value
        wavelength = float(line.split()[0].replace(",","."))
        x.append(wavelength)

df = pd.DataFrame.from_dict(spectral_data_app, "index")
df.to_csv('spectral_data.csv',index=False)
# %%

# %%

# Check if all position are present in all Date and Acquisition

# unique_positions = set(df['Position'].unique())
# tuples = [tuple(x) for x in df[['Date', 'Acquisition']].values]
# unique_date_acquisition = set(tuples)

# flag = True
# for position in unique_positions:
#     for date_acquisition in unique_date_acquisition:
#         if not df[(df['Date'] == date_acquisition[0]) & (df['Acquisition'] == date_acquisition[1]) & (df['Position'] == position)].empty:
#             pass
#         else:
#             print(f"Position {position} is not present in Date {date_acquisition[0]} and Acquisition {date_acquisition[1]}")
#             flag = False
# if flag:
#     print("All positions have been taken in all Date and Acquisition")
    
# %%

# Normalize all the values of the same plant on the same day amoung 0 and 100

# df = df[(df['Wavelength'] < 950) & (df['Wavelength'] > 450)].reset_index(drop=True)
# grouped = df.groupby(['Date', 'Acquisition', 'Position', 'Wavelength'])

# # Apply a lambda function to normalize 'Reflectance' values between 0 and 100
# normalized_df = grouped[['Reflectance']].apply(lambda x: ((x - x.min()) / (x.max() - x.min())) * 100)

# # Reset index so 'Date', 'Acquisition', 'Position' become columns again
# normalized_df = normalized_df.reset_index(col_level="index")

# # Merge the normalized reflectance values back into the original DataFrame
# merged_df = pd.merge(df, normalized_df[['Reflectance', 'level_4']], left_index=True, right_on='level_4', suffixes=('', '_norm'))
# df = merged_df.drop(columns=['level_4'])

# %%

# Start doing the average of same position in the same day and acquisition 
# and the same wavelength, then plot it

# grouped = df.groupby(['Position', 'Date', 'Acquisition', 'Wavelength'])
# averages = grouped[['Reflectance', 'Reflectance_norm']].mean().reset_index()

# plot_folder_single = 'single_mean_folder'
# plot_folder_all = 'all_mean_folder'
# plot_folder_all_acquisition = 'all_mean_folder_acquisition'
# plot_folder_all_data = 'all_mean_folder_data'

# for type in ['', '_norm']:
# for type in ['_norm']:
    
#     name = ""
#     plt.figure(figsize=(10,6))
#     for position, group in df.groupby(['Position', 'Date', 'Acquisition', 'N']):
#         # Assign first name
#         if name == "":
#             name = str(position[0]) + '_' + str(position[1]) + '_' + str(position[2])
        
#         # Save the file with all he signatures
#         if name != str(position[0]) + '_' + str(position[1]) + '_' + str(position[2]):
#             plt.xlabel('Wavelength')
#             plt.ylabel('Average Reflectance')
#             plt.legend()
#             plt.ylim(0, 100)
#             plt.savefig(plot_folder_all_acquisition + type + '/' + name)
#             plt.close()
#             name = str(position[0]) + '_' + str(position[1]) + '_' + str(position[2])
#             plt.figure(figsize=(10,6))
            
#         # Plot the signature
#         plt.plot(group['Wavelength'], group['Reflectance'+type], label=str(position))
    
#     # Plot all date and the acquisition of all positions
#     for position, group in averages.groupby(['Date', 'Position', 'Acquisition']):
#         plt.figure(figsize=(10,6))
#         plt.plot(group['Wavelength'], group['Reflectance'+type], label=str(position))
#         plt.xlabel('Wavelength')
#         plt.ylabel('Average Reflectance')
#         plt.legend()
#         # plt.show()
#         plt.ylim(0, 100)
#         name = str(position[0]) + '_' + str(position[1]) + '_' + str(position[2])
#         plt.savefig(plot_folder_single + type + '/' + name)
#         plt.close()

#     # Plot all date and the acquisition of the same position together
#     name = ""
#     plt.figure(figsize=(10,6))
#     for position, group in averages.groupby(['Position', 'Date', 'Acquisition']):
#         # Assign first name
#         if name == "":
#             name = str(position[0])
        
#         # Save the file with all he signatures
#         if name != str(position[0]):
#             plt.xlabel('Wavelength')
#             plt.ylabel('Average Reflectance')
#             plt.legend()
#             plt.ylim(0, 100)
#             plt.savefig(plot_folder_all + type + '/' + name)
#             plt.close()
#             name = str(position[0])
#             plt.figure(figsize=(10,6))
            
#         # Plot the signature
#         plt.plot(group['Wavelength'], group['Reflectance'+type], label=str(position))
        
#     # Plot all positions of the same date and acquisition together
#     name = ""
#     plt.figure(figsize=(10,6))
#     for position, group in averages.groupby(['Date', 'Acquisition', 'Position']):
#         # Assign first name
#         if name == "":
#             name = str(position[0]) + '_' + str(position[1])
        
#         # Save the file with all he signatures
#         if name != str(position[0]) + '_' + str(position[1]):
#             plt.xlabel('Wavelength')
#             plt.ylabel('Average Reflectance')
#             plt.legend()
#             plt.ylim(0, 100)
#             plt.savefig(plot_folder_all_data + type + '/' + name)
#             plt.close()
#             name = str(position[0]) + '_' + str(position[1])
#             plt.figure(figsize=(10,6))
            
#         # Plot the signature
#         plt.plot(group['Wavelength'], group['Reflectance'+type], label=str(position))
    