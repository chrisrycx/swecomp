'''
SWE Model V0.3
A hypsometric regression to predict SWE at grid cell locations

V0.3 - Adding section to fill NA via interpolation
V0.2 - Re-organize a bit for a full model run
V0.1 - Try using multiprocessing
V0 - Just get the model running for select sites
Utilizes 50km and 300km snotel search
'''
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import linear_model
import time

#Run parameters
runtype = 'train'  #'train' or 'submit'
outfile = 'Data/ModelOutput/swemodel_v0p3_training_A.csv'
fillna = True #True or False
#Specify cellids - leave empty to run all cells
cellids = []

#Load snotel and cell data
snotel_meta = pd.read_csv('./Data/ground_measures_metadata.csv',index_col=0)
cell_elevations = pd.read_csv('./Data/Cell_elevation.csv',index_col='cell_id',usecols=['cell_id','_mean'])
cell_snotel_50 = pd.read_csv('./Data/cells_snotels_50km.csv',index_col='cell_id')
cell_snotel_300 = pd.read_csv('./Data/cells_snotels_300km.csv',index_col='cell_id')

#Load data needed for either training or submission
if runtype == 'train':
    snotel_data = pd.read_csv('./Data/ground_measures_train_features.csv',index_col=0)
    cell_data = pd.read_csv('./Data/train_labels.csv',index_col='cell_id')
elif runtype == 'submit':
    #Using snotel data with fix to CDEC:TMR station
    snotel_data = pd.read_csv('./Data/ground_measures_test_features_fix.csv',index_col=0)
    cell_data = pd.read_csv('./Data/submission_format.csv',index_col='cell_id')
else:
    print('Bad run type')
    exit(0)

# Elevation regression and prediction function
def swecalc(cellid,datestr,snotels):
    #Extract snotel elevations and data
    try:
        snotel_data_date = pd.merge(snotel_data.loc[snotels,datestr],snotel_meta.loc[snotels,'elevation_m'],left_index=True, right_index=True)
    except KeyError as ke:
        return np.nan

    #Drop snotels with no data
    snotel_data_date = snotel_data_date.dropna()

    #If less than two snotels exit
    if len(snotel_data_date) < 2:
        return np.nan

    #Export data for model fitting
    swe = snotel_data_date[datestr].to_numpy()
    elevation = snotel_data_date['elevation_m'].to_numpy()
    elevation = elevation[:, np.newaxis] #Rearrange into a vertical array

    #Extract cell elevation
    cell_elev = cell_elevations._mean[cellid]

    #Create linear regression
    regr = linear_model.LinearRegression()
    regr.fit(elevation, swe)

    #Predicted value for site
    cell_predict = regr.predict([[cell_elev]])

    #Set negative values to zero
    if cell_predict[0] < 0:
        cell_prediction = 0
    else:
        cell_prediction = cell_predict[0]

    return cell_prediction
    

if __name__=='__main__':
    
    #Cellids for run
    if not cellids:
        cellids = cell_data.index.tolist()

    #Extract observation dates
    observation_dates = cell_data.columns
    observation_dates

    #Create output model dataframe
    cell_model = cell_data.loc[cellids]
    cell_model.loc[:] = np.nan

    #Create an iterable of function inputs for multiprocessing
    swecalc_inputs = []
    for gridcell in cellids:
        
        #Get snotel IDs
        snotel_ids = cell_snotel_50.loc[[gridcell],'index_right'].to_list()

        if len(snotel_ids) < 3:
            snotel_ids = cell_snotel_300.loc[[gridcell],'index_right'].to_list()

        for obsdate in observation_dates:
            swecalc_inputs.append((gridcell,obsdate,snotel_ids))


    #Calculate with multiprocessing pool
    print("Starting multiprocessing")
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        predictions = pool.starmap(swecalc,swecalc_inputs)

    duration = time.time() - start_time
    print(f"Duration {duration} seconds")

    #Build output dataframe
    counter = 0
    for calc_input in swecalc_inputs:
        cell_model.loc[calc_input[0],calc_input[1]] = predictions[counter]
        counter = counter + 1

    if fillna:
        cell_model = cell_model.interpolate(axis=1).fillna(method='bfill',axis=1)
    
    print(cell_model.head())

    #Save to CSV
    cell_model.to_csv(outfile)




