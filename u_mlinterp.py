import os 
import numpy as np
import rasterio
import torch
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from upaths import OUT_TILES_DPATH
import os

def get_best_gpu():
    """Selects the best available GPU based on memory size."""
    if not torch.cuda.is_available():
        return 'CPU'
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return 'cuda:0'
    
    best_gpu = max(range(num_gpus), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return f'cuda:{best_gpu}'

def read_dem(dem_path):
    """Reads a DEM file and returns the data as a NumPy array along with its metadata."""
    with rasterio.open(dem_path) as src:
        data = src.read(1).astype(np.float32)  # Read first band
        meta = src.meta
    return data, meta

def mask_invalid_values(data, nodata_value=-9999, min_valid=-50, max_valid=10000):
    """Masks out invalid values by setting them to NaN."""
    data = np.where((data <= min_valid) | (data >= max_valid) | (data == nodata_value), np.nan, data)
    return data

def interpolate_missing_values_smodel(data, model_type='catboost'):
    """Performs interpolation on missing values in a DEM using the specified model."""
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    missing_coords = np.array(np.nonzero(~mask)).T
    
    if len(values) == 0 or len(missing_coords) == 0:
        print("No valid data to train the model or no missing values to interpolate.")
        return data
    
    model = None
    device = get_best_gpu()
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'catboost':
        model = CatBoostRegressor(iterations=2000, 
                                  verbose=200, 
                                  use_best_model=True,
                                  early_stopping_rounds=500,
                                  task_type='GPU' if 'cuda' in device else 'CPU', 
                                  devices=[int(device.split(':')[-1])] if 'cuda' in device else None)
    elif model_type == 'lightgbm':
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
    else:
        raise ValueError("Unsupported model type. Choose from 'rf', 'catboost', 'lightgbm', or 'xgboost'.")
    
    model.fit(coords, values)
    data[~mask] = model.predict(missing_coords)
    return data

def interpolate_missing_values_emodel(data, model_type='catboost', n_seeds=5):
    """Performs interpolation on missing values in a DEM using an ensemble of models with different random seeds."""
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    missing_coords = np.array(np.nonzero(~mask)).T
    
    if len(values) == 0 or len(missing_coords) == 0:
        print("No valid data to train the model or no missing values to interpolate.")
        return data
    
    device = get_best_gpu()
    predictions = np.zeros((n_seeds, len(missing_coords)))
    
    for i, seed in enumerate(range(n_seeds)):
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=seed)
        elif model_type == 'catboost':
            model = CatBoostRegressor(iterations=2000, 
                                  verbose=500, 
                                  use_best_model=True,
                                  early_stopping_rounds=250,
                                  task_type='GPU' if 'cuda' in device else 'CPU', 
                                  devices=[int(device.split(':')[-1])] if 'cuda' in device else None)
            
            #model = CatBoostRegressor(iterations=2000, verbose=200, task_type='GPU' if 'cuda' in device else 'CPU', devices=[int(device.split(':')[-1])] if 'cuda' in device else None, random_seed=seed)
        elif model_type == 'lightgbm':
            model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=seed)
        elif model_type == 'xgboost':
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror', random_state=seed)
        else:
            raise ValueError("Unsupported model type. Choose from 'rf', 'catboost', 'lightgbm', or 'xgboost'.")
        
        model.fit(coords, values)
        predictions[i] = model.predict(missing_coords)
    
    data[~mask] = predictions.mean(axis=0)
    return data


def save_dem(dem_path, data, meta):
    """Saves the processed DEM back to a file."""
    meta.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(dem_path, 'w', **meta) as dst:
        dst.write(data, 1)


def mlinterps(dem_ipath, dem_opath, model_type='catboost'):
    """Full pipeline: Read, mask, interpolate, and save the DEM."""
    print('read_dem')
    data, meta = read_dem(dem_ipath)
    print('mask_invalid_values')
    data = mask_invalid_values(data)
    print('interpolate_missing_values')
    data = interpolate_missing_values_smodel(data, model_type)
    print('save_dem')
    save_dem(dem_opath, data, meta)
    print('mlinterps')

def mlinterpe(dem_ipath, dem_opath, model_type='catboost'):
    """Full pipeline: Read, mask, interpolate, and save the DEM."""
    print('read_dem')
    data, meta = read_dem(dem_ipath)
    print('mask_invalid_values')
    data = mask_invalid_values(data)
    print('interpolate_missing_values')
    data = interpolate_missing_values_emodel(data, model_type)
    print('save_dem')
    save_dem(dem_opath, data, meta)
    print('mlinterpe')

# improve the code to include params and hpo, and saving 