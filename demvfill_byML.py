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

def mask_invalid_values(data, nodata_value=-9999, min_valid=-999, max_valid=10000):
    """Masks out invalid values by setting them to NaN."""
    data = np.where((data <= min_valid) | (data >= max_valid) | (data == nodata_value), np.nan, data)
    return data

def interpolate_missing_values(data, model_type='catboost'):
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
        model = CatBoostRegressor(iterations=1000, verbose=100, task_type='GPU' if 'cuda' in device else 'CPU', devices=[int(device.split(':')[-1])] if 'cuda' in device else None)
    elif model_type == 'lightgbm':
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
    else:
        raise ValueError("Unsupported model type. Choose from 'rf', 'catboost', 'lightgbm', or 'xgboost'.")
    
    model.fit(coords, values)
    data[~mask] = model.predict(missing_coords)
    return data

def save_dem(dem_path, data, meta):
    """Saves the processed DEM back to a file."""
    meta.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(dem_path, 'w', **meta) as dst:
        dst.write(data, 1)

def demvfill_byML(dem_ipath, dem_opath, model_type='catboost'):
    """Full pipeline: Read, mask, interpolate, and save the DEM."""
    print('read_dem')
    data, meta = read_dem(dem_ipath)
    print('mask_invalid_values')
    data = mask_invalid_values(data)
    print('interpolate_missing_values')
    data = interpolate_missing_values(data, model_type)
    print('save_dem')
    save_dem(dem_opath, data, meta)
    print('demvfill_byML')


X = 30
if __name__ == '__main__':

    outdir = f"{OUT_TILES_DPATH}/DEMVFILL/TILES{X}"
    tiles_xdpath = f"{OUT_TILES_DPATH}/TILES{X}"
    tilenames = os.listdir(tiles_xdpath)
    for tilename in tilenames:
        #tilename = 'N10E105'
        dem_ipath = f"{tiles_xdpath}/{tilename}/{tilename}_tdem_DEM__Fw.tif"
        tile_odpath = f"{outdir}/{tilename}/" 
        os.makedirs(tile_odpath, exist_ok=True)
        dem_opath = f"{tile_odpath}/{tilename}_tdem_DEM__iML.tif"

        print(os.path.isfile(dem_ipath))
        if not os.path.isfile(dem_opath):
            demvfill_byML(dem_ipath, dem_opath, model_type='catboost')
        else:
            print(f'file aready created\nsaved at {dem_opath}')

# instead of training just one model given model type, use at least 5 seeds to create an ensemble to train and predict