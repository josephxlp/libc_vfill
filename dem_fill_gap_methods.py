import numpy as np
import rasterio
from rasterio.fill import fillnodata
from scipy.interpolate import griddata, Rbf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import cv2

# Utility Functions

def load_raster(fpath):
    with rasterio.open(fpath) as src:
        data = src.read(1, masked=True)
        meta = src.meta.copy()
    return data, meta

def write_raster(output_path, data, meta):
    meta.update(dtype=np.float32, nodata=np.nan)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)

# 1. Neighbourhood-Based Methods

def nearest_neighbour(data):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    grid = np.array(np.nonzero(~mask)).T
    filled = griddata(coords, values, grid, method='nearest')
    data[~mask] = filled
    return data

def bilinear_interpolation(data):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    grid = np.array(np.nonzero(~mask)).T
    filled = griddata(coords, values, grid, method='linear')
    data[~mask] = filled
    return data

def idw(data, power=2):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    grid = np.array(np.nonzero(~mask)).T
    distances = np.sqrt(np.sum((coords[:, None] - grid) ** 2, axis=2))
    weights = 1 / (distances ** power)
    weights /= weights.sum(axis=0)
    interpolated = np.dot(weights.T, values)
    data[~mask] = interpolated
    return data

# 2. Surface Fitting Methods

def polynomial_fitting(data, degree=2):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    X = np.c_[coords[:, 0] ** degree, coords[:, 1] ** degree]
    coef = np.linalg.lstsq(X, values, rcond=None)[0]
    missing_coords = np.array(np.nonzero(~mask)).T
    missing_X = np.c_[missing_coords[:, 0] ** degree, missing_coords[:, 1] ** degree]
    data[~mask] = missing_X @ coef
    return data

def spline_interpolation(data):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    grid = np.array(np.nonzero(~mask)).T
    rbfi = Rbf(coords[:, 0], coords[:, 1], values, function='thin_plate')
    filled = rbfi(grid[:, 0], grid[:, 1])
    data[~mask] = filled
    return data

# 3. Geostatistical Methods

def kriging_interpolation(data):
    # Placeholder for kriging (external libraries like PyKrige or GSTools can be used)
    raise NotImplementedError("Kriging interpolation requires external libraries.")

# 4. Morphometric-Based Methods

def gradient_constrained_interpolation(data):
    mask = np.isfinite(data)
    filled = cv2.inpaint(data.astype(np.float32), (~mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return filled

# 5. Machine Learning Methods

def random_forest_interpolation(data):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    missing_coords = np.array(np.nonzero(~mask)).T

    model = RandomForestRegressor()
    model.fit(coords, values)
    predicted = model.predict(missing_coords)
    data[~mask] = predicted
    return data

def neural_network_interpolation(data):
    mask = np.isfinite(data)
    coords = np.array(np.nonzero(mask)).T
    values = data[mask]
    missing_coords = np.array(np.nonzero(~mask)).T

    scaler = StandardScaler()
    coords = scaler.fit_transform(coords)
    missing_coords = scaler.transform(missing_coords)

    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(coords, values)
    predicted = model.predict(missing_coords)
    data[~mask] = predicted
    return data

# 6. Multi-Source Fusion

def multi_source_fusion(data, auxiliary_data):
    mask = np.isfinite(data)
    auxiliary_data[~mask] = data[~mask]
    return auxiliary_data

# 7. Filtering and Morphological Methods

def median_filter(data, kernel_size=3):
    filled = cv2.medianBlur(data.astype(np.float32), kernel_size)
    return filled

# 8. Hybrid Approaches

def hybrid_interpolation(data):
    data = nearest_neighbour(data)
    data = spline_interpolation(data)
    return data

# Example Usage
def main():
    input_path = "input.tif"
    output_path = "output.tif"

    # Load data
    data, meta = load_raster(input_path)

    # Choose method
    filled_data = spline_interpolation(data)

    # Write output
    write_raster(output_path, filled_data, meta)

if __name__ == "__main__":
    main()
