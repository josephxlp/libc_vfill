import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# 1. Define the editing area, which includes the gap and its bounding box plus a margin of N pixels.
def define_editing_area(gap_mask, margin):
    from skimage.morphology import binary_dilation
    
    editing_area = binary_dilation(gap_mask, iterations=margin)
    return editing_area

# 2. Select the best available DEM reference.
def select_reference_dem(reference_paths, target_extent):
    # Logic to pick the best DEM that intersects with target_extent
    # Here we assume the first available DEM is selected for simplicity
    return reference_paths[0]

# 3. Resample the fill surface to match the output DEM posting.
def resample_to_match(src_path, target_profile):
    with rasterio.open(src_path) as src:
        data = src.read(1, out_shape=(
            target_profile['height'],
            target_profile['width']
        ), resampling=rasterio.enums.Resampling.bilinear)
        return data

# 4. Derive the unreliable mask.
def derive_unreliable_mask(dem_data, threshold):
    unreliable_mask = dem_data < threshold
    return unreliable_mask

# 5. Compute non-rigid shifts on the reference to compensate for residual misalignments.
def compute_non_rigid_shifts(reference_dem, target_dem):
    shift_map = target_dem - reference_dem
    return shift_map

# 6. Create the delta surface.
def create_delta_surface(reference_dem, target_dem):
    delta_surface = target_dem - reference_dem
    return delta_surface

# 7. Populate the center of large voids in the delta surface with a mean value.
def fill_large_voids(delta_surface, void_mask):
    mean_value = np.mean(delta_surface[~void_mask])
    delta_surface[void_mask] = mean_value
    return delta_surface

# 8. Interpolate across the voids in the delta surface.
def interpolate_voids(delta_surface, void_mask):
    x, y = np.meshgrid(np.arange(delta_surface.shape[1]), np.arange(delta_surface.shape[0]))
    points = np.column_stack((x[~void_mask], y[~void_mask]))
    values = delta_surface[~void_mask]
    
    interpolated = griddata(points, values, (x, y), method='cubic')
    delta_surface[void_mask] = interpolated[void_mask]
    return delta_surface

# 9. Smooth the delta surface with a low-pass filter.
def smooth_delta_surface(delta_surface, sigma):
    smoothed_surface = gaussian_filter(delta_surface, sigma=sigma)
    return smoothed_surface

# 10. Combine the interpolated delta with the filling source within the original voids.
def combine_delta_with_source(original_dem, delta_surface, void_mask):
    filled_dem = original_dem.copy()
    filled_dem[void_mask] += delta_surface[void_mask]
    return filled_dem

# Example workflow
def stable_delta_interpolation_workflow(input_dem_path, reference_dems, gap_mask_path, output_dem_path, margin=10, threshold=0, sigma=3):
    with rasterio.open(input_dem_path) as src:
        original_dem = src.read(1)
        profile = src.profile
    
    with rasterio.open(gap_mask_path) as mask_src:
        gap_mask = mask_src.read(1).astype(bool)

    editing_area = define_editing_area(gap_mask, margin)
    reference_dem_path = select_reference_dem(reference_dems, profile['transform'])
    reference_dem = resample_to_match(reference_dem_path, profile)
    
    unreliable_mask = derive_unreliable_mask(original_dem, threshold)
    shift_map = compute_non_rigid_shifts(reference_dem, original_dem)
    delta_surface = create_delta_surface(reference_dem, original_dem)

    delta_surface = fill_large_voids(delta_surface, gap_mask)
    delta_surface = interpolate_voids(delta_surface, gap_mask)
    delta_surface = smooth_delta_surface(delta_surface, sigma)
    
    filled_dem = combine_delta_with_source(original_dem, delta_surface, gap_mask)

    with rasterio.open(output_dem_path, 'w', **profile) as dst:
        dst.write(filled_dem, 1)

# Parameters and paths
input_dem_path = "input_dem.tif"
gap_mask_path = "gap_mask.tif"
reference_dems = ["reference_dem1.tif", "reference_dem2.tif"]
output_dem_path = "output_dem.tif"

stable_delta_interpolation_workflow(input_dem_path, reference_dems, gap_mask_path, output_dem_path)

# modify this code so that the mask is extracted from input_dem_path if None : no paths is provided
# this should be nodata from metadata, and anyvalue > 1000 or anyvalue <-50

# reference_dems it should work even if just 1 dem in the list is provided
