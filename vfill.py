import os
from pathlib import Path
from osgeo import gdal 
import rasterio
import numpy as np 
import random
import math
import cv2

from scipy.interpolate import NearestNDInterpolator
import pyinterp.fill

from skimage.restoration import inpaint_biharmonic
import rasterio.fill
import matplotlib.pyplot as plt


def load_data_obj(fpath):
    with rasterio.open(fpath) as src:
        data = src.read(1)  
    return src,data

def write_raster(fpath, data, reference_fpath):
    """
    Writes a numpy array to a raster file using a reference file for metadata.

    Parameters:
    fpath (str): File path to save the raster file.
    data (numpy.ndarray): Data to be written to the raster file.
    reference_fpath (str): File path to the reference raster file for metadata.
    """
    with rasterio.open(reference_fpath) as src:
        meta = src.meta.copy()

    meta.update(dtype=rasterio.float32, nodata=np.nan)

    with rasterio.open(fpath, 'w', **meta) as dst:
        dst.write(data, 1)


def get_neighboring_pixel_single_band(img, x, y):
    x_rand, y_rand = 0, 0

    max_num_tries = 30
    max_tries_per_neighbourhood = 8#3
    neighbourhood_size_increment = 64#10
    current_window_size = 32#10
    total_tries = 0
    for _ in range(math.ceil(max_num_tries / max_tries_per_neighbourhood)):
        for _ in range(max_tries_per_neighbourhood):
            min_x = max(0, x - current_window_size)
            max_x = min(img.shape[0], x + current_window_size)
            min_y = max(0, y - current_window_size)
            max_y = min(img.shape[1], y + current_window_size)
            x_rand = random.randint(min_x, max_x - 1)
            y_rand = random.randint(min_y, max_y - 1)
            total_tries += 1
            if not np.isnan(img[x_rand, y_rand]):
                return x_rand, y_rand
        current_window_size += neighbourhood_size_increment

    return x_rand, y_rand

def fill_swath_with_neighboring_pixel_single_band(img):
    img_with_neighbor_filled = img.copy()
    x_swath, y_swath = np.where(np.isnan(img))

    for i in range(len(x_swath)):
        x_rand, y_rand = get_neighboring_pixel_single_band(img, x_swath[i], y_swath[i])
        img_with_neighbor_filled[x_swath[i], y_swath[i]] = img[x_rand, y_rand]
    
    return img_with_neighbor_filled

def scipy_nearest(a):
    mask = np.where(np.isfinite(a))
    interp = NearestNDInterpolator(np.transpose(mask), a[mask])
    return interp(*np.indices(a.shape))

def opencv(a, algo):
    assert algo in {cv2.INPAINT_NS, cv2.INPAINT_TELEA}
    a_min = np.nanmin(a)
    a_max = np.nanmax(a)
    a_norm = (a - a_min) / (a_max - a_min)
    a_norm = a_norm.astype(np.float32)
    
    mask = np.isnan(a).astype(np.uint8)
    a_norm[np.isnan(a_norm)] = 0

    res_norm = cv2.inpaint(a_norm, mask, inpaintRadius=10, flags=algo)
    res = res_norm * (a_max - a_min) + a_min
    return res

def pyinterp_make_grid(src,data):
    x = pyinterp.Axis(np.linspace(src.bounds.left, src.bounds.right, data.shape[1]), is_circle=False)
    y = pyinterp.Axis(np.linspace(src.bounds.bottom, src.bounds.top, data.shape[0]), is_circle=False)
    grid = pyinterp.Grid2D(x, y, data)
    return grid

def split_data_and_mask(data,src):
    darray = data.astype(np.float32).copy()
    darray[data == src.nodata] = np.nan 
    mask = np.isnan(darray) | np.isinf(darray)
    return darray, mask

def pyinterp_loess(src,data):
    #x_axis = pyinterp.Axis(x, is_circle=False)
    #y_axis = pyinterp.Axis(y, is_circle=False)
    #grid = pyinterp.Grid2D(x_axis, y_axis, a)
    grid = pyinterp_make_grid(src,data)
    res = pyinterp.fill.loess(grid, nx=55, ny=55)
    return res

def fill_rasterio(a,si=0):
    mask = np.isfinite(a)
    max_search_distance = int(math.sqrt(a.shape[0] ** 2 + a.shape[1] ** 2)) + 1
    return rasterio.fill.fillnodata(a, mask=mask, max_search_distance=max_search_distance, smoothing_iterations=si)


def fill_skimage_inpaint_biharmonic_region(src,data):
    #mask = np.isnan(a)
    darray, mask = split_data_and_mask(data,src)
    return inpaint_biharmonic(darray, mask, split_into_regions=True)

def fill_pyinterp_gauss_seidel(data,src):
    #https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.fill.gauss_seidel.html#pyinterp.fill.gauss_seidel
    darray, mask = split_data_and_mask(data,src)
    grid = pyinterp_make_grid(src,darray)
    _, converged = pyinterp.fill.gauss_seidel(
                mesh=grid,
                first_guess='zonal_average',  # or 'zero'
                max_iteration=None,  # or specify a number
                epsilon=0.0001,
                relaxation=None,  # or specify a value
                num_threads=0
                )
    return converged


def split_raster(input_raster_path: str, output_dir: str, num_parts: int) -> None:
  """
  Split a raster into specified number of equal parts and save them.

  Args:
      input_raster_path (str): Path to the input raster file
      output_dir (str): Directory where the split rasters will be saved
      num_parts (int): Number of parts to split the raster into
  """
  # Calculate optimal number of rows and columns for splitting
  # Try to keep the aspect ratio of splits similar to original
  num_rows = int(math.sqrt(num_parts))
  num_cols = int(math.ceil(num_parts / num_rows))

  # Adjust rows if needed to get exact number of parts
  while num_rows * num_cols < num_parts:
      num_rows += 1

  # Create output directory if it doesn't exist
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  # Open the input raster
  with rasterio.open(input_raster_path) as src:
      # Get metadata
      meta = src.meta
      height = src.height
      width = src.width

      # Calculate dimensions for each part
      h_split = height // num_rows
      w_split = width // num_cols

      # Update metadata for new dimensions
      meta.update({
          'height': h_split,
          'width': w_split
      })

      # Read the entire raster
      data = src.read()

      # Counter for parts
      part_count = 0

      # Split and save parts
      for i in range(num_rows):
          for j in range(num_cols):
              if part_count >= num_parts:
                  break

              # Calculate window
              window_data = data[:,
                               i * h_split:(i + 1) * h_split,
                               j * w_split:(j + 1) * w_split]

              # Create output filename
              output_filename = f"part_{i}_{j}.tif"
              output_path = os.path.join(output_dir, output_filename)

              # Update transform for the new part
              part_transform = rasterio.Affine(
                  meta['transform'].a,
                  meta['transform'].b,
                  meta['transform'].c + (j * w_split * meta['transform'].a),
                  meta['transform'].d,
                  meta['transform'].e,
                  meta['transform'].f + (i * h_split * meta['transform'].e)
              )
              meta['transform'] = part_transform

              # Write the part
              with rasterio.open(output_path, 'w', **meta) as dst:
                  dst.write(window_data)

              part_count += 1

              if part_count >= num_parts:
                  break




def read_raster(file_path: str) -> tuple[np.ndarray, dict]:
    """
    Read a raster file and replace NoData values with np.nan.
    
    Args:
        file_path (str): Path to the raster file
    
    Returns:
        tuple: (raster_data, metadata)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        ndv = src.nodata
        if ndv is not None:
            data = np.where(data == ndv, np.nan, data)
        meta = src.meta
    return data, meta

def get_percentile_bounds(data: np.ndarray, lower: float = 10, upper: float = 70) -> tuple[float, float]:
    """
    Calculate percentile bounds for non-NaN values.
    
    Args:
        data (np.ndarray): Input array with possible NaN values
        lower (float): Lower percentile bound
        upper (float): Upper percentile bound
    
    Returns:
        tuple: (vmin, vmax) percentile values
    """
    valid_data = data[~np.isnan(data)]
    vmin = np.percentile(valid_data, lower)
    vmax = np.percentile(valid_data, upper)
    return vmin, vmax

def plot_raster(ax, data: np.ndarray, title: str, vmin: float = None, vmax: float = None, 
                cmap: str = 'viridis') -> None:
    """
    Plot a single raster on a given axis with specified value bounds.
    
    Args:
        ax: Matplotlib axis
        data (np.ndarray): Raster data
        title (str): Plot title
        vmin (float): Minimum value for color scaling
        vmax (float): Maximum value for color scaling
        cmap (str): Colormap name
    """
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

def add_statistics(ax, data: np.ndarray, vmin: float = None, vmax: float = None) -> None:
    """
    Add statistics text to plot including percentile bounds.
    
    Args:
        ax: Matplotlib axis
        data (np.ndarray): Raster data
        vmin (float): Lower percentile value
        vmax (float): Upper percentile value
    """
    stats = (
        f'Min: {np.nanmin(data):.2f}\n'
        f'Max: {np.nanmax(data):.2f}\n'
        f'Mean: {np.nanmean(data):.2f}\n'
        f'10th percentile: {vmin:.2f}\n'
        f'70th percentile: {vmax:.2f}'
    )
    ax.text(-0.1, -0.2, stats, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

def plot_xy_rasters(xfile: str, yfile: str, lower_pct: float = 10, upper_pct: float = 70) -> None:
    """
    Plot X and Y rasters side by side using percentile-based scaling from X.
    
    Args:
        xfile (str): Path to X raster
        yfile (str): Path to Y raster
        lower_pct (float): Lower percentile for scaling
        upper_pct (float): Upper percentile for scaling
    """
    # Read raster data
    x_data, _ = read_raster(xfile)
    y_data, _ = read_raster(yfile)
    
    # Calculate percentile bounds from x_data
    vmin, vmax = get_percentile_bounds(x_data, lower_pct, upper_pct)
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rasters with same value bounds
    plot_raster(ax1, x_data, 'X Raster', vmin=vmin, vmax=vmax)
    plot_raster(ax2, y_data, 'Y Raster', vmin=vmin, vmax=vmax)
    
    # Add statistics
    add_statistics(ax1, x_data, vmin, vmax)
    add_statistics(ax2, y_data, vmin, vmax)
    
    plt.tight_layout()
    plt.show()

def riofiller_inpaint(fipath, fopath, siter=0):
    fopath = fopath.replace('.tif', f'_{siter}.tif')
    if not os.path.isfile(fopath):
        src,data = load_data_obj(fipath)
        data, mask = split_data_and_mask(data,src)
        data_res = fill_rasterio(data,si=siter)
    
        write_raster(fopath, data_res, fipath)
        print(f'newly created :{fopath}')
    else:
        print(f'already created :{fopath}')
