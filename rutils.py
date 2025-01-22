import os
from osgeo import gdal, gdalconst
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


# Conversion factor (degrees per meter at the equator)
conversion_factor = 9.259259259259259e-6

# Function to convert meters to degrees
def meters_to_degrees(meters):
    return conversion_factor * meters

# Function to convert degrees to meters
def degrees_to_meters(degrees):
    return degrees / conversion_factor

# Example usage
x_degrees = 0.0001111111111111111164
x_meters = degrees_to_meters(x_degrees)
print(f"x in meters: {x_meters}")

# Example for a 30m grid size
grid_size_meters = 30
grid_size_degrees = meters_to_degrees(grid_size_meters)
print(f"Grid size in degrees for {grid_size_meters}m: {grid_size_degrees}")


def read_xyz(rpath, default_band_name='Band'):
    """
    Reads raster data and extracts coordinates, bands, and metadata.

    Parameters:
        rpath (str): Path to the raster file.
        default_band_name (str): Default prefix for band names.

    Returns:
        tuple: x-coordinates, y-coordinates, bands, CRS, band names, and NoData value.
    """
    with rasterio.open(rpath) as f:
        bands = f.read()  # Read all bands
        x = np.linspace(f.bounds.left, f.bounds.right, f.width)
        y = np.linspace(f.bounds.top, f.bounds.bottom, f.height)
        crs = f.crs  # Get the CRS
        nodata = f.nodata  # Get the NoData value

        # Determine band names
        if f.descriptions and any(f.descriptions):
            band_names = [
                desc if desc else f'{default_band_name}_{i + 1}' 
                for i, desc in enumerate(f.descriptions)
            ]
        else:
            band_names = [
                default_band_name if f.count == 1 
                else f'{default_band_name}_{i + 1}' 
                for i in range(f.count)
            ]
    return x, y, bands, crs, band_names, nodata

def xyz2df(x, y, bands, band_names, nodata):
    """
    Converts raster data into a DataFrame.

    Parameters:
        x (array): x-coordinates.
        y (array): y-coordinates.
        bands (array): Raster bands.
        band_names (list): Band names.
        nodata (float): NoData value.

    Returns:
        pd.DataFrame: Flattened raster data in a tabular format.
    """
    xx, yy = np.meshgrid(x, y)
    data = {'x': xx.flatten(), 'y': yy.flatten()}
    
    for i, band in enumerate(bands):
        band_data = band.flatten()
        if nodata is not None:
            band_data[band_data == nodata] = np.nan  # Replace NoData values with NaN
        data[band_names[i]] = band_data
    
    return pd.DataFrame(data)

def reproject_latlon(df, crs, epsgcode='epsg:4326'):
    """
    Reprojects raster coordinates to latitude and longitude.

    Parameters:
        df (pd.DataFrame): DataFrame containing raster data.
        crs: Source CRS of the raster.
        epsgcode (str): Target EPSG code for reprojection.

    Returns:
        pd.DataFrame: DataFrame with latitude and longitude columns.
    """
    transformer = Transformer.from_crs(crs, epsgcode, always_xy=True)
    df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)
    return df[['lat', 'lon'] + [col for col in df.columns if col not in ['x', 'y', 'lat', 'lon']]]

def raster2df(rpath, default_band_name='Band', epsgcode='epsg:4326'):
    """
    Converts raster data into a reprojected DataFrame.

    Parameters:
        rpath (str): Path to the raster file.
        default_band_name (str): Default prefix for band names.
        epsgcode (str): Target EPSG code for reprojection.

    Returns:
        pd.DataFrame: Reprojected raster data in tabular format.
    """
    x, y, bands, crs, band_names, nodata = read_xyz(rpath, default_band_name)
    df = xyz2df(x, y, bands, band_names, nodata)
    return reproject_latlon(df, crs, epsgcode)

def get_raster_info(tif_path):
    """
    Extracts raster metadata including projection, resolution, and bounding box.

    Parameters:
        tif_path (str): Path to the raster file.

    Returns:
        tuple: Raster projection, resolution, bounding box, and dimensions.
    """
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    xres = geotrans[1]
    yres = geotrans[5]
    w, h = ds.RasterXSize, ds.RasterYSize
    xmin, ymax = geotrans[0], geotrans[3]
    xmax = xmin + (xres * w)
    ymin = ymax + (yres * h)
    ds = None
    return proj, xres, yres, xmin, xmax, ymin, ymax, w, h

def get_nodata_value(raster_path):
    """
    Retrieves the NoData value of a raster.

    Parameters:
        raster_path (str): Path to the raster file.

    Returns:
        float: NoData value.
    """
    with rasterio.open(raster_path) as src:
        return src.nodata

def gdal_regrid(fi, fo, xmin, ymin, xmax, ymax, xres, yres,
                mode, t_epsg='EPSG:4979', overwrite=False):
    """
    Regrids a raster file using GDAL.

    Parameters:
        fi (str): Input raster file path.
        fo (str): Output raster file path.
        xmin, ymin, xmax, ymax (float): Bounding box.
        xres, yres (float): Target resolution.
        mode (str): Regridding mode ('num' or 'cat').
        t_epsg (str): Target EPSG code.
        overwrite (bool): Whether to overwrite existing output.

    Returns:
        None
    """
    if mode == 'num':
        ndv, algo, dtype = num_regrid_params()
    elif mode == 'cat':
        ndv, algo, dtype = cat_regrid_params()
    else:
        raise ValueError("Invalid mode. Use 'num' or 'cat'.")

    src_ndv = get_nodata_value(fi)
    dst_ndv = ndv

    print(f"Source NoData Value: {src_ndv}")
    print(f"Destination NoData Value: {dst_ndv}")

    overwrite_option = "-overwrite" if overwrite else ""
    output_width = round((xmax - xmin) / xres)
    output_height = round((ymax - ymin) / abs(yres))

    cmd = (f'gdalwarp -ot {dtype} -multi {overwrite_option} '
           f'-te {xmin} {ymin} {xmax} {ymax} '
           f'-ts {output_width} {output_height} '
           f'-r {algo} -t_srs {t_epsg} -tr {xres} {yres} -tap '
           f'-co compress=lzw -co num_threads=all_cpus -co TILED=YES '
           f'-srcnodata {src_ndv} -dstnodata {dst_ndv} '
           f'{fi} {fo}')

    os.system(cmd)

def cat_regrid_params():
    """
    Returns parameters for categorical regridding.

    Returns:
        tuple: NoData value, resampling algorithm, and data type.
    """
    return 0, 'near', 'Byte'

def num_regrid_params():
    """
    Returns parameters for numerical regridding.

    Returns:
        tuple: NoData value, resampling algorithm, and data type.
    """
    return -9999.0, 'bilinear', 'Float32'
