import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio import MemoryFile, features, transform

from config import COUNTRIES_FILE, GADM_FILE, SMOD_FILE

countries = pd.read_csv(COUNTRIES_FILE)
countries = dict(zip(countries["name"], countries["alpha-3"]))
smod = rxr.open_rasterio(SMOD_FILE, masked=True).squeeze()


def read_boundary(country_name: str = None, admin_level: int = 0) -> gpd.GeoDataFrame:
    """
    Function for reading country boundary data.
    Reads country boundary data from the GADM database and returns a GeoDataFrame
    of either admin 0 or admin 2 level.

    Args:
        country_name (str): Name of the country. Defaults to None.
        admin_level (int, optional): Admin level. Defaults to 0.

    Returns:
        gpd.GeoDataFrame: boundary data for the country
    """
    if admin_level == 0:
        gdf = gpd.read_file(
            GADM_FILE,
            layer="country_boundaries",
        )
    else:
        gdf = gpd.read_file(
            GADM_FILE,
            layer="district_boundaries",
        )
    return gdf[gdf["country_name"] == country_name] if country_name else gdf


def sample_points(
    raster: xr.DataArray, points: gpd.GeoDataFrame, column: str, out_crs="EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Function for extracting raster values at point locations.

    Args:
        raster (xr.DataArray): raster data
        points (gpd.GeoDataFrame): point data
        column (str): column name for extracted raster values
        out_crs (str, optional): output crs. Defaults to 'EPSG:4326'.

    Returns:
        gpd.GeoDataFrame: point data with extracted raster values
    """
    # prepare raster metadata
    meta = {
        "driver": "GTiff",
        "dtype": raster.dtype,
        "count": 1,
        "crs": raster.rio.crs,
        "transform": raster.rio.transform(),
        "width": raster.rio.width,
        "height": raster.rio.height,
    }
    # export raster to memory to avoid unnecessary writing to disk
    with MemoryFile() as src:
        with src.open(**meta) as dataset:
            dataset.write(raster.values, 1)
        with src.open() as dataset:
            # transfom point crs to match raster
            points = points.to_crs(dataset.crs) if points.crs != dataset.crs else points
            coord_list = list(zip(points["geometry"].x, points["geometry"].y))
            points.loc[:, column] = [
                x[0] for x in dataset.sample(coord_list, indexes=1, masked=True)
            ]
            points.loc[:, column] = points[column].map(
                lambda x: x if isinstance(x, float) else x.astype(np.float64)
            )
    return points.to_crs(out_crs) if points.crs != out_crs else points


def rasterize_polygons(
    polygons: gpd.GeoDataFrame, column: str, resolution: float = 0.01
) -> xr.DataArray:
    """
    Function for rasterizing polygons to a raster.
    Creates a raster from polygons with values from a specified column of the given resolution.
    Args:
        polygons (gpd.GeoDataFrame): polygons to rasterize
        column (str): column name for raster values
        resolution (float): desired resolution

    Returns:
        xr.DataArray: rasterized data
    """

    xmin, ymin, xmax, ymax = polygons.total_bounds
    pixel_size = resolution  # todo make provision for geographic and projected crs

    # Create the coordinate grids
    x = np.arange(xmin, xmax + pixel_size, pixel_size)
    y = np.arange(ymin, ymax + pixel_size, pixel_size)

    template = xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.float32),
        dims=["y", "x"],
        coords={
            "y": y[::-1],
            "x": x,
        },  # Note: y should be reversed to have top-left (NW) origin
    ).rio.write_crs(polygons.crs)

    burned = features.rasterize(
        shapes=(
            (geom, value)
            for geom, value in zip(polygons.geometry, polygons[f"{column}"])
        ),
        out_shape=(len(template.y), len(template.x)),
        fill=np.nan,
        transform=transform.from_origin(
            west=xmin, north=ymax, xsize=pixel_size, ysize=pixel_size
        ),
        all_touched=True,
        dtype=np.float32,
    )
    raster = (
        xr.DataArray(burned, dims=["y", "x"], coords={"y": template.y, "x": template.x})
        .rio.write_crs(polygons.crs)
        .rio.write_nodata(np.nan)
    )
    return raster


def rasterize_points(
    points: gpd.GeoDataFrame, column: str, cell_size: float = 0.0144
) -> xr.DataArray:
    """
    Function for rasterizing points to a raster.
    Creates a raster from polygons with values from a specified column of the given resolution.
    Args:
        points (gpd.GeoDataFrame): points to rasterize
        column (str): column name for raster values
        cell_size (float): desired cell resolution

    Returns:
        xr.DataArray: rasterized data
    """
    # Extract geometric information
    xmin, ymin, xmax, ymax = points.total_bounds

    # # Define raster dimensions
    x_coods = np.arange(xmin, xmax + cell_size, cell_size)
    y_coods = np.arange(ymin, ymax + cell_size, cell_size)
    ncols = len(x_coods)
    nrows = len(y_coods)

    # Get indices of the grid cells into which x and y coordinates fall into
    agg = pd.DataFrame(
        {
            "x_idx": np.digitize(points.geometry.x.values, x_coods) - 1,
            "y_idx": np.digitize(points.geometry.y.values, y_coods) - 1,
            "value": points[column].values,
        }
    )
    # Aggregate values per cell using maximum
    agg = agg.groupby(["y_idx", "x_idx"])["value"].max().reset_index()

    # Create raster and assign the aggregated values to the raster array
    out_raster = np.full((nrows, ncols), np.nan, dtype=np.float32)
    out_raster[agg["y_idx"], agg["x_idx"]] = agg["value"]
    out_raster = np.flipud(out_raster)  # rasterio raster origin is at the top-left

    # Output raster
    da = xr.DataArray(
        out_raster, dims=["y", "x"], coords={"y": y_coods[::-1], "x": x_coods}
    )
    # Write spatial information
    da = (
        da.rio.write_crs(points.crs)
        .rio.write_transform(
            transform.from_origin(
                west=xmin, north=ymax, xsize=cell_size, ysize=cell_size
            )
        )
        .rio.write_nodata(np.nan)
    )
    return da
    # return out_raster.shape


def fix_dims(raster):
    """
    Utility function to ensure that spatial dimensions conform to
    rasterio convention.
    Renames dimensions to 'x' and 'y' in the returned dataset.

    Args:
        raster (_type_): Raster to be fixed.

    Returns:
        _type_: Fixed raster.
    """
    y, x = raster.dims.keys() if isinstance(raster, xr.Dataset) else raster.dims
    # fix raster dimensions if required
    if y != "y" or x != "x":
        raster = raster.rename({y: "y", x: "x"})
    return raster


# if __name__ == "__main__":
#     # Example usage of the functions
#     gdf = read_boundary("Angola", admin_level=0)
#     from model_sampling import lee_model, mccallum_model, yeh_model, chi_model
#     rasterize_points(lee_model('Angola'), 'Lee').rio.to_raster(os.path.join(outputs, 'angola_lee.tif'))
#     fix_dims(mccallum_model('Angola')).rio.to_raster(os.path.join(outputs, 'angola_mccallum.tif'))
#     chi_model('Angola').rio.to_raster(os.path.join(outputs, 'angola_chi.tif'))
#     rasterize_polygons(yeh_model('Angola'), 'Yeh').rio.to_raster(os.path.join(outputs, 'angola_yeh.tif'))
