import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays
from shapely.geometry import Point

from config import DATA_DIR, INTERIM_DIR, MODEL_NAMES

from . import utils

inputs = DATA_DIR / "raw"

countries = utils.countries
# smod = utils.smod


def mccallum_model(country_name: str, version: str = "terciles") -> xr.DataArray:
    """
    Function for loading McCallum's model data

    Args:
        country_name (str): Name of the country
        version (str): Version of the model to load ("terciles" or "quintiles")

    Returns:
        xr.DataArray: raster data for the country
    """
    # determine file name based on terciles/quintiles
    types = {"terciles": "wc", "quintiles": "wq"}

    print(
        f"McCallum model currently uses '{version}'. Change this behaviour if needed."
    )

    # load data from McCallum's model
    mccallum = (
        rxr.open_rasterio(
            os.path.join(inputs, "McCallum", f"{country_name}_{types[version]}.tif"),
            masked=False,
        )
        .squeeze()
        .rio.reproject("EPSG:4326")
        .rio.write_nodata(np.nan)
    )
    mccallum = mccallum.where(mccallum > 0)
    return utils.fix_dims(mccallum)  # ~2.5km resolution


def chi_model(country_name: str) -> xr.DataArray:
    """
    Function for loading Chi's model data.

    Args:
        country_name (str): Name of the country.

    Returns:
        xr.DataArray: raster data for the country
    """
    # load data from the Chi model
    country_code = countries[country_name].lower()
    chi = pd.read_csv(
        os.path.join(inputs, "Chi", f"{country_code}_relative_wealth_index.csv")
    )
    chi = chi.rename(columns={"rwi": "Chi"})
    chi = (
        chi[["Chi", "latitude", "longitude"]]
        .pivot(index="latitude", columns="longitude")
        .sort_index(ascending=False)
        .droplevel(0, axis=1)
    )
    chi = xr.DataArray(chi).rio.write_crs("EPSG:4326").rio.write_nodata(np.nan)
    return utils.fix_dims(chi)  # ~2.4km resolution


def lee_model(country_name: str) -> xr.DataArray:
    """
    Function for loading Lee's model data.

    Args:
        country_name (str, optional): Name of the country.

    Returns:
        xr.DataArray: raster data for the country
    """
    # load data from Lee's model
    lee = pd.read_csv(
        os.path.join(inputs, "Lee", f"{country_name}_estimated_wealth_index.csv")
    )
    lee = lee.rename(columns={"estimated_IWI": "Lee"})
    lee = gpd.GeoDataFrame(
        lee[["country_name", "Lee"]],
        geometry=[Point(xy) for xy in zip(lee["lon"], lee["lat"])],
        crs="EPSG:4326",
    )
    lee = lee[lee["country_name"] == country_name]
    return utils.rasterize_points(lee, "Lee", cell_size=0.0144)  # ~1.6km resolution


def yeh_model(country_name: str) -> xr.DataArray:
    """
    Function for loading Yeh's model data.

    Args:
        country_name (str, optional): Name of the country.

    Returns:
        xr.DataArray: raster data for the country
    """
    # load data from Yeh's model
    yeh = pd.read_csv(os.path.join(inputs, "Yeh", "cluster_pred_dhs_indices_gadm2.csv"))
    yeh = yeh.rename(columns={"index": "Yeh"})
    yeh = gpd.GeoDataFrame(
        yeh[["country", "Yeh"]],
        geometry=[Point(xy) for xy in zip(yeh["lon"], yeh["lat"])],
        crs="EPSG:4326",
    )
    yeh = yeh[yeh["country"] == country_name]
    return utils.rasterize_points(yeh, "Yeh", cell_size=0.06048)  # ~6.72km resolution


def dhs_model_contemporary(model_name: str, country_name: str = None) -> xr.DataArray:
    """
    Function for loading DHS data which was latest at the time each model was published.
    The DHS data used will differ based on the model and the country being compared.

    Args:
        country_name (str, optional): Name of the country. If None, loads data for all countries. Defaults to None.
        model_name (str): Name of the model. Determines the DHS survey year to compare with.

    Returns:
        xr.DataArray: rasterized DHS clusters where pixel value is DHS relative wealth index.
    """
    # load data from DHS
    dhs = pd.read_csv(os.path.join(inputs, "DHS", f"DHS_{model_name.lower()}.csv"))
    # filter for country if specified
    if country_name is not None:
        dhs = dhs[dhs["country_name"] == country_name.lower()]
    dhs = dhs.rename(columns={"wealth_index": "DHS"})
    dhs = gpd.GeoDataFrame(
        dhs[["country_name", "urban_rural", "DHS"]],
        geometry=[Point(xy) for xy in zip(dhs["LONGNUM"], dhs["LATNUM"])],
        crs="EPSG:4326",
    )
    # buffer points to match DHS cluster size
    dhs["buffer"] = dhs["urban_rural"].map(lambda x: 0.018018 if x == 2 else 0.045045)
    dhs["geometry"] = dhs.geometry.buffer(dhs["buffer"])
    # dhs=dhs.clip(boundary)
    return utils.rasterize_polygons(dhs, "DHS", resolution=0.018)  # ~2km resolution


def dhs_model_latest(country_name: str) -> xr.DataArray:
    """
    Function for loading latest DHS data for respective countries.
    The DHS data is latest as at 2022.

    Args:
        country_name (str): Name of the country.

    Returns:
        xr.DataArray: rasterized DHS clusters where pixel value is DHS relative wealth index.
    """
    # load data from DHS
    dhs = pd.read_csv(os.path.join(inputs, "DHS", f"{country_name.lower()}.csv"))
    dhs = dhs.rename(columns={"wealth_index": "DHS"})
    dhs = gpd.GeoDataFrame(
        dhs[["urban_rural", "DHS"]],
        geometry=[Point(xy) for xy in zip(dhs["LONGNUM"], dhs["LATNUM"])],
        crs="EPSG:4326",
    )
    # buffer points to match DHS cluster size
    dhs["buffer"] = dhs["urban_rural"].map(lambda x: 0.018018 if x == 2 else 0.045045)
    dhs["geometry"] = dhs.geometry.buffer(dhs["buffer"])
    return utils.rasterize_polygons(dhs, "DHS", resolution=0.018)  # ~2km resolution


def weighted_aggregation(
    da: xr.DataArray, gdf: gpd.GeoDataFrame, model: str, country: str
) -> gpd.GeoDataFrame:
    """
    Function for aggregating model pixels to admin 2.
    The function calculates population weighted index for each admin 2 unit, using
    WorldPop population density data.

    Args:
        da (xr.DataArray): rasterized model data
        gdf (gpd.GeoDataFrame): admin 2 polygons data
        model (str): model name
        country (str): country name

    Returns:
        gpd.GeoDataFrame: admin 2 data with population weighted model index
    """
    # prepare population data
    raster = (
        rxr.open_rasterio(
            os.path.join(
                DATA_DIR,
                f"external/population/{countries[country].lower()}_pd_2020_1km.tif",
            ),
            masked=True,
        )
        .squeeze()
        .rio.set_crs("EPSG:4326")
        .rio.write_nodata(np.nan)
        .to_dataset(name="popn")
    )
    # align common pixels in both rasters
    raster[f"{model}"] = da.rio.reproject_match(raster["popn"])
    raster["popn"] = raster["popn"].where(raster[f"{model}"].notnull())

    # calculate population weighted index and aggregate to admin 2
    raster[f"{model}_popn"] = (raster[f"{model}"] * raster["popn"]).rio.write_nodata(
        np.nan
    )
    for var in raster.data_vars.keys():
        gdf = utils.sample_polygons(raster[var], gdf, "sum").rename(
            columns={"sum": var}
        )
    gdf[f"{model}_weighted"] = (gdf[f"{model}_popn"] / gdf["popn"]).astype(
        "float"
    )  # .round()
    gdf = gdf.drop(columns=[f"{model}", "popn", f"{model}_popn"]).rename(
        columns={f"{model}_weighted": f"{model}_index"}
    )
    return gdf


def coincident_pixels(
    da: xr.DataArray, unanimous_only: bool = False, dim="model"
) -> xr.DataArray:
    """
    This function is for generating country mask based on concident pixels in the input rasters.
    It compares pixel values from the input rasters and returns the pixel only when all models
    have a predicted value in the pixel. Comparison is only made if all input models are not NaN
    in the pixel.

    Args:
        da (xr.DataArray): Stack of rasters to compare.
        unanimous_only (bool, optional): Whether to return pixels where all models are present or at least 2 are present.
            Set True for considering the unanimously present case. Defaults to False.
        dim (str, optional): Name of dimension along which rasters are stacked. Defaults to 'model'.

    Returns:
        xr.DataArray: raster with the mode of the input rasters.
    """
    # Extract numpy array from the DataArray
    np_array = da.values

    # Function to determine coincident pixels
    def resolve_mode(ndarray):
        # Ignore NaN values
        valid_values = ndarray[~np.isnan(ndarray)]
        if unanimous_only:
            if len(valid_values) != len(da.model):
                return np.nan  # not all models present
        else:
            if len(valid_values) < 2:
                return np.nan  # not enough models present
        return 1

    # Apply function along the axis
    result = np.apply_along_axis(resolve_mode, axis=da.get_axis_num(dim), arr=np_array)
    # Convert the result back into an xarray DataArray
    mask = (
        xr.DataArray(
            data=result.astype(np.float32),
            dims=[d for d in da.dims if d != dim],
            coords={k: v for k, v in da.coords.items() if k != dim},
        )
        .rio.write_crs(da.rio.crs)
        .rio.write_nodata(np.nan)
    )
    return mask


def generate_quantiles(raster: xr.DataArray, q: int = 3) -> xr.DataArray:
    """
    Function to generate quantiles for each model in the rasters DataArray. The
    function returns a DataArray with the quantile labels as pixel values.

    Args:
        rasters (xr.DataArray): Raster for which quantiles are to be generated.
        q (int): Number of quantiles; e.g. 5 for quintiles. Defaults to 3 for terciles.

    Returns:
        xr.DataArray: DataArray with quantile labels as pixel values.
    """
    # calculate quantiles, ignoring NaN values
    quantiles = raster.quantile(np.linspace(0, 1, q + 1)[1:-1], skipna=True).values
    # flatten the original data array to apply digitize
    flat_data = raster.values.flatten()
    # assign bins based on the quantiles; +1 to make it 1-indexed
    quantiles = np.digitize(flat_data, quantiles, right=True).astype(np.float32) + 1
    # retain NaN values from the original data
    quantiles = np.where(np.isnan(flat_data), np.nan, quantiles)
    # reshape back to the original shape
    quantiles = quantiles.reshape(raster.shape)
    # Creating a new DataArray with quantile labels
    da = (
        xr.DataArray(
            quantiles,
            dims=[d for d in raster.dims if d != "model"],
            coords={k: v for k, v in raster.coords.items() if k != "model"},
        )
        .rio.write_crs(raster.rio.crs)
        .rio.write_nodata(np.nan)
    )
    return da


def generate_weighted_quantiles(
    raster: xr.DataArray, country: str, q: int = 3
) -> xr.DataArray:
    """
    Generate weighted quantiles (e.g., terciles or quintiles) for a raster using
    population weights from another raster (possibly with a different CRS).

    Args:
        raster (xr.DataArray): Raster with model predictions.
        country (str): Name of the country.
        q (int): Number of quantiles (e.g., 3 for terciles, 5 for quintiles).

    Returns:
        xr.DataArray: DataArray of quantile labels (1..q).
    """
    # prepare population data
    if country not in countries.keys():
        # aggregate population for all countries
        popn = (
            merge_arrays(
                [
                    rxr.open_rasterio(
                        os.path.join(
                            DATA_DIR,
                            f"external/population/{countries[cntry].lower()}_pd_2020_1km.tif",
                        ),
                        masked=True,
                    ).squeeze()
                    for cntry in countries.keys()
                ],
                nodata=np.nan,
            )
            .rio.set_crs("EPSG:4326")
            .rio.write_nodata(np.nan)
        )
    else:
        popn = (
            rxr.open_rasterio(
                os.path.join(
                    DATA_DIR,
                    f"external/population/{countries[country].lower()}_pd_2020_1km.tif",
                ),
                masked=True,
            )
            .squeeze()
            .rio.set_crs("EPSG:4326")
            .rio.write_nodata(np.nan)
        )
    # match raster CRS and resolution
    popn = popn.rio.reproject_match(raster, resampling=Resampling.bilinear)

    # flatten arrays
    values = raster.values.flatten()
    weights = popn.values.flatten()

    # mask out NaNs and invalid popn values
    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    # weighted quantile thresholds
    probs = np.linspace(0, 1, q + 1)[1:-1]
    # sort values and calculate cummulative sum of weights
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cdf = np.cumsum(weights_sorted) / np.sum(weights_sorted)
    thresholds = np.interp(probs, cdf, values_sorted)

    # digitize original (unmasked) values
    binned = np.digitize(raster.values, thresholds).astype(np.float32) + 1
    binned = np.where(np.isnan(raster.values), np.nan, binned)

    # return DataArray with quantile labels
    da = (
        xr.DataArray(
            binned,
            dims=[d for d in raster.dims if d != "model"],
            coords={k: v for k, v in raster.coords.items() if k != "model"},
        )
        .rio.write_crs(raster.rio.crs)
        .rio.write_nodata(np.nan)
    )

    return da


def generate_quantiles_v(col: pd.Series, q: int = 3) -> pd.Series:
    """
    Function to generate quantiles for each model in the dataframe. The function returns a Series
    with the quantile labels as values.

    Args:
        col (pd.Series): Column for which quantiles are to be generated.
        q (int, optional): Number of quantiles e.g., 5 for quintiles. Defaults to 3 for terciles.

    Returns:
        pd.Series: Column with quantile labels as values.
    """
    # quantile labels
    labels = list(range(1, q + 1))
    try:
        quantiles = pd.qcut(col, q, labels=labels).astype(float)
    except ValueError:
        # raised if there are too few unique values (McCallum in DRC)
        quantiles = pd.qcut(col.rank(method="first"), q, labels=labels).astype(float)
    return quantiles


def spatial_alignment(country: str, model_list: list = MODEL_NAMES) -> xr.DataArray:
    """
    Function to spatially align rasters and resample pixel sizes to a common resolution.
    The function reprojects rasters to match Lee's (1.6km) / Yeh's (6.72km) maps and generates a mask for analysis.

    Args:
        country (str): Name of the country for which the processing is done.
        models (list): List of model names to process. Defaults to MODEL_NAMES.
    Returns:
        xr.DataArray: Stacked rasters with spatial alignment.
    """
    rasters = dict()
    for model in model_list:
        # Path to ensemble models
        if model == "Ensemble":
            input_path = os.path.join(
                INTERIM_DIR,
                "raster_stacks",
                "majority_ensemble",
                f"{country}_ensemble.tif",
            )
        elif model.endswith("ensemble"):
            input_path = os.path.join(
                INTERIM_DIR,
                "raster_stacks",
                "leave-one-out",
                f"{country}_{model}.tif",
            )
        else:
            input_path = os.path.join(
                INTERIM_DIR, "rasterized", f"{model}_{country}.tif"
            )
        # read the raster for the model and country
        rasters[model] = (
            rxr.open_rasterio(
                input_path,
                masked=True,
            )
            .squeeze()
            .drop("band")
        )

    # spatial alignment and raster resampling to match Lee's maps
    for model in model_list:
        if model in ["Yeh", "Chi"]:
            rasters[model] = rasters[model].rio.reproject_match(
                rasters["Lee"], resampling=Resampling.bilinear
            )
        elif model == "McCallum":
            rasters[model] = rasters[model].rio.reproject_match(
                rasters["Lee"], resampling=Resampling.nearest
            )
        elif model == "DHS":
            rasters[model] = rasters[model].rio.reproject_match(
                rasters[model_list[1]], resampling=Resampling.bilinear
            )

    # # try downsampling to Yeh's courser resolution
    # for model in model_list:
    #     if model in ["Lee", "Chi"]:
    #         rasters[model] = rasters[model].rio.reproject_match(
    #             rasters["Yeh"], resampling=Resampling.bilinear
    #         )
    #     elif model == "McCallum":
    #         rasters[model] = rasters[model].rio.reproject_match(
    #             rasters["Yeh"], resampling=Resampling.nearest
    #         )
    #     elif model == "DHS":
    #         rasters[model] = rasters[model].rio.reproject_match(
    #             rasters["Ensemble"], resampling=Resampling.bilinear
    #         )
    # stack along 'model' axis
    rasters = xr.concat(
        list(rasters.values()),
        dim="model",
    ).assign_coords(model=model_list)

    return rasters


def align_dhs(
    dhs: xr.DataArray, smod: xr.DataArray, model_name: str, country: str
) -> xr.DataArray:
    """Aligns the DHS and SMOD rasters to the specified model raster.

    Args:
        dhs (xr.DataArray): The DHS raster to align.
        smod (xr.DataArray): The SMOD raster to align.
        model_name (str): The name of the model raster to align to.
        country (str): The name of the country being processed.

    Returns:
        xr.DataArray: The aligned raster.
    """
    # read the model raster for the country
    da = rxr.open_rasterio(
        os.path.join(INTERIM_DIR, "rasterized", f"{model_name}_{country}.tif"),
        masked=True,
    ).squeeze()

    smod = rxr.open_rasterio(
        os.path.join(inputs, f"SMOD/{countries[country]}_smod.tif"), masked=True
    ).squeeze()

    # align the DHS and SMOD rasters to the model raster
    dhs = dhs.rio.reproject_match(da, resampling=Resampling.bilinear).squeeze()
    smod = smod.rio.reproject_match(da, resampling=Resampling.nearest).squeeze()

    da = xr.concat([da, dhs, smod], dim="model").assign_coords(
        model=[model_name, "DHS", "smod"]
    )
    # # generate mask showing which pixels to include in the analysis (i.e., only where the DHS and model overlap)
    # mask = coincident_pixels(da, unanimous_only=True)

    # return da.where(mask.notnull())
    return da


if __name__ == "__main__":
    # for n_bins in [5]:
    #     # for model in ['Chi', 'Lee', 'McCallum', 'Yeh']:
    #     #     dhs_comparisons(model, n_bins).to_csv(os.path.join(outputs, 'disaggregated_analysis', f'{model}_dhs_{n_bins}.csv'), index=False)

    #     for name, model in zip(['McCallum'], [mccallum_model]):
    #         for country in countries.keys():
    #             print("Processing ", name, country)
    #             model(country, n_bins).rio.to_raster(os.path.join(outputs, 'maps', f'{name}_{country}_{n_bins}.tif'), compress='lzw')

    # for country in countries.keys():
    #     print("Processing ", country)
    #     dhs_model_latest(country, 3).rio.to_raster(os.path.join(outputs, 'DHS_reference', 'terciles', f'{country}_dhs_3.tif'), compress='lzw')
    # for mv in [
    #     "Chi_Lee_McCallum",
    #     "Chi_Lee_Yeh",
    #     "Chi_McCallum_Yeh",
    #     "Lee_McCallum_Yeh",
    # ]:
    #     dhs_comparisons_latest(3, mv).to_csv(
    #         os.path.join(
    #             outputs, "Ensemble_models_new", f"{mv}", f"DHS_{mv}_ensemble.csv"
    #         ),
    #         index=False,
    #     )
    print("Testing complete.")
