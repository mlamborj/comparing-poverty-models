import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Point
from . import utils
from config import DATA_DIR

inputs = DATA_DIR / "raw"

countries = utils.countries
smod = utils.smod


def mccallum_model(country_name: str, n_bins: int = 3) -> xr.DataArray:
    """
    Function for loading McCallum's model data

    Args:
        country_name (str): Name of the country
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        xr.DataArray: raster data for the country
    """
    boundary = utils.read_boundary(country_name)
    rasext = "wq" if n_bins == 5 else "wc"
    # load data from McCallum's model
    mccallum = (
        rxr.open_rasterio(
            os.path.join(inputs, "McCallum", f"{country_name}_{rasext}.tif"),
            masked=True,
        )
        .squeeze()
        .rio.reproject("EPSG:4326")
        .rio.clip(boundary.geometry)
        .rio.set_nodata(np.nan)
    )
    return utils.fix_dims(mccallum)


def chi_model(country_name: str, n_bins: int = 3) -> xr.DataArray:
    """
    Function for loading Chi's model data.
    The relative wealth index (rwi) is optionally reclassified to terciles/quintiles for comparison with other models.

    Args:
        country_name (str): Name of the country
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        xr.DataArray: raster data for the country
    """
    boundary = utils.read_boundary(country_name)
    # load data from the Chi model
    country_code = countries[country_name].lower()
    chi = pd.read_csv(
        os.path.join(inputs, "Chi", f"{country_code}_relative_wealth_index.csv")
    )
    if n_bins is None:
        chi = chi.rename(columns={"rwi": "Chi"})
    else:
        # reclassify rwi to terciles/quintiles for comparison with other models
        chi.loc[:, "Chi"] = pd.qcut(chi["rwi"], n_bins, labels=False) + 1
    chi = (
        chi[["Chi", "latitude", "longitude"]]
        .pivot(index="latitude", columns="longitude")
        .sort_index(ascending=False)
        .droplevel(0, axis=1)
    )
    chi = (
        xr.DataArray(chi)
        .rio.write_crs("EPSG:4326")
        .rio.clip(boundary.geometry)
        .rio.write_nodata(np.nan)
    )
    return utils.fix_dims(chi)


def lee_model(country_name: str, n_bins: int = 3) -> xr.DataArray:
    """
    Function for loading Lee's model data.
    Individual country data is first merged then international wealth index (iwi)
    is reclassified to terciles/quintiles for comparison with other models.

    Args:
        country_name (str, optional): Name of the country.
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        xr.DataArray: raster data for the country
    """
    boundary = utils.read_boundary(country_name)
    # load data from Lee's model
    lee = pd.concat(
        [
            pd.read_csv(
                os.path.join(inputs, "Lee", f"{name}_estimated_wealth_index.csv")
            )
            for name in countries.keys()
        ]
    )
    if n_bins is None:
        lee = lee.rename(columns={"estimated_IWI": "Lee"})
    else:
        # reclassify iwi to terciles/quintiles for comparison with other models
        lee.loc[:, "Lee"] = pd.qcut(lee["estimated_IWI"], n_bins, labels=False) + 1
    lee = gpd.GeoDataFrame(
        lee[["country_name", "Lee"]],
        geometry=[Point(xy) for xy in zip(lee["lon"], lee["lat"])],
        crs="EPSG:4326",
    )
    lee = lee[lee["country_name"] == country_name].clip(boundary)
    return utils.rasterize_points(lee, "Lee")


def dhs_model_contemporary(
    model_name: str, country_name: str = None, n_bins: int = 3
) -> xr.DataArray:
    """
    Function for loading contemporary DHS data for respective models.
    The relative wealth index (rwi) is optionally reclassified to terciles/quintiles for comparison with other models.

    Args:
        country_name (str, optional): Name of the country.
        model_name (str): Name of the model. Determines the DHS survey year
        to compare with.
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        gpd.GeoDataFrame: point data for all countries
    """
    # boundary=utils.read_boundary(country_name)
    # load data from DHS
    dhs = pd.read_csv(os.path.join(inputs, "DHS", f"DHS_{model_name.lower()}.csv"))
    if country_name is not None:
        dhs = dhs[dhs["country_name"] == country_name.lower()]
    if n_bins is None:
        dhs = dhs.rename(columns={"wealth_index": "DHS"})
    else:
        # reclassify rwi to terciles/quintiles for comparison with other models
        dhs.loc[:, "DHS"] = pd.qcut(dhs["wealth_index"], n_bins, labels=False) + 1
    dhs = gpd.GeoDataFrame(
        dhs[["country_name", "urban_rural", "DHS"]],
        geometry=[Point(xy) for xy in zip(dhs["LONGNUM"], dhs["LATNUM"])],
        crs="EPSG:4326",
    )
    # buffer points to match DHS cluster size
    dhs["buffer"] = dhs["urban_rural"].map(lambda x: 0.018018 if x == 2 else 0.045045)
    dhs["geometry"] = dhs.geometry.buffer(dhs["buffer"])
    # dhs=dhs.clip(boundary)
    return utils.rasterize_polygons(dhs, "DHS")


def dhs_model_latest(country_name: str, n_bins: int = 3) -> gpd.GeoDataFrame:
    """
    Function for loading latest DHS data for respective countries.
    The relative wealth index (rwi) is optionally reclassified to terciles/quintiles for comparison with other models.

    Args:
        country_name (str): Name of the country.
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        gpd.GeoDataFrame: point data for all countries
    """
    # load data from DHS
    dhs = pd.read_csv(os.path.join(inputs, "DHS", f"{country_name.lower()}.csv"))
    if n_bins is None:
        dhs = dhs.rename(columns={"wealth_index": "DHS"})
    else:
        # reclassify rwi to terciles/quintiles for comparison with other models
        dhs.loc[:, "DHS"] = pd.qcut(dhs["wealth_index"], n_bins, labels=False) + 1
    dhs = gpd.GeoDataFrame(
        dhs[["urban_rural", "DHS"]],
        geometry=[Point(xy) for xy in zip(dhs["LONGNUM"], dhs["LATNUM"])],
        crs="EPSG:4326",
    )
    # buffer points to match DHS cluster size
    dhs["buffer"] = dhs["urban_rural"].map(lambda x: 0.018018 if x == 2 else 0.045045)
    dhs["geometry"] = dhs.geometry.buffer(dhs["buffer"])
    return utils.rasterize_polygons(dhs, "DHS")


def yeh_model(country_name: str, n_bins: int = 3) -> gpd.GeoDataFrame:
    """
    Function for loading Yeh's model data.

    Args:
        country_name (str, optional): Name of the country.
        n_bins (int, optional): Number of bins for reclassification. Defaults to 3.

    Returns:
        xr.DataArray: raster data for the country
    """
    # load country boundaries
    boundary = utils.read_boundary(country_name, admin_level=2)
    # load data from Yeh's model
    yeh = pd.read_csv(os.path.join(inputs, "Yeh", "geolevel2_dhs_indices_gadm2.csv"))
    yeh = yeh.groupby(by="geolev")[["svyid", "predictions"]].last()
    if n_bins is None:
        yeh = yeh.rename(columns={"predictions": "Yeh"})
    else:
        # reclassify iwi to quintiles for comparison with other models
        yeh.loc[:, "Yeh"] = pd.qcut(yeh["predictions"], n_bins, labels=False) + 1
    yeh = boundary.merge(
        yeh, left_on="GID_2", right_index=True, how="left", validate="1:1"
    )[["country_name", "GID_2", "NAME_1", "NAME_2", "svyid", "Yeh", "geometry"]]
    yeh["Yeh"] = yeh["Yeh"].astype(np.float64)  # required for rasterization
    yeh = yeh[yeh["country_name"] == country_name]
    return utils.rasterize_polygons(yeh, "Yeh")


def lee_etal(n_bins: int = None) -> pd.DataFrame:
    """
    Function for extracting McCallum's, Chi's, Yeh's and Lee's model predictions for all countries.
    Extracts McCallum's, Chi's and Yeh's model predictions (raw/quintiles) at Lee's point locations.

    Returns:
        pd.DataFrame: point values for all countries
    """
    boundaries = utils.read_boundary()
    models = pd.DataFrame()
    for name in countries.keys():
        boundary = boundaries[boundaries["country_name"] == name]
        ds = lee_model(name, n_bins).to_dataset(name="Lee")
        ds["smod"] = (
            smod.rio.clip(boundary.geometry)
            .rio.reproject_match(ds["Lee"])
            .where(ds["Lee"].notnull())
        )
        ds["McCallum"] = (
            mccallum_model(name, n_bins)
            .rio.reproject_match(ds["Lee"])
            .where(ds["Lee"].notnull())
        )
        ds["Chi"] = (
            chi_model(name, n_bins)
            .rio.reproject_match(ds["Lee"])
            .where(ds["Lee"].notnull())
        )
        ds["Yeh"] = (
            yeh_model(name, n_bins)
            .rio.reproject_match(ds["Lee"])
            .where(ds["Lee"].notnull())
        )
        ds = ds.to_dataframe()
        ds.loc[:, "country_name"] = name
        models = pd.concat([models, ds], ignore_index=True)
    models = models.reset_index(drop=True).dropna(
        subset=["McCallum", "Chi", "Yeh", "Lee"]
    )
    return models[["country_name", "smod", "McCallum", "Chi", "Yeh", "Lee"]]


def mccallum_etal(n_bins: int = None) -> pd.DataFrame:
    """
    Function for extracting Chi's, Yeh's and McCallum's model predictions for all countries.
    Extracts Chi's, Yeh's and McCallum's model predictions (raw/quintiles) at coincident
    raster locations.

    Args:
        n_bins (int, optional): Number of bins for reclassification. Defaults to None for raw values.

    Returns:
        pd.DataFrame: raster values for all countries
    """
    boundaries = utils.read_boundary()
    models = pd.DataFrame()
    for name in countries.keys():
        boundary = boundaries[boundaries["country_name"] == name]
        ds = mccallum_model(name, n_bins).to_dataset(name="McCallum")
        ds["smod"] = (
            smod.rio.clip(boundary.geometry)
            .rio.reproject_match(ds["McCallum"])
            .where(ds["McCallum"].notnull())
        )
        ds["Chi"] = (
            chi_model(name, n_bins)
            .rio.reproject_match(ds["McCallum"])
            .where(ds["McCallum"].notnull())
        )
        ds["Yeh"] = (
            yeh_model(name, n_bins)
            .rio.reproject_match(ds["McCallum"])
            .where(ds["McCallum"].notnull())
        )
        ds = ds.to_dataframe()
        ds.loc[:, "country_name"] = name
        models = pd.concat([models, ds], ignore_index=True)
    models = models.reset_index(drop=True).dropna(subset=["McCallum", "Chi", "Yeh"])
    return models[["country_name", "smod", "McCallum", "Chi", "Yeh"]]


def chi_yeh(n_bins: int = None) -> pd.DataFrame:
    """
    Function for extracting Chi's and Yeh's model predictions for all countries.
    Extracts Chi's and Yeh's model predictions at coincident raster locations.

    Args:
        n_bins (int, optional): Number of bins for reclassification. Defaults to None.

    Returns:
        pd.DataFrame: raster values for all countries
    """
    boundaries = utils.read_boundary()
    models = pd.DataFrame()
    for name in countries.keys():
        print("Processing ", name)
        boundary = boundaries[boundaries["country_name"] == name]
        ds = chi_model(name, n_bins).to_dataset(name="Chi")
        # ds=utils.fix_dims(ds)
        ds["smod"] = (
            smod.rio.clip(boundary.geometry)
            .rio.reproject_match(ds["Chi"])
            .where(ds["Chi"].notnull())
        )
        # yeh=utils.rasterize_polygons(, 'Yeh')
        ds["Yeh"] = (
            yeh_model(name, n_bins)
            .rio.reproject_match(ds["Chi"])
            .where(ds["Chi"].notnull())
        )
        ds = ds.to_dataframe()
        ds.loc[:, "country_name"] = name
        models = pd.concat([models, ds], ignore_index=True)
        print("Completed ", name)
    models = models.reset_index(drop=True).dropna(subset=["Chi", "Yeh"])
    return models[["country_name", "smod", "Chi", "Yeh"]]


# def dhs_comparisons(model: str, n_bins: int = None) -> pd.DataFrame:
#     """
#     Function for extracting model predictions at DHS cluster points for all countries.
#     The DHS data used is the latest available for the specified model.
#     Extracts Yeh, Chi's and McCallum's model predictions (terciles/quintiles) at DHS
#     cluster locations.

#     Args:
#         n_bins (int, optional): Number of bins for reclassification. Defaults to None.

#     Returns:
#         pd.DataFrame: point values for all countries
#     """
#     boundaries = utils.read_boundary()
#     dhs = rxr.open_rasterio(
#         os.path.join(outputs, "DHS_reference", f"dhs_{model.lower()}_{n_bins}.tif"),
#         masked=True,
#     ).squeeze()
#     models = pd.DataFrame()
#     for name in countries.keys():
#         boundary = boundaries[boundaries["country_name"] == name]
#         ds = dhs.to_dataset(name="DHS").rio.clip(boundary.geometry)
#         ds["smod"] = (
#             smod.rio.clip(boundary.geometry)
#             .rio.reproject_match(ds["DHS"])
#             .where(ds["DHS"].notnull())
#         )
#         if model == "Chi":
#             ds["Chi"] = (
#                 chi_model(name, n_bins)
#                 .rio.reproject_match(ds["DHS"])
#                 .where(ds["DHS"].notnull())
#             )
#         elif model == "Lee":
#             ds["Lee"] = (
#                 lee_model(name, n_bins)
#                 .rio.reproject_match(ds["DHS"])
#                 .where(ds["DHS"].notnull())
#             )
#         elif model == "McCallum":
#             ds["McCallum"] = (
#                 mccallum_model(name, n_bins)
#                 .rio.reproject_match(ds["DHS"])
#                 .where(ds["DHS"].notnull())
#             )
#         elif model == "Yeh":
#             ds["Yeh"] = (
#                 yeh_model(name, n_bins)
#                 .rio.reproject_match(ds["DHS"])
#                 .where(ds["DHS"].notnull())
#             )
#         ds = ds.to_dataframe()
#         ds.loc[:, "country_name"] = name
#         models = pd.concat([models, ds], ignore_index=True)
#     models = models.reset_index(drop=True).dropna(subset=["DHS", model])
#     return models[["country_name", "smod", "DHS", model]]


# def dhs_comparisons_latest(n_bins: int, majority_vote: str) -> pd.DataFrame:
#     """
#     Function for extracting model predictions at DHS cluster points for all countries.
#     The DHS data used is the latest survey available for each country.
#     Extracts Yeh, Chi's and McCallum's model predictions (terciles/quintiles) at DHS
#     cluster locations.

#     Args:
#         n_bins (int): Number of bins for reclassification.

#     Returns:
#         pd.DataFrame: point values for all countries
#     """
#     # ensemble=(rxr.open_rasterio(os.path.join(outputs, 'Ensemble_models', 'terciles', f'ensemble_model_{n_bins}.tif'), masked=True)
#     #           .squeeze())
#     ensemble = rxr.open_rasterio(
#         os.path.join(
#             outputs,
#             "Ensemble_models_new",
#             f"{majority_vote}",
#             "all_countries_models.tif",
#         ),
#         masked=True,
#     ).squeeze()
#     boundaries = utils.read_boundary()
#     models = pd.DataFrame()
#     for name in countries.keys():
#         print("Processing ", name)
#         boundary = boundaries[boundaries["country_name"] == name]
#         dhs = rxr.open_rasterio(
#             os.path.join(
#                 outputs, "DHS_reference", "terciles", f"{name}_dhs_{n_bins}.tif"
#             ),
#             masked=True,
#         ).squeeze()
#         ds = dhs.to_dataset(name="DHS").rio.clip(boundary.geometry)
#         ds["smod"] = (
#             smod.rio.clip(boundary.geometry)
#             .rio.reproject_match(ds["DHS"])
#             .where(ds["DHS"].notnull())
#         )
#         ds["Ensemble"] = (
#             ensemble.rio.clip(boundary.geometry)
#             .rio.reproject_match(ds["DHS"])
#             .where(ds["DHS"].notnull())
#         )

#         ds = ds.to_dataframe()
#         ds.loc[:, "country_name"] = name
#         models = pd.concat([models, ds], ignore_index=True)
#     models = models.dropna(subset=["smod", "DHS", "Ensemble"]).reset_index(drop=True)
#     return models[["country_name", "smod", "DHS", "Ensemble"]]


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
