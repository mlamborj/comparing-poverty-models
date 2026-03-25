import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr
from rioxarray.merge import merge_arrays

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Calculate quantiles for each model on a country-by-country basis, then compare
#### overlapping pixels across models for each country, generating spatial agreement maps
############################################################################################
# countries_list=iter(countries.keys())
# country=next(countries_list)
# MODEL_NAMES.remove(
#     "McCallum"
# )  # consider only models with continuous-value wealth indices

for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution
    rasters = sampling.spatial_alignment(country, resolution="Lee")
    # generate mask showing which pixels to include in the analysis (i.e., models overlapping)
    mask = sampling.coincident_pixels(rasters, full_overlap=False)
    # calculate quantiles for each model
    quantiles = dict()
    for model_name in MODEL_NAMES:
        print(f"Generating wealth quantiles for {model_name}")

        da = rasters.sel(model=model_name).squeeze().drop("model")
        # calculate quantiles, ignoring McCallum's model
        quantiles[model_name] = (
            da
            if model_name == "McCallum"
            # else sampling.generate_weighted_quantiles(da, country, q=3)
            else sampling.generate_quantiles(da, q=5)  # unweighted
        )
    # stack rasters along the 'model' dimension
    quantiles = (
        xr.concat(quantiles.values(), dim="model")
        .assign_coords(model=MODEL_NAMES)
        .where(mask.notnull())  # mask each model's data with the country mask
    )

    ############################################################################################
    #### Determine wealth class in overlapping pixels by majority vote and generate maps
    ############################################################################################
    print("Calculating majority vote ensemble")
    # determine majority class label (i.e., pixel value with majority vote of models)
    ma.calculate_mode(quantiles, return_freq=False).rio.to_raster(
        os.path.join(
            INTERIM_DIR, "raster_stacks/partial-overlap", f"{country}_ensemble.tif"
        )
    )
    ############################################################################################
    #### Determine spatial agreement of quantiles in overlapping pixels and generate maps
    ############################################################################################
    print("Calculating spatial agreement\n")
    # determine spatial agreement (i.e., no. of models in agreement per pixel)
    ma.calculate_mode(quantiles, return_freq=True).rio.to_raster(
        os.path.join(
            INTERIM_DIR, "raster_stacks/partial-overlap", f"{country}_agrmnt.tif"
        )
    )
print("All countries completed.")

####################################################################################
#### Calculate summary statistics for majority-vote ensemble by country
####################################################################################
strata = {1: "rural", 2: "urban", None: "all"}
# merge all the country ensemble maps into a single raster
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/quintiles/unpooled/partial-overlap")
raster_dir = glob(
    os.path.join(INTERIM_DIR, "raster_stacks/partial-overlap", "*_ensemble.tif")
)
rasters = merge_arrays(
    [(rxr.open_rasterio(raster, masked=True).squeeze()) for raster in raster_dir]
)
rasters.rio.to_raster(os.path.join(out_path, "majority_ensemble_map.tif"))
print("Majority-vote ensemble map completed.")

print("Calculating summary statistics for the majority-vote ensemble")
stats = pd.DataFrame()
raster_dir.append(os.path.join(out_path, "majority_ensemble_map.tif"))
for raster in raster_dir:
    country = os.path.basename(raster).split("_")[0]
    if country not in countries.keys():
        print("Summarising overall statistics")
    else:
        print(f"Summarising statistics for {country}")

    raster = (
        rxr.open_rasterio(raster, masked=True).squeeze().to_dataset(name="ensemble")
    )
    # get urbanisation raster
    raster = utils.urbanisation_class(raster, country=country)
    # calculate pixel stats for each class by urbanisation
    for cluster in strata.keys():
        # mask raster by urban/rural/all
        if cluster is not None:
            ras = raster["ensemble"].where(raster["smod"] == cluster).squeeze()
        else:
            ras = raster["ensemble"].squeeze()

        freq_table = ma.frequency_table(
            ras,
            classes={
                1: "Poorest",
                2: "Poorer",
                3: "Average",
                4: "Richer",
                5: "Richest",
            },
            # classes={1: "Poor", 2: "Average", 3: "Richer"},
        )
        freq_table.loc[:, "Country"] = (
            country if country in countries.keys() else "Overall"
        )
        freq_table = (
            freq_table.pivot(index="Country", columns="value", values="proportion")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        freq_table.loc[:, "Cluster"] = strata[cluster]
        stats = pd.concat([stats, freq_table])
stats = (
    stats[["Country", "Cluster", "Poorest", "Poorer", "Average", "Richer", "Richest"]]
    # stats[["Country", "Cluster", "Poor", "Average", "Richer"]]
    .fillna(0)
    .sort_values(by=["Country", "Cluster"])
    .reset_index(drop=True)
)
stats.to_csv(os.path.join(out_path, "majority_pixel_stats.csv"), index=False)
####################################################################################
#### Calculate summary statistics for spatial agreement by country
####################################################################################
# merge all the country agreement maps into a single raster
raster_dir = glob(
    os.path.join(INTERIM_DIR, "raster_stacks/partial-overlap", "*_agrmnt.tif")
)
rasters = merge_arrays(
    [rxr.open_rasterio(raster, masked=True).squeeze() for raster in raster_dir]
)
rasters.rio.to_raster(os.path.join(out_path, "spatial_agreement_map_m.tif"))
print("Spatial agreement map completed.")

print("Calculating summary statistics for spatial agreement")
stats = pd.DataFrame()
raster_dir.append(os.path.join(out_path, "spatial_agreement_map_m.tif"))
for raster in raster_dir:
    country = os.path.basename(raster).split("_")[0]
    if country not in countries.keys():
        print("Summarising overall statistics")
    else:
        print(f"Summarising statistics for {country}")

    raster = (
        rxr.open_rasterio(raster, masked=True).squeeze().to_dataset(name="ensemble")
    )
    # get urbanisation raster
    raster = utils.urbanisation_class(raster, country=country)
    # calculate pixel stats for each class by urbanisation
    for cluster in strata.keys():
        # mask raster by urban/rural/all
        if cluster is not None:
            ras = raster["ensemble"].where(raster["smod"] == cluster).squeeze()
        else:
            ras = raster["ensemble"].squeeze()

        freq_table = ma.frequency_table(
            ras,
            classes={
                0: "No agreement",
                1: "Split agreement",
                2: "2 models agree",
                3: "3 models agree",
                4: "4 models agree",
            },
        )
        freq_table.loc[:, "Country"] = (
            country if country in countries.keys() else "Overall"
        )
        freq_table = (
            freq_table.pivot(index="Country", columns="value", values="proportion")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        freq_table.loc[:, "Cluster"] = strata[cluster]
        stats = pd.concat([stats, freq_table])
stats = (
    stats[
        [
            "Country",
            "Cluster",
            "No agreement",  # this category is not needed if considering pixels with all 4 models overlapping
            # "Split agreement",  # this category is not needed if only 3 models are being compared
            "2 models agree",
            "3 models agree",
            # "4 models agree",  # this category is not needed if only 3 models are being compared
        ]
    ]
    .fillna(0)
    .sort_values(by="Country")
    .reset_index(drop=True)
)
stats.to_csv(
    os.path.join(out_path, "agreement_pixel_stats.csv"),
    index=False,
)
