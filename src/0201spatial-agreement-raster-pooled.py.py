import os

import numpy as np
import pandas as pd
import xarray as xr
from rioxarray.merge import merge_arrays

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in at least 2 models and calculate quantiles for each model
############################################################################################
raster_dir = []
for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution (1.6 km)
    rasters = sampling.spatial_alignment(
        country, raster_dir=os.path.join(INTERIM_DIR, "rasterized")
    )
    # generate mask showing which pixels to include in the analysis (i.e., at least 2 models overlapping)
    mask = sampling.coincident_pixels(rasters, unanimous_only=False)

    # mask all models with the country mask
    raster_dir.append(rasters.where(mask.notnull()).squeeze())

# merge all the country ensemble maps into a single raster
da = merge_arrays(raster_dir, nodata=np.nan)

# calculate quantiles for each model for pooled countries
quantiles = dict()
for model_name in MODEL_NAMES:
    # calculate quantiles, ignoring McCallum's model
    quantiles[model_name] = (
        da.sel(model=model_name).drop("model")
        if model_name == "McCallum"
        else sampling.generate_quantiles(da.sel(model=model_name), q=3)
    )
# stack rasters along the 'model' dimension
quantiles = xr.concat(quantiles.values(), dim="model").assign_coords(model=MODEL_NAMES)

# ############################################################################################
# #### Determine wealth class in overlapping pixels by majority vote and generate maps
# #### Calculate summary statistics for majority-vote ensemble by country
# ############################################################################################

out_path = os.path.join(PROCESSED_DIR, "pixel-wise/terciles/pooled/majority")
print("Calculating majority vote ensemble")
# determine majority class label (i.e., pixel value with majority vote of models)
ensemble = ma.calculate_mode(quantiles, return_freq=False)
ensemble.rio.to_raster(os.path.join(out_path, "majority_ensemble_map.tif"))
############################################################################################
#### Determine spatial agreement of quantiles in overlapping pixels and generate maps
############################################################################################
print("Calculating spatial agreement\n")
# determine spatial agreement (i.e., no. of models in agreement per pixel)
agreement = ma.calculate_mode(quantiles, return_freq=True)
agreement.rio.to_raster(os.path.join(out_path, "spatial_agreement_map.tif"))
print("All countries completed.")

####################################################################################
#### Calculate summary statistics for majority-vote ensemble by country
####################################################################################

print("Calculating summary statistics for the majority-vote ensemble")
stats = pd.DataFrame()
raster_list = [c for c in countries.keys()]
raster_list.append("Pooled")
for country in raster_list:
    if country not in countries.keys():
        print("Summarising overall statistics")
        raster = ensemble.copy()
    else:
        print(f"Summarising statistics for {country}")
        boundary = utils.read_boundary(country)

        raster = ensemble.rio.clip(boundary.geometry)
    # calculate pixel stats for each class
    freq_table = ma.frequency_table(
        raster,
        # classes={
        #     1: "Poorest",
        #     2: "Poorer",
        #     3: "Average",
        #     4: "Richer",
        #     5: "Richest",
        # },
        classes={1: "Poor", 2: "Average", 3: "Richer"},
    )
    freq_table.loc[:, "Country"] = country if country in countries.keys() else "Overall"
    freq_table = (
        freq_table.pivot(index="Country", columns="value", values="proportion")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    stats = pd.concat([stats, freq_table])
stats = (
    # stats[["Country", "Poorest", "Poorer", "Average", "Richer", "Richest"]]
    stats[["Country", "Poor", "Average", "Richer"]]
    .fillna(0)
    .sort_values(by="Country")
    .reset_index(drop=True)
)
stats.to_csv(os.path.join(out_path, "majority_pixel_stats.csv"), index=False)
####################################################################################
#### Calculate summary statistics for spatial agreement by country
####################################################################################

print("Calculating summary statistics for spatial agreement")
stats = pd.DataFrame()

for country in raster_list:
    if country not in countries.keys():
        print("Summarising overall statistics")
        raster = agreement.copy()
    else:
        print(f"Summarising statistics for {country}")
        boundary = utils.read_boundary(country)

        raster = agreement.rio.clip(boundary.geometry)
    # calculate pixel stats for each class
    freq_table = ma.frequency_table(
        raster,
        classes={
            0: "No agreement",
            1: "Split agreement",
            2: "2 models agree",
            3: "3 models agree",
            4: "All models agree",
        },
    )
    freq_table.loc[:, "Country"] = country if country in countries.keys() else "Overall"
    freq_table = (
        freq_table.pivot(index="Country", columns="value", values="proportion")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    stats = pd.concat([stats, freq_table])
stats = (
    stats[
        [
            "Country",
            "No agreement",
            "Split agreement",
            "2 models agree",
            "3 models agree",
            "All models agree",
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
