import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr
from rioxarray.merge import merge_arrays

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR

# custom imports
from modules import model_agreement as ma
from modules.utils import countries

############################################################################################
#### Generate model agreement maps and save as geotiff raster files
############################################################################################
for country in countries.keys():
    print(f"Stacking poverty maps for {country}")
    rasters = dict()
    for model in MODEL_NAMES:
        rasters[model] = (
            rxr.open_rasterio(
                os.path.join(INTERIM_DIR, "model_maps", f"{model}_{country}.tif"),
                masked=True,
            )
            .squeeze()
            .drop("band")
        )
    # spatial alignment and raster resampling to match Lee's maps, then stack along axis; 'model'
    rasters = xr.concat(
        [rasters[model_].rio.reproject_match(rasters["Lee"]) for model_ in MODEL_NAMES],
        dim="model",
    ).assign_coords(model=MODEL_NAMES)
    # determine spatial agreement (i.e., no. of models in agreement per pixel)
    ma.calculate_mode(rasters, return_freq=True).rio.to_raster(
        os.path.join(
            INTERIM_DIR, "raster_stacks", "spatial_agreement", f"{country}_models.tif"
        )
    )
print("Model stacks completed.")

# merge all the country agreement maps into a single raster
rasters = glob(
    os.path.join(INTERIM_DIR, "raster_stacks", "spatial_agreement", "*_models.tif")
)
rasters = merge_arrays(
    [
        (rxr.open_rasterio(raster, masked=True).squeeze())
        for raster in rasters
        if "spatial_agreement_map" not in raster
    ]
)
rasters.rio.to_raster(os.path.join(PROCESSED_DIR, "spatial_agreement_map.tif"))

# calculate pixel proportions for each class in the spatial agreement by country
proportions = pd.DataFrame()
rasters = glob(os.path.join(INTERIM_DIR, "raster_stacks", "spatial_agreement", "*.tif"))
for raster in rasters:
    country = os.path.basename(raster).split("_")[0]
    if country not in countries.keys():
        print("Calculating overall pixel proportions")
    print(f"Calculating pixel proportions for {country}")

    raster = rxr.open_rasterio(raster, masked=True).squeeze()
    # calculate pixel proportions for each class
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
    proportions = pd.concat([proportions, freq_table])
proportions = (
    proportions[
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
proportions.to_csv(
    os.path.join(PROCESSED_DIR, "agreement_pixel_proportions.csv"),
    index=False,
)
