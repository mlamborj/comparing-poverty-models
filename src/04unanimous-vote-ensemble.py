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

##############################################################################################
#### Generate unanimous-vote ensemble maps and save as geotiff raster files
##############################################################################################
# align model outputs for each country
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
    # determine unanimous class label (i.e., pixel value with unanimous vote of models)
    ma.unanimous_mode(rasters).rio.to_raster(
        os.path.join(
            INTERIM_DIR, "raster_stacks", "unanimous_ensemble", f"{country}_models.tif"
        )
    )
print("Model stacks completed.")

# merge all the country ensemble maps into a single raster
rasters = glob(
    os.path.join(INTERIM_DIR, "raster_stacks", "unanimous_ensemble", "*_models.tif")
)
rasters = merge_arrays(
    [
        (rxr.open_rasterio(raster, masked=True).squeeze())
        for raster in rasters
        if "unanimous_ensemble_map" not in raster
    ]
)
rasters.rio.to_raster(os.path.join(PROCESSED_DIR, "unanimous_ensemble_map.tif"))

# calculate pixel proportions for each class in the unanimous ensembles by country
proportions = pd.DataFrame()
rasters = glob(
    os.path.join(INTERIM_DIR, "raster_stacks", "unanimous_ensemble", "*.tif")
)
for raster in rasters:
    country = os.path.basename(raster).split("_")[0]
    if country not in countries.keys():
        print("Calculating overall pixel proportions")
    print(f"Calculating pixel proportions for {country}")

    raster = rxr.open_rasterio(raster, masked=True).squeeze()
    # calculate pixel proportions for each class
    freq_table = ma.frequency_table(
        raster, classes={1: "Poor", 2: "Average", 3: "Richer"}
    )
    freq_table.loc[:, "Country"] = country if country in countries.keys() else "Overall"
    freq_table = (
        freq_table.pivot(index="Country", columns="value", values="proportion")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    proportions = pd.concat([proportions, freq_table])
proportions = (
    proportions[["Country", "Poor", "Average", "Richer"]]
    .fillna(0)
    .sort_values(by="Country")
    .reset_index(drop=True)
)
proportions.to_csv(
    os.path.join(PROCESSED_DIR, "unanimous_pixel_proportions.csv"),
    index=False,
)
