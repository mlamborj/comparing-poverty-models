# %%
import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr
from rioxarray.merge import merge_arrays

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in all 4 models and calculate terciles for each model
############################################################################################
for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution (1.6 km)
    rasters = sampling.spatial_alignment(
        country, raster_dir=os.path.join(INTERIM_DIR, "rasterized")
    )
    # generate mask showing which pixels to include in the analysis (i.e., all 4 models overlapping)
    mask = sampling.coincident_pixels(rasters, unanimous_only=True)
    # calculate terciles for each model
    terciles = dict()
    for model_name in MODEL_NAMES:
        print(f"Generating wealth terciles for {model_name}")
        # mask each model's data with the country mask
        da = rasters.sel(model=model_name).where(mask.notnull()).squeeze().drop("model")
        # calculate terciles, ignoring McCallum's model
        terciles[model_name] = (
            da if model_name == "McCallum" else sampling.generate_quantiles(da)
        )
    # stack rasters along the 'model' dimension
    terciles = xr.concat(terciles.values(), dim="model").assign_coords(
        model=MODEL_NAMES
    )

    ############################################################################################
    #### Determine wealth class in overlapping pixels by majority vote and generate maps
    ############################################################################################
    print("Calculating unanimous vote ensemble")
    # determine unanimous class label (i.e., pixel value with unanimous vote of models)
    ma.unanimous_mode(terciles).rio.to_raster(
        os.path.join(
            INTERIM_DIR, "raster_stacks/unanimous_ensemble", f"{country}_unsmbl.tif"
        )
    )
print("All countries completed.")

####################################################################################
#### Calculate summary statistics for majority-vote ensemble by country
####################################################################################
# merge all the country ensemble maps into a single raster
raster_dir = glob(
    os.path.join(INTERIM_DIR, "raster_stacks/unanimous_ensemble", "*_unsmbl.tif")
)
rasters = merge_arrays(
    [(rxr.open_rasterio(raster, masked=True).squeeze()) for raster in raster_dir]
)
rasters.rio.to_raster(os.path.join(PROCESSED_DIR, "unanimous_ensemble_map.tif"))
print("Unanimous-vote ensemble map completed.")

print("Calculating summary statistics for the unanimous-vote ensemble")

# calculate pixel stats for each class in the unanimous ensembles by country
stats = pd.DataFrame()
rasters = glob(
    os.path.join(INTERIM_DIR, "raster_stacks", "unanimous_ensemble", "*.tif")
)
for raster in raster_dir:
    country = os.path.basename(raster).split("_")[0]
    if country not in countries.keys():
        print("Summarising overall statistics")
    print(f"Summarising statistics for {country}")

    raster = rxr.open_rasterio(raster, masked=True).squeeze()
    # calculate pixel stats for each class
    freq_table = ma.frequency_table(
        raster, classes={1: "Poor", 2: "Average", 3: "Richer"}
    )
    freq_table.loc[:, "Country"] = country if country in countries.keys() else "Overall"
    freq_table = (
        freq_table.pivot(index="Country", columns="value", values="proportion")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    stats = pd.concat([stats, freq_table])
stats = (
    stats[["Country", "Poor", "Average", "Richer"]]
    .fillna(0)
    .sort_values(by="Country")
    .reset_index(drop=True)
)
stats.to_csv(
    os.path.join(PROCESSED_DIR, "unanimous_pixel_stats.csv"),
    index=False,
)
