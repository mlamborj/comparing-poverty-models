import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr

from config import INTERIM_DIR, MODEL_PAIRS, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in the model pair and calculate terciles for each model
############################################################################################
for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution (1.6 km)
    rasters = sampling.spatial_alignment(
        country, raster_dir=os.path.join(INTERIM_DIR, "rasterized")
    )
    for pair in MODEL_PAIRS:
        print(f"Processing {pair} pair")
        print("*" * 30)
        x, y = pair.split("_")
        raster_pair = rasters.sel(model=[x, y])
        # generate mask showing which pixels to include in the analysis (i.e., at least 2 models overlapping)
        mask = sampling.coincident_pixels(raster_pair, unanimous_only=True)
        # calculate terciles for each model
        terciles = dict()
        for model_name in [x, y]:
            print(f"Generating wealth terciles for {model_name}")
            # mask each model's data with the country mask
            da = (
                raster_pair.sel(model=model_name)
                .where(mask.notnull())
                .squeeze()
                .drop("model")
            )
            # calculate terciles, ignoring McCallum's model
            terciles[model_name] = (
                da if model_name == "McCallum" else sampling.generate_quantiles(da)
            )
        # stack rasters along the 'model' dimension
        terciles = xr.concat(terciles.values(), dim="model").assign_coords(model=[x, y])
        ##############################################################################################
        ### Determine spatial agreement for model pair
        ##############################################################################################
        ma.pairwise_agreement(terciles).rio.to_raster(
            os.path.join(
                INTERIM_DIR,
                "raster_stacks/pairwise_agreement",
                f"{pair}_{country}_models.tif",
            )
        )
print("All countries completed.")

####################################################################################
#### Calculate summary statistics for pairwise agreement by country
####################################################################################
print("Calculating summary statistics for the pairwise agreement")
dff = pd.DataFrame()
for pair in MODEL_PAIRS:
    rasters = glob(
        os.path.join(INTERIM_DIR, "raster_stacks/pairwise_agreement", f"{pair}_*.tif")
    )
    rasters = sorted(rasters)
    print(f"Processing {pair}")

    proportions = pd.DataFrame()
    for raster in rasters:
        country = os.path.basename(raster).split("_")[2]
        print(f"Calculating pixel proportions for {country}")

        da = rxr.open_rasterio(raster, masked=True).squeeze()
        freq_table, N = ma.frequency_table(da)
        freq_table.loc[:, "Country"] = country
        freq_table = (
            freq_table.pivot(index="Country", columns="value", values="proportion")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        freq_table["N"] = N
        proportions = pd.concat([proportions, freq_table])
    proportions["model_pair"] = pair
    proportions["agree"] = 100 - proportions[0]
    dff = pd.concat([dff, proportions.reset_index(drop=True)])
dff = (
    dff[["Country", "model_pair", 0, "agree", 1, 2, 3, "N"]]
    .rename(
        columns={0: "disagree (0)", 1: "poor (1)", 2: "average (2)", 3: "richer (3)"}
    )
    .fillna(0)
    .reset_index(drop=True)
    .sort_values(by=["model_pair", "Country"])
)
dff.to_csv(
    os.path.join(PROCESSED_DIR, "Pairwise_wealth_classes.csv"),
    index=False,
)

dff = (
    dff.pivot_table(index="Country", columns="model_pair", values="agree")
    .reset_index()
    .sort_values(by="Country")
)
dff.to_csv(
    os.path.join(PROCESSED_DIR, "Pairwise_agreement_pix_stats.csv"),
    index=False,
)
