import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr

from config import INTERIM_DIR, MODEL_PAIRS, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in the model pair and calculate quantiles for each model
############################################################################################
# todo parallelise this section
for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution (1.6 km)
    rasters = sampling.spatial_alignment(country)

    for pair in MODEL_PAIRS:
        print(f"Processing {pair} pair")
        print("*" * 30)
        x, y = pair.split("_")
        raster_pair = rasters.sel(model=[x, y])
        # generate mask showing which pixels to include in the analysis (i.e., only where the 2 models overlap)
        mask = sampling.coincident_pixels(raster_pair, unanimous_only=True)
        # calculate quantiles for each model
        quantiles = dict()
        for model_name in [x, y]:
            print(f"Generating wealth quantiles for {model_name}")
            # mask each model's data with the country mask
            da = raster_pair.sel(model=model_name).squeeze().drop("model")
            # calculate quantiles, ignoring McCallum's model
            quantiles[model_name] = (
                da if model_name == "McCallum" else sampling.generate_quantiles(da, q=5)
            )
        # stack rasters along the 'model' dimension
        quantiles = (
            xr.concat(quantiles.values(), dim="model")
            .assign_coords(model=[x, y])
            .where(mask.notnull())  # mask each model's data with the country mask
        )
        ##############################################################################################
        ### Determine spatial agreement for model pair
        ##############################################################################################
        ma.pairwise_agreement(quantiles).rio.to_raster(
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
strata = {1: "rural", 2: "urban", None: "all"}
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/quintiles/unpooled")
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

        da = (
            rxr.open_rasterio(raster, masked=True).squeeze().to_dataset(name="ensemble")
        )
        # get urbanisation raster
        da = utils.urbanisation_class(da, country=country)
        # calculate pixel stats for each class by urbanisation
        for cluster in strata.keys():
            # mask raster by urban/rural/all
            if cluster is not None:
                ras = da["ensemble"].where(da["smod"] == cluster).squeeze()
            else:
                ras = da["ensemble"].squeeze()

            freq_table, N = ma.frequency_table(ras)
            freq_table.loc[:, "Country"] = country
            freq_table = (
                freq_table.pivot(index="Country", columns="value", values="proportion")
                .reset_index()
                .rename_axis(None, axis=1)
            )
            freq_table["N"] = N
            freq_table.loc[:, "Cluster"] = strata[cluster]
            proportions = pd.concat([proportions, freq_table])
    proportions["model_pair"] = pair
    proportions["agree"] = 100 - proportions[0]
    dff = pd.concat([dff, proportions.reset_index(drop=True)])
dff = (
    # dff[["Country", "Cluster", "model_pair", 0, "agree", 1, 2, 3, "N"]]
    dff[["Country", "Cluster", "model_pair", 0, "agree", 1, 2, 3, 4, 5, "N"]]
    .rename(
        # columns={0: "disagree (0)", 1: "poor (1)", 2: "average (2)", 3: "richer (3)"}
        columns={
            0: "disagree (0)",
            1: "poorest (1)",
            2: "poorer (2)",
            3: "average (3)",
            4: "richer (4)",
            5: "richest (5)",
        }
    )
    .fillna(0)
    .reset_index(drop=True)
    .sort_values(by=["model_pair", "Country"])
)
dff.to_csv(
    os.path.join(out_path, "Pairwise_wealth_classes.csv"),
    index=False,
)

dff = (
    dff.pivot_table(index=["Country", "Cluster"], columns="model_pair", values="agree")
    .reset_index()
    .sort_values(by=["Country", "Cluster"])
)
dff.to_csv(
    os.path.join(out_path, "Pairwise_agreement_pixel_stats.csv"),
    index=False,
)
