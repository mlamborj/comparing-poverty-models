import os
from glob import glob

import pandas as pd
import rioxarray as rxr
import xarray as xr

from config import INTERIM_DIR, MODEL_PAIRS, PROCESSED_DIR

# custom imports
from modules import model_agreement as ma
from modules.utils import countries

##############################################################################################
#### Generate pairwise agreement maps and save as geotiff raster files
##############################################################################################
# align model outputs for each country
for country in list(countries.keys()):  
    if country!='Togo':# need to sort out Chi_Yeh and Chi_Lee for Togo
        print(f"Creating pairwise maps for {country}")
        for pair in MODEL_PAIRS:
            rasters = dict()
            print(f"Processing {pair} pair")
            x, y = pair.split("_")
            rasters[x] = (
                rxr.open_rasterio(
                    os.path.join(INTERIM_DIR, "model_maps", f"{x}_{country}.tif"),
                    masked=True,
                )
                .squeeze()
                .drop("band")
            )
            rasters[y] = (
                rxr.open_rasterio(
                    os.path.join(INTERIM_DIR, "model_maps", f"{y}_{country}.tif"),
                    masked=True,
                )
                .squeeze()
                .drop("band")
                .rio.reproject_match(rasters[x])
            )  # always align raster to match the finest raster, which is assumed to be the first in the pair
            rasters = xr.concat(list(rasters.values()), dim="model").assign_coords(
                model=[x, y]
            )

            # determine spatial agreement for pair
            ma.pairwise_agreement(rasters).rio.to_raster(
                os.path.join(
                    INTERIM_DIR,
                    "raster_stacks",
                    "pairwise_agreement",
                    f"{pair}_{country}_models.tif",
                )
            )
print("Pairwise model stacks completed.")

# calculate proportions of pixels in agreement by country
dff = pd.DataFrame()
for pair in MODEL_PAIRS:
    rasters = glob(
        os.path.join(
            INTERIM_DIR, "raster_stacks", "pairwise_agreement", f"{pair}_*.tif"
        )
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
    os.path.join(PROCESSED_DIR, "Pairwise_agreement_proportions.csv"),
    index=False,
)
