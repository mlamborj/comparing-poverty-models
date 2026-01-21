import os

import pandas as pd
import xarray as xr

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in at least 2 models and calculate quantiles for each model
############################################################################################
# define model combinations for leave-one-out ensembles
combinations = {
    "CLM": "Chi_Lee_McCallum",
    "CLY": "Chi_Lee_Yeh",
    "CMY": "Chi_McCallum_Yeh",
    "LMY": "Lee_McCallum_Yeh",
}

for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution
    rasters = sampling.spatial_alignment(country)
    # generate mask showing which pixels to include in the analysis (i.e., at least 2 models overlapping)
    mask = sampling.coincident_pixels(rasters)
    # calculate quantiles for each model
    quantiles = dict()
    for model_name in MODEL_NAMES:
        print(f"Generating wealth quantiles for {model_name}")

        da = rasters.sel(model=model_name).squeeze().drop("model")
        # calculate quantiles, ignoring McCallum's model
        # quantiles[model_name] = (
        #     da if model_name == "McCallum" else sampling.generate_quantiles(da, q=3)
        # )
        quantiles[model_name] = (
            da
            if model_name == "McCallum"
            else sampling.generate_weighted_quantiles(da, country, q=3)
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
    for k, v in combinations.items():
        print(f"Calculating majority vote ensemble for {v}")
        models_in_combo = v.split("_")
        # determine majority class label (i.e., pixel value with majority vote of models)
        ma.calculate_mode(
            quantiles.sel(model=models_in_combo), return_freq=False
        ).rio.to_raster(
            os.path.join(
                INTERIM_DIR,
                "raster_stacks/leave-one-out",
                f"{country}_{k}_ensemble.tif",
            )
        )
print("All countries completed.")

############################################################################################
#### Calculate model performance metrics against DHS data
############################################################################################
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/terciles/unpooled")
class_stats = pd.DataFrame()

for country in countries.keys():
    print(f"Processing {country}:")
    print("*" * 50)

    for k, v in combinations.items():
        print(f"Processing stats for {v} ensemble")
        # align rasters to each other and resample to the same resolution
        rasters = sampling.spatial_alignment(
            country, model_list=["DHS", f"{k}_ensemble"]
        )
        # generate mask showing which pixels to include in the analysis (i.e., only where the 2 models overlap)
        mask = sampling.coincident_pixels(rasters, unanimous_only=True)
        # calculate quantiles for DHS model
        quantiles = dict()

        for model_name in ["DHS", f"{k}_ensemble"]:
            da = rasters.sel(model=model_name).squeeze().drop("model")
            # calculate quantiles, ignoring Ensemble model
            quantiles[model_name] = (
                da
                if model_name.endswith("ensemble")
                else sampling.generate_weighted_quantiles(da, country, q=3)
            )
        # stack rasters along the 'model' dimension
        quantiles = (
            xr.concat(quantiles.values(), dim="model")
            .assign_coords(model=["DHS", f"{k}"])
            .where(mask.notnull())  # mask each model's data with the country mask
        )

        quantiles = quantiles.sel(model=["DHS", f"{k}"]).to_dataset(dim="model")
        # get urbanisation class and prepare df
        df = (
            utils.urbanisation_class(quantiles, country=country)
            .to_dataframe()
            .dropna()
            .reset_index(drop=True)
        )[["DHS", f"{k}", "smod"]]

        # calculate model performance metrics
        perf_df = ma.model_performance(df).reset_index(drop=True)
        perf_df.loc[:, ["Country", "Model"]] = (
            country,
            f"{k}",
        )
        class_stats = pd.concat([class_stats, perf_df])

        print("*" * 30)

stats = pd.melt(class_stats, id_vars=["Country", "Model", "Cluster"], var_name="metric")
stats = (
    pd.pivot_table(
        stats, index=["Country", "Cluster", "metric"], columns="Model", values="value"
    )
    .reset_index()
    .rename_axis(None, axis=1)
)

# combine full ensemble metrics
base_models = pd.read_csv(os.path.join(PROCESSED_DIR, out_path, "DHS_metrics.csv"))[
    ["Country", "Cluster", "metric", "Ensemble"]
]
stats = pd.merge(
    base_models[base_models["metric"] != "correlation"],
    stats,
    on=["Country", "Cluster", "metric"],
    how="left",
    validate="1:1",
)
stats.to_csv(
    os.path.join(out_path, "Ensemble_metrics.csv"),
    index=False,
)

print("All countries completed.")
