import os

import pandas as pd
import xarray as xr

from config import PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Calculate model performance metrics against DHS data
############################################################################################
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/quintiles/unpooled")

# corr_stats = pd.DataFrame()
class_stats = pd.DataFrame()

model_list = ["DHS", "Ensemble"]

for country in countries.keys():
    print(f"Processing {country}:")
    print("*" * 50)

    # align rasters to each other and resample to the same resolution
    rasters = sampling.spatial_alignment(country, model_list=model_list)
    # generate mask showing which pixels to include in the analysis (i.e., only where the 2 models overlap)
    mask = sampling.coincident_pixels(rasters, unanimous_only=True)
    # calculate quantiles for DHS model
    quantiles = dict()
    for model_name in model_list:
        print(f"Generating wealth quantiles for {model_name}")

        da = rasters.sel(model=model_name).squeeze().drop("model")
        # calculate quantiles, ignoring Ensemble model
        quantiles[model_name] = (
            da if model_name == "Ensemble" else sampling.generate_quantiles(da, q=5)
        )
    # stack rasters along the 'model' dimension
    quantiles = (
        xr.concat(quantiles.values(), dim="model")
        .assign_coords(model=model_list)
        .where(mask.notnull())  # mask each model's data with the country mask
    )

    quantiles = quantiles.sel(model=["DHS", "Ensemble"]).to_dataset(dim="model")
    # get urbanisation class and prepare df
    df = (
        utils.urbanisation_class(quantiles, country=country)
        .to_dataframe()
        .dropna()
        .reset_index(drop=True)
    )[["DHS", "Ensemble", "smod"]]

    # calculate model performance metrics
    perf_df = ma.model_performance(df).reset_index(drop=True)
    perf_df.loc[:, ["Country", "Model"]] = (
        country,
        "Ensemble",
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

# combine with base model metrics
base_models_stats = pd.read_csv(
    os.path.join(PROCESSED_DIR, out_path, "DHS_metrics.csv")
)
stats = pd.merge(
    base_models_stats,
    stats,
    on=["Country", "Cluster", "metric"],
    how="left",
    validate="1:1",
)
stats.to_csv(
    os.path.join(out_path, "DHS_metrics.csv"),
    index=False,
)

print("All countries completed.")
