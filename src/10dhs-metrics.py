import os

import pandas as pd
import xarray as xr
import rioxarray as rxr

from config import SMOD_FILE, INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling

countries = sampling.countries

############################################################################################
#### Calculate model performance metrics against DHS data
############################################################################################
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/quintiles/unpooled")

# load SMOD raster for urban/rural classification
smod = rxr.open_rasterio(SMOD_FILE, masked=True).squeeze()
corr_stats = pd.DataFrame()
class_stats = pd.DataFrame()

for model in MODEL_NAMES:
    print(f"Processing performance metrics for {model}:")
    print("*" * 50)
    # load DHS raster for the model
    dhs = rxr.open_rasterio(
        os.path.join(INTERIM_DIR, "rasterized", f"DHS_{model}.tif"), masked=True
    ).squeeze()

    for country in countries.keys():
        print(f"Processing {country}:")

        rasters = []
        aligned = sampling.align_dhs(dhs, smod=smod, model_name=model, country=country)

        # raw values dataframe for DHS and model
        df_1 = (
            aligned.to_dataset(dim="model")
            .to_dataframe()
            .dropna()
            .reset_index(drop=True)
        )[[model, "DHS", "smod"]]

        # generate mask showing which pixels to include in the analysis (i.e., all models overlapping)
        mask = sampling.coincident_pixels(aligned, unanimous_only=True)

        # calculate quantiles, ignoring McCallum's model
        for name in aligned.model.values:
            ras = aligned.sel(model=name).drop("model")
            if name in ["McCallum", "smod"]:
                rasters.append(ras)
            else:
                # rasters.append(sampling.generate_weighted_quantiles(ras, country, q=3))
                rasters.append(sampling.generate_quantiles(ras, q=5))
        da = (
            xr.concat(rasters, dim="model")
            .assign_coords(model=[name for name in aligned.model.values])
            .where(mask.notnull())
        )

        # quantiles dataframe for DHS and model
        df_2 = (
            da.to_dataset(dim="model").to_dataframe().dropna().reset_index(drop=True)
        )[[model, "DHS", "smod"]]

        # calculate model correlation with DHS
        coeff_df = ma.model_correlation(df_1).reset_index(drop=True)
        coeff_df.loc[:, ["Country", "Model"]] = (
            country,
            model,
        )
        corr_stats = pd.concat([corr_stats, coeff_df])

        # calculate model performance metrics
        perf_df = ma.model_performance(df_2).reset_index(drop=True)
        perf_df.loc[:, ["Country", "Model"]] = (
            country,
            model,
        )
        class_stats = pd.concat([class_stats, perf_df])

        print("*" * 30)

# reshaping stats dataframe
stats = pd.merge(
    corr_stats.drop(columns="p value"),
    class_stats,
    on=["Country", "Model", "Cluster"],
    how="inner",
    validate="1:1",
)
stats = pd.melt(stats, id_vars=["Country", "Model", "Cluster"], var_name="metric")
stats = (
    pd.pivot_table(
        stats, index=["Country", "Cluster", "metric"], columns="Model", values="value"
    )
    .reset_index()
    .rename_axis(None, axis=1)
)

stats.to_csv(
    os.path.join(out_path, "DHS_metrics.csv"),
    index=False,
)

print("All countries completed.")
