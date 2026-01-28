import os

import pandas as pd

from config import MODEL_PAIRS, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling, utils

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in the model pair and calculate model correlation
############################################################################################
out_path = os.path.join(PROCESSED_DIR, "pixel-wise/terciles/unpooled")
stats = pd.DataFrame()

for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)
    # align rasters to each other and resample to the same resolution
    rasters = sampling.spatial_alignment(country)

    for pair in MODEL_PAIRS:
        x, y = pair.split("_")
        ds = rasters.sel(model=[x, y]).to_dataset(dim="model")
        # get urbanisation class and prepare df
        df = (
            utils.urbanisation_class(ds, country=country)
            .to_dataframe()
            .dropna()
            .reset_index(drop=True)
        )[[x, y, "smod"]]
        # df.loc[:, ["Country", "Models"]] = (
        #     country,
        #     pair,
        # )  # todo scatter plot to visualise correlation

        print(f"Model correlation for {pair}")

        coeff_df = ma.model_correlation(df).reset_index(drop=True)
        coeff_df.loc[:, ["Country", "Models"]] = (
            country,
            pair,
        )
        stats = pd.concat([stats, coeff_df])

        print("*" * 30)

# merge wealth class info
wealth_classes = pd.read_csv(os.path.join(out_path, "Pairwise_wealth_classes.csv"))
wealth_classes = pd.merge(
    stats[stats["Cluster"] == "all"][["Country", "Models", "correlation"]],
    wealth_classes,
    left_on=["Country", "Models"],
    right_on=["Country", "model_pair"],
    validate="1:1",
).drop(columns=["model_pair"])

wealth_classes.to_csv(
    os.path.join(out_path, "Pairwise_wealth_classes.csv"),
    index=False,
)

stats = stats.pivot_table(
    index=["Country", "Cluster"], columns="Models", values="correlation"
).reset_index()

stats.to_csv(
    os.path.join(out_path, "Pairwise_correlation.csv"),
    index=False,
)

print("All countries completed.")
