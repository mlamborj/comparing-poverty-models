import os
import warnings

from config import INTERIM_DIR, MODEL_NAMES

# custom imports
from modules import sampling, utils

# Ignore all user warnings
warnings.simplefilter(action="ignore", category=UserWarning)

countries = utils.countries["name"].to_list()
# valid_countries = sampling.valid_countries

# functions for generating raster maps for each model
model_list = {
    # "Chi": sampling.chi_model,
    # "Lee": sampling.lee_model,
    # # "McCallum": sampling.mccallum_model,
    # "Yeh": sampling.yeh_model,
    # "DHS_contemporary": sampling.dhs_model_contemporary,
    "DHS_contemporary": sampling.dhs_model,
    "DHS_latest": sampling.dhs_model,
}

#########################################################################################
#### Generate raster maps for each model from raw wealth indices and save as geotiffs
#########################################################################################
for name, model in model_list.items():
    print("*" * 50)
    print(f"Generating maps for {name}")
    print("*" * 50)

    if name == "DHS_contemporary":
        # DHS model differs based on the model and country being processed
        for model_name in MODEL_NAMES + ["Ensemble"]:
            print(f"{model_name}")

            model(model_name).rio.to_raster(
                os.path.join(INTERIM_DIR, "rasterized", "DHS", f"DHS_{model_name}.tif"),
                compress="lzw",
            )
    elif name == "DHS_latest":
        # # Only 15 countries have DHS data that does not overlap with the training data of the models
        # for country in valid_countries:
        #     print(f"Generating DHS map for {country}")
        #     model(country, train_overlap=False).rio.to_raster(
        #         os.path.join(INTERIM_DIR, "rasterized", f"DHS_{country}_latest.tif"),
        #         compress="lzw",
        #     )
        print(f"{name}")

        model().rio.to_raster(
            os.path.join(INTERIM_DIR, "rasterized", "DHS", f"{name}.tif"),
            compress="lzw",
        )
    else:
        for country in countries.keys():
            print(f"{country}")

            model(country).rio.to_raster(
                os.path.join(INTERIM_DIR, "rasterized", f"{name}_{country}.tif"),
                compress="lzw",
            )

    print("*" * 50)

print("Raster maps completed.")
