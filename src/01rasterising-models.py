import os

from config import INTERIM_DIR, MODEL_NAMES

# custom imports
from modules import sampling

countries = sampling.countries

# functions for generating raster maps for each model
model_list = {
    "Chi": sampling.chi_model,
    "Lee": sampling.lee_model,
    "McCallum": sampling.mccallum_model,
    "Yeh": sampling.yeh_model,
    "DHS_contemporary": sampling.dhs_model_contemporary,
    "DHS": sampling.dhs_model_latest,
}

#########################################################################################
#### Generate raster maps for each model from raw wealth indices and save as geotiffs
#########################################################################################
for name, model in model_list.items():
    print(f"Rasterizing {name} model")
    print("*" * 50)

    if name == "DHS_contemporary":
        # DHS model differs based on the model and country being processed
        for model_name in MODEL_NAMES:
            print(f"Generating DHS map for {model_name}")
            model(model_name).rio.to_raster(
                os.path.join(INTERIM_DIR, "rasterized", f"DHS_{model_name}.tif"),
                compress="lzw",
            )
    else:
        for country in countries.keys():
            print(f"Generating map for {country}")
            model(country).rio.to_raster(
                os.path.join(INTERIM_DIR, "rasterized", f"{name}_{country}.tif"),
                compress="lzw",
            )

    print("*" * 50)

print("Raster maps completed.")
