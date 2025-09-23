import os

from config import INTERIM_DIR

# custom imports
from modules import sampling

countries = sampling.countries

# functions for generating raster maps for each model
model_list = {
    "Chi": sampling.chi_model,
    "Lee": sampling.lee_model,
    "McCallum": sampling.mccallum_model,
    "Yeh": sampling.yeh_model,
}

#########################################################################################
#### Generate raster maps for each model from raw wealth indices and save as geotiffs
#########################################################################################
for name, model in model_list.items():
    print(f"Rasterizing {name} model")
    print("*" * 50)

    for country in countries.keys():
        print(f"Generating map for {country}")
        model(country).rio.to_raster(
            os.path.join(INTERIM_DIR, "rasterized", f"{name}_{country}.tif"),
            compress="lzw",
        )

    print("*" * 50)
print("Raster maps completed.")
