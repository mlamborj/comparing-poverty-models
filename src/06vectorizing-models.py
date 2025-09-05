import os

import rioxarray as rxr

from config import INTERIM_DIR, MODEL_NAMES

# custom imports
from modules import sampling, utils

countries = sampling.countries


##########################################################################################
#### Generate vector layers for each country with aggregated wealth indices for each model
##########################################################################################
for country in countries.keys():
    print(f"Aggregating indices for {country}")
    print("*" * 50)

    # read admin polygons
    gdf = utils.read_boundary(country, admin_level=2)[
        ["GID_2", "country_name", "NAME_2", "geometry"]
    ]
    # todo add SMOD classification for each admin (do we weight as well???)
    for model in MODEL_NAMES:
        print(f"Vectorizing {model} model")
        # get the rasterized model
        raster = rxr.open_rasterio(
            os.path.join(INTERIM_DIR, "rasterized", f"{model}_{country}.tif"),
            masked=True,
        ).squeeze()
        # apply weighted aggregation and export layer
        gdf = sampling.weighted_aggregation(raster, gdf, model, country)
        gdf.to_file(
            os.path.join(INTERIM_DIR, "vectorized", "admin_indices.gpkg"),
            layer=f"{country}_models",
        )
    print("*" * 50)
print("Admin-2 maps completed.")
