import os

import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.merge import merge as merge_arrays

from config import SMOD_FILE, PROCESSED_DIR, EXTERNAL_DIR

# custom imports
from modules import sampling

countries = sampling.countries

############################################################################################
#### Summarise population by model agreement and urbanisation class
############################################################################################

# read spatial agreement map
ds = (
    rxr.open_rasterio(
        os.path.join(
            PROCESSED_DIR,
            "pixel-wise/terciles/unpooled/majority",
            "spatial_agreement_map.tif",
        ),
        masked=True,
    )
    .squeeze()
    .to_dataset(name="agreement")
)

# merge population density rasters for countries into single raster
popn = merge_arrays(
    [
        rxr.open_rasterio(
            f"{EXTERNAL_DIR}/population/{country.lower()}_ppp_2020_1km_Aggregated.tif",
            masked=True,
        ).squeeze()
        for country in countries.values()
    ]
)
# align population raster to agreement map
ds["popn"] = popn.rio.reproject_match(
    ds["agreement"], resampling=Resampling.bilinear
).where(ds["agreement"].notnull())
# read urbanisation raster
ds["smod"] = (
    rxr.open_rasterio(SMOD_FILE, masked=True)
    .squeeze()
    .rio.reproject_match(ds["agreement"], resampling=Resampling.nearest)
    .where(ds["agreement"].notnull())
)
# export values to dataframe
df = ds.to_dataframe().dropna().reset_index(drop=True)[["agreement", "popn", "smod"]]

stats = pd.DataFrame()
# summarise population by model agreement and urbanisation
for i, cluster in enumerate([None, "rural", "urban"]):
    data = df[df["smod"] == i] if cluster else df.copy()
    summary = data["popn"].groupby(by=[data["agreement"]]).sum().reset_index()
    summary["urban_class"] = cluster if cluster else "all"
    stats = pd.concat([stats, summary], ignore_index=True)
# calculate percentage population (within each urban_class)
stats["pct_popn"] = (
    stats["popn"] / stats.groupby("urban_class")["popn"].transform("sum")
) * 100

stats["pct_popn"] = stats["pct_popn"].round(1)
