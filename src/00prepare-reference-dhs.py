import os
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, RAW_DIR, MODEL_NAMES

# custom imports
from modules import utils

countries = utils.countries

dhs_data = Path(EXTERNAL_DIR) / "DHS"
out_dir = Path(RAW_DIR) / "DHS"


def extract_date(file_path):
    filename = file_path.name
    return int(filename[3:7])  # survey year


def prepare_dhs_data(zipfile_name: str) -> pd.DataFrame:
    with ZipFile(zipfile_name) as zf:
        for file in zf.namelist():
            if file.endswith((".dta", ".DTA")):
                with zf.open(file) as f:
                    hr_df = pd.read_stata(f, convert_categoricals=False)[
                        ["hv001", "hv271"]
                    ].rename(columns={"hv001": "cluster_id", "hv271": "wealth_index"})
            if file.endswith((".shp", ".SHP")):
                ge_path = f"zip://{zipfile_name}!{file}"
                gdf = gpd.read_file(ge_path)[
                    ["DHSCC", "DHSCLUST", "DHSYEAR", "URBAN_RURA", "LATNUM", "LONGNUM"]
                ].rename(
                    columns={"DHSCLUST": "cluster_id", "URBAN_RURA": "urban_rural"}
                )
    # scale wealth index back to original scale and average by cluster
    hr_df["wealth_index"] = hr_df["wealth_index"] / 1e5
    hr_df = hr_df.groupby(by="cluster_id", as_index=False)["wealth_index"].mean()
    # join cluster coordinates
    gdf["urban_rural"] = (
        gdf.loc[:, "urban_rural"].map({"R": 1, "U": 2}).astype(np.uint8)
    )
    dhs = pd.merge(hr_df, gdf, on="cluster_id", validate="1:1")
    return dhs


#########################################################################################
#### Compile latest DHS data in each country and save as CSV
#########################################################################################

print("*" * 50)
print("Compiling latest DHS data")
print("*" * 50)

surveys = []

for _, country in countries.iterrows():
    # todo: change between min and max
    # # find the latest DHS data file for the country
    # latest_file = min(
    #     Path(os.path.join(dhs_data, country['name'])).glob(f"{country['alpha-2']}_*.zip"),
    #     key=extract_date_from_path,
    # )
    # find the latest DHS data file for the country
    latest_file = next(
        (
            f
            for f in Path(os.path.join(dhs_data, country["name"])).glob(
                f"{country['alpha-2']}_*.zip"
            )
            if extract_date(f) == country["latest_survey"]
        ),
        None,
    )

    print(f"{country['name']} : {latest_file.name}")

    dhs = prepare_dhs_data(latest_file)
    surveys.append(dhs)
# export combined DHS data to CSV
pd.concat(surveys).to_csv(out_dir / "latest" / "DHS_latest.csv", index=False)

#########################################################################################
#### Compile contemporary DHS data for each model and save as CSV
#########################################################################################

print("\n")
print("*" * 50)
print("Compiling contemporary DHS data")
print("*" * 50)

surveys = []

for model in MODEL_NAMES + ["Ensemble"]:
    print(f"\nProcessing {model}:")
    print("*" * 20)
    for _, country in countries.iterrows():
        # todo: remove this when the Yeh model is updated with Senegal data
        # skip Senegal for the Yeh model since it is not available
        if model in ["Chi", "Yeh"] and country["name"] == "Senegal":
            pass
        else:
            # find the latest DHS data contemporary with the model for the country
            contemporary_file = next(
                (
                    f
                    for f in Path(os.path.join(dhs_data, country["name"])).glob(
                        f"{country['alpha-2']}_*.zip"
                    )
                    if extract_date(f) == country[model]
                ),
                None,
            )

            print(f"{country['name']} : {contemporary_file.name}")

            dhs = prepare_dhs_data(contemporary_file)
            surveys.append(dhs)
    # export combined DHS data to CSV
    pd.concat(surveys).to_csv(
        out_dir / "contemporary" / f"DHS_{model}.csv", index=False
    )
