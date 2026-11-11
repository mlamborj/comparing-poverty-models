import os
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, RAW_DIR
from modules.international_wealth_index import build_iwi

# custom imports
from modules import utils

countries = utils.countries

dhs_data = Path(EXTERNAL_DIR) / "DHS"
out_dir = Path(RAW_DIR) / "DHS"


def extract_date(file_path):
    filename = file_path.name
    return int(filename[3:7])  # survey year


def prepare_dhs_rwi(zipfile_name: str) -> pd.DataFrame:
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
        gdf.loc[:, "urban_rural"].map({"R": 0, "U": 1}).astype(np.uint8)
    )
    dhs = pd.merge(hr_df, gdf, on="cluster_id", validate="1:1")
    return dhs


def prepare_dhs_iwi(zipfile_name: str) -> pd.DataFrame:
    with ZipFile(zipfile_name) as zf:
        for file in zf.namelist():
            if file.endswith((".dta", ".DTA")):
                with zf.open(file) as f:
                    hr_df = pd.read_stata(f, convert_categoricals=False)
                    # compute iwi
                    iwi_df = build_iwi(
                        hr_df,
                        missing="propagate",  # households with >3 missing items get NaN iwi
                        # phone_columns=["hv221"],
                        decimals=5,
                    )
                    # merge back to hr_df
                    hr_df = pd.merge(
                        hr_df,
                        iwi_df[["hv001", "hv002", "iwi"]],
                        on=["hv001", "hv002"],
                        validate="1:1",
                    )[["hv001", "iwi"]].rename(columns={"hv001": "cluster_id"})
            if file.endswith((".shp", ".SHP")):
                ge_path = f"zip://{zipfile_name}!{file}"
                gdf = gpd.read_file(ge_path)[
                    ["DHSCC", "DHSCLUST", "DHSYEAR", "URBAN_RURA", "LATNUM", "LONGNUM"]
                ].rename(
                    columns={"DHSCLUST": "cluster_id", "URBAN_RURA": "urban_rural"}
                )
    # average iwi by cluster
    hr_df = hr_df.groupby(by="cluster_id", as_index=False)["iwi"].mean()
    # join cluster coordinates
    gdf["urban_rural"] = (
        gdf.loc[:, "urban_rural"].map({"R": 0, "U": 1}).astype(np.uint8)
    )
    dhs = pd.merge(hr_df, gdf, on="cluster_id", validate="1:1")
    return dhs


#########################################################################################
#### Compile relative wealth index in each country and save as CSV
#########################################################################################

print("*" * 50)
print("Compiling DHS relative wealth index")
print("*" * 50)

rwi_surveys = []

# rwi is the unmodified DHS index in the original scale, which is used to train Chi's model
for _, country in countries.iterrows():
    # todo: remove this when the Yeh model is updated with Senegal data
    # skip Senegal for the Yeh model since it is not available
    if country["name"] == "Senegal":
        pass
    else:
        # find the DHS data file for the country
        dhs_file = next(
            (
                f
                for f in Path(os.path.join(dhs_data, country["name"])).glob(
                    f"{country['alpha-2']}_*.zip"
                )
                if extract_date(f) == country["common_survey"]
            ),
            None,
        )

        print(f"{country['name']} : {dhs_file.name}")

        rwi = prepare_dhs_rwi(dhs_file)
        rwi.loc[:, "country"] = country["name"].lower()
        rwi_surveys.append(rwi)
# export combined DHS data to CSV
pd.concat(rwi_surveys).to_csv(out_dir / "common" / "DHS_RWI.csv", index=False)

#########################################################################################
#### Compile international wealth index by Lee in each country and save as CSV
#########################################################################################

print("*" * 50)
print("Compiling DHS international wealth index")
print("*" * 50)

iwi_surveys = []

# iwi was used to train Lee's model, and is computed here from DHS variables
for _, country in countries.iterrows():
    # todo: remove this when the Yeh model is updated with Senegal data
    # skip Senegal for the Yeh model since it is not available
    if country["name"] == "Senegal":
        pass
    else:
        # find the DHS data file for the country
        dhs_file = next(
            (
                f
                for f in Path(os.path.join(dhs_data, country["name"])).glob(
                    f"{country['alpha-2']}_*.zip"
                )
                if extract_date(f) == country["common_survey"]
            ),
            None,
        )

        print(f"{country['name']} : {dhs_file.name}")

        iwi = prepare_dhs_iwi(dhs_file)
        iwi.loc[:, "country"] = country["name"].lower()
        iwi_surveys.append(iwi)
# export combined DHS data to CSV
pd.concat(iwi_surveys).to_csv(out_dir / "common" / "DHS_IWI.csv", index=False)

#########################################################################################
#### Compile harmonised asset wealth index by Yeh in each country and save as CSV
#########################################################################################

# `wealthpooled` is the one used to train Yeh's model
dhs_pooled = pd.read_csv(os.path.join(RAW_DIR, "Yeh", "dhs_wealth_index.csv"))
dhs_pooled = dhs_pooled.dropna(subset=["LATNUM", "LONGNUM"]).rename(
    columns={
        "cluster": "cluster_id",
        "year": "DHSYEAR",
        "cname": "DHSCC",
    }
)

hwi_surveys = []

# filter to only include survey years that are in the common survey years
for _, country in countries.iterrows():
    dhs = dhs_pooled[
        (dhs_pooled["country"] == country["name"])
        & (dhs_pooled["DHSYEAR"] == country["common_survey"])
    ]
    hwi_surveys.append(dhs)
# export combined DHS data to CSV
pd.concat(hwi_surveys).to_csv(out_dir / "common" / "DHS_HWI.csv", index=False)
