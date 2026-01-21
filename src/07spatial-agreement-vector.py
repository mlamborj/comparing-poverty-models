import os

import geopandas as gpd
import pandas as pd

from config import INTERIM_DIR, MODEL_NAMES, PROCESSED_DIR
from modules import model_agreement as ma

# custom imports
from modules import sampling

countries = sampling.countries

############################################################################################
#### Determine pixels overlapping in at least 2 models and calculate quantiles for each model
############################################################################################
for country in countries.keys():
    print(f"Processing {country}")
    print("*" * 50)

    # get admin-level indices
    vectors = gpd.read_file(
        os.path.join(INTERIM_DIR, "vectorized", "admin_indices.gpkg"),
        layer=f"{country}_models",
    )
    indices = [col for col in vectors.columns if col.endswith("index")]
    # # filter admin units to include in the analysis (i.e., at least 2 models present in a polygon)
    # vectors["models"] = vectors[indices].notna().sum(axis=1)
    # mask = vectors["models"] >= 2

    print("Generating wealth quantiles for all models")
    # # calculate quantiles at admin level for each model
    # vectors[MODEL_NAMES] = vectors[mask][indices].apply(
    #     lambda x: sampling.generate_quantiles_v(x, q=3), axis=0
    # )
    vectors[MODEL_NAMES] = vectors[indices].apply(
        lambda x: sampling.generate_quantiles_v(x, q=3), axis=0
    )

    ############################################################################################
    #### Determine wealth class in overlapping admin units by majority vote and generate maps
    ############################################################################################
    print("Calculating majority vote ensemble")
    # # determine majority class label (i.e., quantile value for majority of models)
    # vectors["majority"] = vectors[mask][MODEL_NAMES].apply(ma.calculate_mode_v, axis=1)
    vectors["majority"] = vectors[MODEL_NAMES].apply(ma.calculate_mode_v, axis=1)

    ############################################################################################
    #### Determine spatial agreement of quantiles in overlapping admins and export
    ############################################################################################
    print("Calculating spatial agreement\n")
    # # determine spatial agreement (i.e., no. of models in agreement per admin unit)
    # vectors["agreement"] = vectors[mask][MODEL_NAMES].apply(
    #     lambda x: ma.calculate_mode_v(x, return_freq=True), axis=1
    # )
    vectors["agreement"] = vectors[MODEL_NAMES].apply(
        lambda x: ma.calculate_mode_v(x, return_freq=True), axis=1
    )
    # export spatial agreement vector
    vectors.to_file(
        os.path.join(INTERIM_DIR, "model_agreement.gpkg"),
        layer=f"{country}_models",
        driver="GPKG",
        index=False,
    )
print("All countries completed.")

####################################################################################
#### Calculate summary statistics for majority-vote ensemble by country
####################################################################################
# merge all the country ensemble maps into a single dataframe
out_path = os.path.join(PROCESSED_DIR, "admin-2/terciles/unpooled")
vectors = pd.concat(
    [
        gpd.read_file(
            os.path.join(INTERIM_DIR, "model_agreement.gpkg"),
            layer=f"{country}_models",
        )
        for country in countries.keys()
    ]
)
vectors.to_file(os.path.join(out_path, "admin_ensembles.geojson"), driver="GeoJSON")
print("Majority-vote ensemble map completed.")

print("Calculating summary statistics")

stats = {"ensemble_stats": pd.DataFrame(), "agreement_stats": pd.DataFrame()}
country_list = list(countries.keys()) + ["overall"]

for k, df in stats.items():
    col = "majority" if k == "ensemble_stats" else "agreement"
    classes = (
        {1: "Poor", 2: "Average", 3: "Richer"}
        # {
        #     1: "Poorest",
        #     2: "Poorer",
        #     3: "Average",
        #     4: "Richer",
        #     5: "Richest",
        # }  # Extended to quintiles
        if k == "ensemble_stats"
        else {
            0: "No agreement",
            1: "Split agreement",
            2: "2 models agree",
            3: "3 models agree",
            4: "All models agree",
        }
    )

    for country in country_list:
        if country not in countries.keys():
            print("Summarising overall statistics")
            vector = vectors.copy()
        else:
            print(f"Summarising statistics for {country}")
            vector = vectors[vectors["country_name"] == country]

        # calculate admin stats for each class
        freq_table = ma.frequency_table(vector[col], classes=classes)
        freq_table.loc[:, "Country"] = (
            country if country in countries.keys() else "Overall"
        )
        freq_table = (
            freq_table.pivot(index="Country", columns="value", values="proportion")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        df = pd.concat([df, freq_table])

    columns = list(classes.values())
    columns.insert(0, "Country")

    df = df[columns].fillna(0).sort_values(by="Country").reset_index(drop=True)
    df.to_csv(os.path.join(out_path, f"{k}_admin_stats.csv"), index=False)
