from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import CategoricalDtype
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_mode(da: xr.DataArray, dim="model", return_freq=False) -> xr.DataArray:
    """
    This function is for generating the Spatial Agreement map based on majority vote.
    It compares pixel values from the 4 models and returns the number of models that
    agree on the same value (i.e., mode frequency). Comparison is only made if there
    are at least 2 non-NaN pixels, to determine a majority vote.

    Args:
        da (xr.DataArray): Stack of rasters to compare.
        dim (str, optional): Name of dimension along which rasters are stacked. Defaults to 'model'.
        return_freq (bool, optional): If True, returns the frequency of the mode, or the
            actual mode if False. Defaults to False.

    Returns:
        xr.DataArray: raster with the mode or mode frequency of the input rasters.
    """
    # Extract numpy array from the DataArray
    np_array = da.values

    # Function to determine mode or mode frequency
    def resolve_mode(ndarray):
        # Ignore NaN values
        valid_values = ndarray[~np.isnan(ndarray)]
        if len(valid_values) == 0:
            return np.nan
        vals, counts = np.unique(valid_values, return_counts=True)
        max_count = np.max(counts)
        # logic to handle mode frequency
        if return_freq:
            if max_count == 1 and np.sum(counts == max_count) > 1:
                return 0  # complete mismatch
            elif max_count == 1 and np.sum(counts == max_count) == 1:
                return np.nan  # uncontested mode frequency
            elif np.sum(counts == max_count) > 1:
                return 1  # tied mode
            else:
                return max_count  # frequency of true mode
        else:  # logic to handle majority vote ensemble
            if np.sum(counts == max_count) > 1:
                return np.nan  # no mode
            elif max_count == 1:
                return np.nan  # uncontested mode
            else:
                return vals[np.argmax(counts)]  # true mode

    # Apply function along the axis
    result = np.apply_along_axis(resolve_mode, axis=da.get_axis_num(dim), arr=np_array)
    # Convert the result back into an xarray DataArray
    mode = (
        xr.DataArray(
            data=result.astype(np.float32),
            dims=[d for d in da.dims if d != dim],
            coords={k: v for k, v in da.coords.items() if k != dim},
        )
        .rio.write_crs(da.rio.crs)
        .rio.write_nodata(np.nan)
    )
    mode = mode.where(mode >= 0)  # dealing with artifacts in Togo
    return mode


def calculate_mode_v(row: pd.Series, return_freq=False) -> float:
    """
    This function is the vector equivalent for generating the Spatial Agreement based on majority vote.
    It compares values from the 3 models and returns the number of models that agree on the same value.
    Comparison is only made if there are at least 2 non-NaN values, to determine a majority vote.

    Args:
        row (pd.Series): The input data for a single admin unit across models.
        return_freq (bool, optional): If True, returns the frequency of the mode, or the
            actual mode if False. Defaults to False.

    Returns:
        float: The mode or mode frequency of the input data.
    """
    # calculate mode ignoring NaN values
    mode = row.mode(dropna=True)
    if return_freq:
        # logic to handle mode frequency
        if len(mode) == 1:
            counts = row.value_counts()
            # frequency of unanimous mode
            return counts.loc[mode.iloc[0]] if row.notna().sum() >= 2 else np.nan
        elif len(mode) == 2:
            case = row.value_counts().iloc[0]
            # the 2 available models disagree or split agreement
            return 0 if case == 1 else 1.0
        else:
            # the 3 available models disagree
            return 0 if row.notna().sum() > 0 else np.nan
    else:
        # logic to handle majority vote ensemble
        if len(mode) == 1:
            return mode.iloc[0] if row.notna().sum() >= 2 else np.nan
        else:
            return np.nan


def unanimous_mode(da: xr.DataArray, dim="model") -> xr.DataArray:
    """
    This function is for generating the ensemble map based on unanimous vote.
    It compares pixel values from the 4 models and returns the class label only when all
     4 models agree (i.e., mode value). Comparison is only made if all 4 models are not NaN
     in the pixel.

    Args:
        da (xr.DataArray): Stack of rasters to compare.
        dim (str, optional): Name of dimension along which rasters are stacked. Defaults to 'model'.

    Returns:
        xr.DataArray: raster with the mode of the input rasters.
    """
    # Extract numpy array from the DataArray
    np_array = da.values

    # Function to determine unanimous mode
    def resolve_mode(ndarray):
        # Ignore NaN values
        valid_values = ndarray[~np.isnan(ndarray)]
        if len(valid_values) == 0:
            return np.nan
        vals, counts = np.unique(valid_values, return_counts=True)
        max_count = np.max(counts)
        # logic to handle mode
        if max_count == 3:  # change to number of models being compared
            return vals[np.argmax(counts)]  # unanimous mode
        else:
            return np.nan  # no unanimous mode

    # Apply function along the axis
    result = np.apply_along_axis(resolve_mode, axis=da.get_axis_num(dim), arr=np_array)
    # Convert the result back into an xarray DataArray
    mode = (
        xr.DataArray(
            data=result.astype(np.float32),
            dims=[d for d in da.dims if d != dim],
            coords={k: v for k, v in da.coords.items() if k != dim},
        )
        .rio.write_crs(da.rio.crs)
        .rio.write_nodata(np.nan)
    )
    mode = mode.where(mode >= 0)  # dealing with artifacts in Togo
    return mode


def pairwise_agreement(da: xr.DataArray, dim="model") -> xr.DataArray:
    """
    Function to compute pair-wise spatial agreement between two models.
    It compares pixel values from the 2 models and returns the class label only when both models
    agree (i.e., mode value). Comparison is only made if both models are not NaN in the pixel, otherwise
    it returns NaN.
    Args:
        da (xr.DataArray): Stack of two rasters to compare.
        dim (str, optional): Name of dimension along which rasters are stacked. Defaults to 'model'.

    Returns:
        xr.DataArray: raster with the mode of the input rasters.
    """
    # Extract numpy array from the DataArray
    np_array = da.values

    # Function to determine unanimous mode
    def resolve_mode(ndarray):
        # Ignore NaN values
        valid_values = ndarray[~np.isnan(ndarray)]
        if len(valid_values) == 0:
            return np.nan
        vals, counts = np.unique(valid_values, return_counts=True)
        max_count = np.max(counts)
        # logic to handle mode
        if max_count == 2:
            return vals[np.argmax(counts)]  # unanimous mode
        elif len(counts) == 1:
            return np.nan  # uncontested mode
        else:
            return 0  # disagreement

    # Apply function along the axis
    result = np.apply_along_axis(resolve_mode, axis=da.get_axis_num(dim), arr=np_array)
    # Convert the result back into an xarray DataArray
    mode = (
        xr.DataArray(
            data=result,
            dims=[d for d in da.dims if d != dim],
            coords={k: v for k, v in da.coords.items() if k != dim},
        )
        .rio.write_crs(da.rio.crs)
        .rio.write_nodata(np.nan)
    )
    return mode.astype(np.float32)


def model_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute Spearman's Rank correlation coefficients between model pairs by stratum.
    It compares pixel values from the 2 models and returns the Spearman's Rank correlation coefficient.
    Comparison is only made for overlapping pixels in the model pair.
    Args:
        df (pd.DataFrame): Pixel values for the model pair. Expects the first two columns to contain the
        pixel value pairs, and an additional column <smod> containing the urbanisation code (1: rural, 2: urban)

    Returns:
        pd.DataFrame: dataframe with the correlation coeffients and p-value for each settlement cluster
    """
    stats = pd.DataFrame()
    cols = ["Cluster", "coeff", "p value"]
    strata = {1: "rural", 2: "urban", None: "all"}
    # compute coefficients for all strata
    for k, v in strata.items():
        corr = pd.DataFrame(columns=cols)
        # filter by urbanisation
        dff = df[df["smod"] == k] if k else df
        corr.loc[0, cols] = v, *spearmanr(dff.iloc[:, 0], dff.iloc[:, 1])
        stats = pd.concat([stats, corr])

    stats[cols[1:]] = stats[cols[1:]].astype(np.float32).round(5)
    stats = stats.rename(columns={"coeff": "correlation"})
    return stats


def frequency_table(
    da: Union[xr.DataArray, pd.Series], classes: dict = None
) -> pd.DataFrame:
    """
    Function to generate frequency table. It summarises the unique values from the input raster or dataframe
    and returns their proportions (percentage), and the total of non-NaN pixels/polygons.

    Args:
        da (xr.DataArray or pd.Series): input data to summarise. Can be an xarray DataArray (raster) or pandas
        Series (i.e., column from a geopandas dataframe).
        classes (dict, optional): dictionary of classes to retain in the frequency table.
            Keys are the original values in the raster, and values are the new values to retain,
            e.g., {1: 'poor', 2: 'average', 3: 'richer'}.
            If None, all unique values will be retained. Defaults to None.

    Returns:
        pd.DataFrame: frequency table
        int (optional): total number of non-NaN pixels in the raster.
    """
    if isinstance(da, xr.DataArray):
        # Extract numpy array from the DataArray
        ndarray = da.values
        # Ignore NaN values
        valid_values = ndarray[~np.isnan(ndarray)]
        vals, counts = np.unique(valid_values, return_counts=True)
        # Create a pandas DataFrame and calculate proportions
        df = pd.DataFrame({"value": vals, "count": counts})
    else:
        # Extract value counts from the column
        counts = da.value_counts()
        # Create a pandas DataFrame and calculate proportions
        df = pd.DataFrame({"value": counts.index, "count": counts.values})
    # Calculate proportions
    df["proportion"] = 100 * df["count"] / df["count"].sum()
    df["proportion"] = df["proportion"].round(1)

    if not classes:
        # If no classes provided, return frequency table with counts (number of pixels)
        return df.drop(columns="count"), df["count"].sum()

    # Retain categories of interest
    df = df.drop(columns="count")
    df["value"] = df["value"].map(classes)
    # Make classes ordinal
    ordered_classes = CategoricalDtype(
        categories=sorted(classes.values(), reverse=True), ordered=True
    )
    df["value"] = df["value"].astype(ordered_classes)
    return df.sort_values(by="value").reset_index(drop=True)


def model_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute the model performance metrics. Calculates accuracy, precision, recall, and F1 score
    of the model predictions against the DHS data. The metrics are computed for three strata: 'all', 'rural',
    and 'urban'.

    Args:
        model_dhs (pd.DataFrame): DataFrame containing DHS data and model predictions at corresponding pixels.

    Returns:
        pd.DataFrame: dataframe with the performance metrics for each settlement cluster.
    """
    stats = pd.DataFrame()
    cols = ["Cluster", "accuracy", "precision", "recall", "f1"]
    strata = {1: "rural", 2: "urban", None: "all"}
    # functions to compute each metric
    metrics_fxns = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted"
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted"
        ),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    }
    # compute metrics for all strata
    for k, v in strata.items():
        metrics_df = pd.DataFrame(columns=cols)
        # filter by urbanisation
        dff = df[df["smod"] == k] if k else df
        for metric in cols[1:]:
            metrics_df.loc[-1, ["Cluster", metric]] = (
                v,
                metrics_fxns[metric](dff.iloc[:, 1], dff.iloc[:, 0]),
            )
        metrics_df.index += 1
        stats = pd.concat([stats, metrics_df])
    stats[cols[1:]] = stats[cols[1:]].astype(np.float32).round(5)
    return stats.reset_index(drop=True)


if __name__ == "__main__":
    # overall_agreement().to_csv(os.path.join(outputs, 'disaggregated_analysis', 'overall_agreement.csv'), index=False)
    # agreement_by_country().to_csv(os.path.join(outputs, 'disaggregated_analysis', 'country_agreements.csv'), index=False)

    # print(model_agreement_f1(df1, 'McCallum_Chi'))
    print("hello")
