import numpy as np
import xarray as xr
import pandas as pd
from pandas.api.types import CategoricalDtype
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
                return max_count  # frequency of unanimous mode
        else:  # logic to handle majority vote ensemble
            if np.sum(counts == max_count) > 1:
                return np.nan  # no mode
            elif max_count == 1:
                return np.nan  # vals[np.argmax(counts)] # uncontested mode
            else:
                return vals[np.argmax(counts)]  # unanimous mode

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
        .rio.write_nodata(da.rio.nodata)
    )
    return mode


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
        if max_count == 4:
            return vals[np.argmax(counts)]  # unanimous mode
        else:
            return np.nan  # no unanimous mode

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
        .rio.write_nodata(da.rio.nodata)
    )
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
        .rio.write_nodata(da.rio.nodata)
    )
    return mode


def frequency_table(da: xr.DataArray, classes: dict = None) -> pd.DataFrame:
    """
    Function to generate raster frequency table. It summarises the unique pixel values
    and returns their proportions (percentage) in the raster, and the total of non-NaN pixels.

    Args:
        da (xr.DataArray): input raster to summarise.
        classes (dict, optional): dictionary of classes to retain in the frequency table.
            Keys are the original values in the raster, and values are the new values to retain,
            e.g., {1: 'poor', 2: 'average', 3: 'richer'}.
            If None, all unique values will be retained. Defaults to None.

    Returns:
        pd.DataFrame: frequency table
        int (optional): total number of non-NaN pixels in the raster.
    """
    # Extract numpy array from the DataArray
    ndarray = da.values
    # Ignore NaN values
    valid_values = ndarray[~np.isnan(ndarray)]
    vals, counts = np.unique(valid_values, return_counts=True)
    # Create a pandas DataFrame and calculate proportions
    df = pd.DataFrame({"value": vals, "count": counts})
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
    return df.sort_values(by="value")


def model_performance(model: str, model_dhs: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute the model performance metrics. Calculates accuracy, precision, recall, and F1 score
    of the model predictions against the DHS data. The metrics are computed for three strata: 'all', 'rural',
    and 'urban'.

    Args:
        model (str): Name of model for which performance is to be computed.
        model_dhs (pd.DataFrame): DataFrame containing DHS data and model predictions at corresponding pixels.

    Returns:
        pd.DataFrame: _description_
    """
    metrics_df = pd.DataFrame(
        columns=["model", "stratum", "accuracy", "precision", "recall", "f1"]
    )

    for i, stratum in enumerate(["all", "rural", "urban"]):
        df = model_dhs[model_dhs["smod"] == i] if stratum != "all" else model_dhs

        metrics_df.loc[-1, ["model", "stratum"]] = model, stratum
        metrics_df.loc[-1, "accuracy"] = accuracy_score(df["DHS"], df[model])
        metrics_df.loc[-1, "precision"] = precision_score(
            df["DHS"], df[model], average="weighted"
        )
        metrics_df.loc[-1, "recall"] = recall_score(
            df["DHS"], df[model], average="weighted"
        )
        metrics_df.loc[-1, "f1"] = f1_score(df["DHS"], df[model], average="weighted")
        metrics_df.index += 1
    return metrics_df.reset_index(drop=True)


if __name__ == "__main__":
    # overall_agreement().to_csv(os.path.join(outputs, 'disaggregated_analysis', 'overall_agreement.csv'), index=False)
    # agreement_by_country().to_csv(os.path.join(outputs, 'disaggregated_analysis', 'country_agreements.csv'), index=False)

    # print(model_agreement_f1(df1, 'McCallum_Chi'))
    print("hello")
