"""
Prepare DHS household assets and reproduce the fixed IWI SPSS 3.1 formulas.
This script is a Python translation of the original SPSS code, which is available at
https://globaldatalab.org/iwi/computing/

Quality mappings are read from the editable quality_recode.xlsx workbook.
Utensils are excluded unless explicitly enabled. No PCA, imputation,
rescaling, or clipping is applied. Missing assets
select their own published formula, rather than being treated as non-ownership.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR

DEFAULT_COLUMNS = {
    "water": "hv201",
    "toilet": "hv205",
    "floor": "hv213",
    "rooms": "hv216",
    "electr": "hv206",
    "tv": "hv208",
    "fridge": "hv209",
    "bicycle": "hv210",
    "car": "hv212",
    "phone": "hv221",
}

# Order and IDs are defined by SPSS; do not reorder these items.
ITEMS = [
    "water",
    "toilet",
    "floor",
    "tv",
    "fridge",
    "phone",
    "electr",
    "car",
    "bicycle",
    "cheaputen",
    "exputen",
    "sroom",
]
ASSETS = ITEMS[3:11]
ITEM_FEATURES = {}
for name in ITEMS:
    if name in ("water", "toilet", "floor", "sroom"):
        prefix = "sleepr" if name == "sroom" else name
        ITEM_FEATURES[name] = [f"{prefix}{level}" for level in (1, 2, 3)]
    else:
        ITEM_FEATURES[name] = [name]
FEATURES = [feature for group in ITEM_FEATURES.values() for feature in group]
DEFAULT_WEIGHTS = Path(EXTERNAL_DIR) / "IWI" / "weights.xlsx"
DEFAULT_QUALITY_RECODE = Path(EXTERNAL_DIR) / "IWI" / "quality_recode.xlsx"


def check_columns(df, columns):
    """Check that explicitly requested columns exist."""
    absent = [name for name in columns if name not in df.columns]
    if absent:
        raise KeyError(f"Missing columns: {absent}")


def load_quality_recode(path=DEFAULT_QUALITY_RECODE):
    """
    Read editable DHS-to-quality maps from water, toilet, and floor sheets.
    Each sheet contains dhs_code, dhs_label, quality_code. Labels are for the
    user and do not affect scoring.
    """
    tables = pd.read_excel(path, sheet_name=["water", "toilet", "floor"])
    mappings = {}
    for name, table in tables.items():
        check_columns(table, ["dhs_code", "dhs_label", "quality_code"])
        table = table.dropna(subset=["dhs_code"])
        if table["dhs_code"].duplicated().any():
            raise ValueError(
                f"{name}: DHS codes must be unique in the recoding workbook."
            )
        if not table["quality_code"].dropna().isin([1, 2, 3]).all():
            raise ValueError(f"{name}: quality codes must be 1, 2, 3, or blank.")
        mappings[name] = table.set_index("dhs_code")["quality_code"].dropna().to_dict()
    return mappings


def make_dummies(category, prefix):
    """Create three dummies, preserving NaN and SPSS -9 as missing."""
    category = category.replace(-9, np.nan)
    return pd.DataFrame(
        {
            f"{prefix}{level}": category.eq(level).astype(float).where(category.notna())
            for level in (1, 2, 3)
        },
        index=category.index,
    )


def any_owned(df, columns, yes_codes=(1,), no_codes=(0,)):
    """Combine observed assets: any yes is yes; only all observed no is no."""
    available = [name for name in columns if name in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    raw = df[available]
    # An absent requested source prevents a definite no, but not a definite yes.
    all_no = raw.isin(no_codes).all(axis=1) & (len(available) == len(columns))
    any_yes = raw.isin(yes_codes).any(axis=1)
    return pd.Series(np.nan, index=df.index).mask(all_no, 0.0).mask(any_yes, 1.0)


def encode_iwi_variables(indicators):
    """
    Encode standard SPSS items and apply its two utensil ownership rules.

    Supply water/toilet/floor/sroom as 1, 2, 3 and binary items as 0, 1.
    Absent columns, NaN, and -9 are missing. Other columns (including IDs)
    are retained. Missing utensils are not inferred from other possessions.
    """
    result = indicators.copy()
    items = indicators.reindex(columns=ITEMS).replace(-9, np.nan).astype(float)
    for name in ITEMS:
        valid = (1, 2, 3) if name in ("water", "toilet", "floor", "sroom") else (0, 1)
        items[name] = items[name].where(items[name].isin(valid))

    expensive = (
        items[["tv", "fridge", "phone", "bicycle", "car", "exputen"]].eq(1).any(axis=1)
    )
    expensive |= items["toilet"].eq(3) | items["floor"].eq(3)
    items.loc[items["cheaputen"].eq(0) & expensive, "cheaputen"] = 1
    items.loc[items["exputen"].eq(0) & items["car"].eq(1), "exputen"] = 1

    for name in ITEMS:
        if name in ("water", "toilet", "floor", "sroom"):
            prefix = "sleepr" if name == "sroom" else name
            result = result.drop(columns=[name], errors="ignore")
            result[ITEM_FEATURES[name]] = make_dummies(items[name], prefix)
        else:
            result[name] = items[name]
    return result


def prepare_iwi_variables(
    df,
    key_cols=None,
    carry_cols=None,
    columns=None,
    quality_recode=DEFAULT_QUALITY_RECODE,
    yes_codes=(1,),
    no_codes=(0,),
    rooms_missing_codes=(98, 99),
    utensil_columns=None,
    phone_columns=None,
    include_utensils=False,
):
    """
    Recode DHS sources to IWI features, leaving unavailable assets missing.

    `columns` maps standard names to raw columns (including optional `cheaputen` and `exputen`).
    `quality_recode` is the editable Excel mapping workbook.
    `utensil_columns` can map cheaputen/exputen to lists of surveyed possessions.
    `phone_columns` optionally combines landline/mobile sources; the original default uses hv221 only.
    Utensil sources are ignored unless `include_utensils` is True. Choose sources using the survey questionnaire.
    """

    key_cols = ["hv001", "hv002"] if key_cols is None else list(key_cols)
    carry_cols = [] if carry_cols is None else list(carry_cols)
    check_columns(df, key_cols + carry_cols)

    if key_cols and df.duplicated(key_cols).any():
        raise ValueError("Use household data with unique household identifiers.")
    if set(yes_codes) & set(no_codes):
        raise ValueError("yes_codes and no_codes must not overlap.")

    sources = {**DEFAULT_COLUMNS, **(columns or {})}
    mappings = load_quality_recode(quality_recode)
    metadata = list(dict.fromkeys(key_cols + carry_cols))

    if set(metadata) & (set(ITEMS) | set(FEATURES) | {"iwi", "itemmis"}):
        raise ValueError("Metadata names overlap with IWI output columns.")
    result = df[metadata].copy()

    for name in ("water", "toilet", "floor"):
        raw = df.get(sources[name], pd.Series(np.nan, index=df.index))
        raw = raw.replace(99, np.nan)
        result[name] = raw.map(mappings[name])
        unmapped = raw[raw.notna() & raw.ne(-9) & result[name].isna()].unique()
        if len(unmapped):
            warnings.warn(
                f"{name}: unmapped codes {unmapped.tolist()} remain missing.",
                stacklevel=2,
            )

    for name in ASSETS:
        source = sources.get(name)
        if name in ("cheaputen", "exputen") and not include_utensils:
            result[name] = np.nan
        else:
            result[name] = any_owned(df, [source], yes_codes, no_codes)

    if phone_columns is not None:
        result["phone"] = any_owned(df, phone_columns, yes_codes, no_codes)
    utensil_sources = (utensil_columns or {}) if include_utensils else {}

    for name, raw_columns in utensil_sources.items():
        if name not in ("cheaputen", "exputen"):
            raise ValueError("utensil_columns keys must be cheaputen or exputen.")
        result[name] = any_owned(df, raw_columns, yes_codes, no_codes)

    rooms = df.get(sources["rooms"], pd.Series(np.nan, index=df.index)).astype(float)
    rooms = rooms.replace(list(rooms_missing_codes) + [-9], np.nan)
    rooms = rooms.where(np.isfinite(rooms) & rooms.ge(0) & rooms.eq(np.floor(rooms)))
    result["sroom"] = np.select(
        [rooms.le(1), rooms.eq(2), rooms.ge(3)],
        [1.0, 2.0, 3.0],
        default=np.nan,
    )
    return encode_iwi_variables(result)


def extract_iwi_weights(sps_path, output_path=None):
    """
    Extract fixed formulas into tables; optionally write template-style sheets.

    Each worksheet is named by its SPSS itemmis code and contains id, variable,
    weight columns, including constant. Numeric terms are parsed without eval.
    """
    text = Path(sps_path).read_text(encoding="utf-8-sig")
    formulas = re.findall(
        r"compute\s+iwi\s*=([^;\r\n]+?)\.\s*compute\s+itemmis\s*=\s*(\d+)\.",
        text,
        flags=re.IGNORECASE,
    )
    feature_ids = {
        feature: i
        for i, group in enumerate(ITEM_FEATURES.values(), 1)
        for feature in group
    }

    tables = {}
    for expression, code in formulas:
        expression = expression.replace(" ", "").replace("+-", "-")
        constant = re.match(r"[+-]?\d+(?:\.\d+)?", expression)
        terms = re.findall(
            r"([+-]\d+(?:\.\d+)?)\*([a-z]+\d*)", expression[constant.end() :]
        )

        if [name for _, name in terms] != FEATURES:
            raise ValueError(f"Unexpected formula structure for itemmis={code}.")

        tables[code] = pd.DataFrame(
            [(0, "constant", float(constant.group()))]
            + [(feature_ids[name], name, float(weight)) for weight, name in terms],
            columns=["id", "variable", "weight"],
        )

    if len(tables) != 299:
        raise ValueError(f"Expected 299 SPSS formulas; found {len(tables)}.")

    if output_path is not None:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for code, table in tables.items():
                table.to_excel(writer, sheet_name=code, index=False)
    return tables


def load_iwi_weights(path=DEFAULT_WEIGHTS):
    """Read all formula sheets, indexed by their stable itemmis codes."""
    return pd.read_excel(path, sheet_name=None)


def missing_item_codes(variables):
    """Encode missing item IDs in base 100 (e.g. water + toilet -> 102)."""
    features = variables.reindex(columns=FEATURES).replace(-9, np.nan)
    codes = np.zeros(len(features), dtype=np.int64)
    counts = np.zeros(len(features), dtype=int)

    for item_id, group in enumerate(ITEM_FEATURES.values(), 1):
        absent = features[group].isna().any(axis=1).to_numpy()
        # Only codes for up to three missing items are meaningful in SPSS.
        codes = np.where(absent & (counts < 3), codes * 100 + item_id, codes)
        counts += absent
    return pd.Series(np.where(counts <= 3, codes, -9), index=variables.index)


def score_fixed_weights(variables, weights, missing="raise"):
    """Apply one selected formula, excluding zero terms before multiplication."""
    coefficients = (
        weights.set_index("variable")["weight"]
        if isinstance(weights, pd.DataFrame)
        else weights
    )

    if (
        not coefficients.index.is_unique
        or not np.isfinite(coefficients.astype(float)).all()
    ):
        raise ValueError("Weights must have unique names and finite numeric values.")

    constant = float(coefficients.loc["constant"])
    active = coefficients.drop("constant").astype(float)
    active = active[active.ne(0)]
    check_columns(variables, active.index)
    values = variables[active.index].replace(-9, np.nan).astype(float)

    if missing == "raise" and values.isna().any(axis=None):
        raise ValueError(
            "Missing values in active indicators for the selected formula."
        )
    return values.to_numpy() @ active.to_numpy() + constant


def compute_iwi(
    variables, weights=None, missing="raise", decimals=None, *, include_utensils=False
):
    """
    Score automatically with a workbook/dict, or use one legacy weight table.

    Automatic mode returns iwi and itemmis, using exact per-household missing patterns.
    More than three missing items gives itemmis=-9 and either raises (default) or leaves
    iwi=NaN with missing='propagate'.
    Pass a single DataFrame or Series to use an explicitly selected-formula interface.
    Utensils are excluded by default, even if supplied in variables. Enable `include_utensils` to use them.
    Fixed tables must respect this choice.
    """
    if missing not in ("raise", "propagate"):
        raise ValueError("missing must be 'raise' or 'propagate'.")

    if weights is None or isinstance(weights, (str, Path)):
        weights = load_iwi_weights(DEFAULT_WEIGHTS if weights is None else weights)

    variables = variables.copy()
    if not include_utensils:
        variables[["cheaputen", "exputen"]] = np.nan

    result = variables.copy()
    if isinstance(weights, dict):
        tables = {str(code): table for code, table in weights.items()}
        codes = missing_item_codes(variables)
        unsupported = ~codes.astype(str).isin(tables)

        if unsupported.any():
            message = f"{int(unsupported.sum())} households have no matching formula (usually >3 missing items)."
            if missing == "raise":
                raise ValueError(message)
            warnings.warn(message + " Their IWI remains NaN.", stacklevel=2)

        scores = np.full(len(variables), np.nan)
        for code in codes[~unsupported].unique():
            positions = np.flatnonzero(codes.eq(code).to_numpy())
            scores[positions] = score_fixed_weights(
                variables.iloc[positions], tables[str(code)]
            )

        result["itemmis"] = codes.to_numpy()
        result["iwi"] = scores
    else:
        coefficients = (
            weights.set_index("variable")["weight"]
            if isinstance(weights, pd.DataFrame)
            else weights
        )

        if (
            not include_utensils
            and coefficients.reindex(["cheaputen", "exputen"], fill_value=0).ne(0).any()
        ):
            raise ValueError(
                "Selected weights use utensils. Set include_utensils=True or select weights excluding both utensils."
            )
        result["iwi"] = score_fixed_weights(variables, weights, missing)

    if decimals is not None:
        result["iwi"] = result["iwi"].round(decimals)
    return result


def build_iwi(
    df,
    weights=None,
    *,
    missing="raise",
    decimals=None,
    include_utensils=False,
    **prepare_options,
):
    """Prepare DHS household indicators and calculate fixed-coefficient IWI."""
    variables = prepare_iwi_variables(
        df, include_utensils=include_utensils, **prepare_options
    )
    return compute_iwi(
        variables,
        weights,
        missing=missing,
        decimals=decimals,
        include_utensils=include_utensils,
    )
