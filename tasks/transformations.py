"""
Reusable Prefect tasks for data transformations.

Provides composable transformation primitives that can be used across
multiple flows: cleaning, deduplication, type enforcement, SCD merges,
and configurable aggregation.
"""

from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from prefect import task, get_run_logger


@task(
    retries=1,
    retry_delay_seconds=5,
    tags=["transform", "clean"],
)
def clean_data(
    df: pd.DataFrame,
    null_strategy: str = "drop_threshold",
    threshold: float = 0.5,
    outlier_method: Optional[str] = None,
    outlier_columns: Optional[list[str]] = None,
    outlier_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Clean data by handling nulls and optionally removing outliers.

    Null strategies:
        - 'drop_all': Drop any row with nulls
        - 'drop_threshold': Drop columns exceeding null threshold
        - 'fill_mean': Fill numeric nulls with column mean
        - 'fill_median': Fill numeric nulls with column median
        - 'fill_mode': Fill all nulls with column mode
        - 'fill_zero': Fill numeric nulls with zero

    Outlier methods:
        - 'zscore': Remove rows with z-score > outlier_factor
        - 'iqr': Remove rows outside IQR * outlier_factor range
    """
    logger = get_run_logger()
    initial_rows = len(df)
    initial_cols = len(df.columns)

    result = df.copy()

    if null_strategy == "drop_all":
        result = result.dropna()

    elif null_strategy == "drop_threshold":
        null_rates = result.isnull().mean()
        cols_to_drop = null_rates[null_rates > threshold].index.tolist()
        if cols_to_drop:
            logger.info(
                f"Dropping {len(cols_to_drop)} columns exceeding "
                f"{threshold:.0%} null threshold: {cols_to_drop}"
            )
            result = result.drop(columns=cols_to_drop)
        remaining_null_cols = result.columns[result.isnull().any()].tolist()
        if remaining_null_cols:
            result = result.dropna(subset=remaining_null_cols, how="all")

    elif null_strategy == "fill_mean":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())

    elif null_strategy == "fill_median":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())

    elif null_strategy == "fill_mode":
        for col in result.columns:
            if result[col].isnull().any():
                mode_val = result[col].mode()
                if not mode_val.empty:
                    result[col] = result[col].fillna(mode_val.iloc[0])

    elif null_strategy == "fill_zero":
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

    if outlier_method and outlier_columns:
        if outlier_method == "zscore":
            for col in outlier_columns:
                if col in result.columns and np.issubdtype(result[col].dtype, np.number):
                    col_mean = result[col].mean()
                    col_std = result[col].std()
                    if col_std > 0:
                        z_scores = np.abs((result[col] - col_mean) / col_std)
                        result = result[z_scores <= outlier_factor]

        elif outlier_method == "iqr":
            for col in outlier_columns:
                if col in result.columns and np.issubdtype(result[col].dtype, np.number):
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - outlier_factor * iqr
                    upper = q3 + outlier_factor * iqr
                    result = result[(result[col] >= lower) & (result[col] <= upper)]

    result = result.reset_index(drop=True)

    logger.info(
        f"Clean: {initial_rows} -> {len(result)} rows, "
        f"{initial_cols} -> {len(result.columns)} columns "
        f"(strategy={null_strategy})"
    )
    return result


@task(
    retries=1,
    tags=["transform", "dedup"],
)
def deduplicate(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    strategy: str = "exact",
    keep: str = "first",
    similarity_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Remove duplicate records using exact or fuzzy matching.

    Strategies:
        - 'exact': Standard pandas deduplication on specified columns
        - 'fuzzy': Similarity-based dedup using string distance
          (applied to string columns only)
    """
    logger = get_run_logger()
    initial_rows = len(df)

    result = df.copy()

    if strategy == "exact":
        if subset:
            valid_subset = [c for c in subset if c in result.columns]
            result = result.drop_duplicates(subset=valid_subset, keep=keep)
        else:
            non_meta_cols = [c for c in result.columns if not c.startswith("_")]
            result = result.drop_duplicates(subset=non_meta_cols, keep=keep)

    elif strategy == "fuzzy":
        string_cols = result.select_dtypes(include=["object"]).columns.tolist()
        if subset:
            string_cols = [c for c in subset if c in string_cols]

        if string_cols:
            result["_concat_key"] = result[string_cols].astype(str).agg("|".join, axis=1)
            result["_normalized_key"] = (
                result["_concat_key"]
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"[^\w\s|]", "", regex=True)
            )

            seen_keys = {}
            rows_to_keep = []

            for idx, key in result["_normalized_key"].items():
                is_duplicate = False
                for seen_key in seen_keys:
                    if _string_similarity(key, seen_key) >= similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    seen_keys[key] = idx
                    rows_to_keep.append(idx)

            result = result.loc[rows_to_keep]
            result = result.drop(columns=["_concat_key", "_normalized_key"])

    result = result.reset_index(drop=True)
    removed = initial_rows - len(result)
    logger.info(
        f"Dedup ({strategy}): {initial_rows} -> {len(result)} rows "
        f"({removed} duplicates removed)"
    )
    return result


def _string_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between two strings based on character bigrams."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    bigrams1 = set(s1[i : i + 2] for i in range(len(s1) - 1))
    bigrams2 = set(s2[i : i + 2] for i in range(len(s2) - 1))

    if not bigrams1 or not bigrams2:
        return 0.0

    intersection = bigrams1 & bigrams2
    union = bigrams1 | bigrams2

    return len(intersection) / len(union)


@task(
    retries=1,
    tags=["transform", "schema"],
)
def type_cast(
    df: pd.DataFrame,
    schema: dict[str, str],
) -> pd.DataFrame:
    """
    Enforce column data types according to a schema specification.

    Handles type coercion with error handling for incompatible values.
    Unspecified columns are left unchanged.
    """
    logger = get_run_logger()
    result = df.copy()
    cast_errors = []

    type_mapping = {
        "string": "object",
        "str": "object",
        "int": "int64",
        "int32": "int32",
        "int64": "int64",
        "float": "float64",
        "float32": "float32",
        "float64": "float64",
        "bool": "bool",
        "boolean": "bool",
        "datetime": "datetime64[ns]",
        "datetime64": "datetime64[ns]",
        "category": "category",
    }

    for col, target_type in schema.items():
        if col not in result.columns:
            logger.warning(f"Schema column '{col}' not found in dataframe")
            continue

        resolved_type = type_mapping.get(target_type, target_type)

        try:
            if resolved_type == "datetime64[ns]":
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif resolved_type in ("int64", "int32"):
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].astype("Int64")
            elif resolved_type in ("float64", "float32"):
                result[col] = pd.to_numeric(result[col], errors="coerce")
            elif resolved_type == "bool":
                result[col] = result[col].astype(bool)
            elif resolved_type == "category":
                result[col] = result[col].astype("category")
            else:
                result[col] = result[col].astype(resolved_type)
        except (ValueError, TypeError) as e:
            cast_errors.append({"column": col, "target_type": target_type, "error": str(e)})
            logger.warning(f"Failed to cast '{col}' to {target_type}: {e}")

    if cast_errors:
        logger.warning(f"Type casting had {len(cast_errors)} errors")

    logger.info(f"Type cast applied: {len(schema)} columns specified")
    return result


@task(
    retries=1,
    tags=["transform", "scd"],
)
def scd_type2_merge(
    current_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    key_columns: list[str],
    tracked_columns: list[str],
    effective_date_col: str = "effective_date",
    end_date_col: str = "end_date",
    current_flag_col: str = "is_current",
) -> pd.DataFrame:
    """
    Perform a Slowly Changing Dimension Type 2 merge.

    Tracks historical changes by expiring old records and inserting
    new versions when tracked columns change.
    """
    logger = get_run_logger()
    now = datetime.utcnow()

    if current_df.empty:
        result = incoming_df.copy()
        result[effective_date_col] = now
        result[end_date_col] = pd.NaT
        result[current_flag_col] = True
        logger.info(f"SCD2: Initial load with {len(result)} records")
        return result

    current = current_df.copy()
    incoming = incoming_df.copy()

    current_active = current[current[current_flag_col] == True].copy()
    current_expired = current[current[current_flag_col] == False].copy()

    merged = current_active.merge(
        incoming[key_columns + tracked_columns],
        on=key_columns,
        how="outer",
        suffixes=("_current", "_incoming"),
        indicator=True,
    )

    unchanged_mask = merged["_merge"] == "both"
    for col in tracked_columns:
        current_col = f"{col}_current"
        incoming_col = f"{col}_incoming"
        if current_col in merged.columns and incoming_col in merged.columns:
            unchanged_mask = unchanged_mask & (
                merged[current_col].fillna("__NULL__") == merged[incoming_col].fillna("__NULL__")
            )

    unchanged = merged[unchanged_mask].copy()
    for col in tracked_columns:
        if f"{col}_current" in unchanged.columns:
            unchanged[col] = unchanged[f"{col}_current"]

    changed = merged[(merged["_merge"] == "both") & ~unchanged_mask].copy()
    new_records = merged[merged["_merge"] == "right_only"].copy()

    expired_records = []
    for col in tracked_columns:
        current_col = f"{col}_current"
        if current_col in changed.columns:
            changed[col] = changed[f"{col}_current"]

    if not changed.empty:
        expired_version = changed.copy()
        expired_version[end_date_col] = now
        expired_version[current_flag_col] = False
        expired_records.append(expired_version)

        new_version = changed.copy()
        for col in tracked_columns:
            incoming_col = f"{col}_incoming"
            if incoming_col in new_version.columns:
                new_version[col] = new_version[incoming_col]
        new_version[effective_date_col] = now
        new_version[end_date_col] = pd.NaT
        new_version[current_flag_col] = True

    if not new_records.empty:
        for col in tracked_columns:
            incoming_col = f"{col}_incoming"
            if incoming_col in new_records.columns:
                new_records[col] = new_records[incoming_col]
        new_records[effective_date_col] = now
        new_records[end_date_col] = pd.NaT
        new_records[current_flag_col] = True

    all_parts = [current_expired, unchanged]
    if not changed.empty:
        all_parts.extend(expired_records)
        all_parts.append(new_version)
    if not new_records.empty:
        all_parts.append(new_records)

    result = pd.concat(all_parts, ignore_index=True)

    drop_cols = [c for c in result.columns if c.endswith(("_current", "_incoming")) or c == "_merge"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])

    logger.info(
        f"SCD2 merge: {len(current)} existing + {len(incoming)} incoming "
        f"-> {len(result)} total records"
    )
    return result


@task(
    retries=1,
    tags=["transform", "aggregate"],
)
def aggregate_metrics(
    df: pd.DataFrame,
    group_by: list[str],
    aggregations: dict[str, list[str]],
    filters: Optional[dict[str, Any]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Compute configurable aggregations grouped by specified columns.

    Supports standard aggregation functions (sum, mean, count, min, max,
    median, std, nunique) with optional pre-filtering and sorting.
    """
    logger = get_run_logger()
    result = df.copy()

    if filters:
        for col, condition in filters.items():
            if col not in result.columns:
                continue
            if isinstance(condition, list):
                result = result[result[col].isin(condition)]
            elif isinstance(condition, dict):
                if "min" in condition:
                    result = result[result[col] >= condition["min"]]
                if "max" in condition:
                    result = result[result[col] <= condition["max"]]
                if "eq" in condition:
                    result = result[result[col] == condition["eq"]]
                if "neq" in condition:
                    result = result[result[col] != condition["neq"]]
            else:
                result = result[result[col] == condition]

    valid_group_by = [c for c in group_by if c in result.columns]
    if not valid_group_by:
        logger.warning(f"No valid group-by columns found: {group_by}")
        return pd.DataFrame()

    valid_aggregations = {}
    for col, agg_funcs in aggregations.items():
        if col in result.columns:
            valid_aggregations[col] = agg_funcs
        else:
            logger.warning(f"Aggregation column '{col}' not found")

    if not valid_aggregations:
        logger.warning("No valid aggregation columns found")
        return pd.DataFrame()

    aggregated = result.groupby(valid_group_by).agg(valid_aggregations)

    aggregated.columns = [
        f"{col}_{func}" for col, func in aggregated.columns
    ]
    aggregated = aggregated.reset_index()

    if sort_by and sort_by in aggregated.columns:
        aggregated = aggregated.sort_values(sort_by, ascending=ascending)

    aggregated = aggregated.reset_index(drop=True)

    logger.info(
        f"Aggregation: {len(df)} rows -> {len(aggregated)} groups "
        f"({len(valid_group_by)} group-by cols, "
        f"{len(valid_aggregations)} metric cols)"
    )
    return aggregated
