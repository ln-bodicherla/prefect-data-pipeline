"""
Feature Engineering Pipeline - Prefect flow for computing derived features.

Runs downstream of the claims ETL pipeline to produce feature tables
for analytics and machine learning consumption.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_latest_parquet(directory: str) -> Optional[str]:
    """Find the most recently modified Parquet file in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    parquet_files = sorted(dir_path.glob("*.parquet"), key=os.path.getmtime)
    return str(parquet_files[-1]) if parquet_files else None


@task(
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=2),
    tags=["features", "provider"],
)
def compute_provider_features(
    claims_df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Compute provider-level features from claims data.

    Features include claim volume metrics, financial summaries,
    denial rates, and specialty concentration indices.
    """
    logger = get_run_logger()
    logger.info(f"Computing provider features: {feature_list}")

    provider_groups = claims_df.groupby("provider_id")

    features = pd.DataFrame({"provider_id": claims_df["provider_id"].unique()})

    if "total_claims" in feature_list:
        total_claims = provider_groups["claim_id"].count().reset_index()
        total_claims.columns = ["provider_id", "provider_total_claims"]
        features = features.merge(total_claims, on="provider_id", how="left")

    if "avg_claim_amount" in feature_list:
        avg_amount = provider_groups["claim_amount"].mean().reset_index()
        avg_amount.columns = ["provider_id", "provider_avg_claim_amount"]
        features = features.merge(avg_amount, on="provider_id", how="left")

    if "claim_denial_rate" in feature_list and "status" in claims_df.columns:
        denial_rate = (
            provider_groups["status"]
            .apply(lambda x: (x == "denied").sum() / len(x) if len(x) > 0 else 0.0)
            .reset_index()
        )
        denial_rate.columns = ["provider_id", "provider_denial_rate"]
        features = features.merge(denial_rate, on="provider_id", how="left")

    if "unique_patients" in feature_list:
        unique_patients = provider_groups["patient_id"].nunique().reset_index()
        unique_patients.columns = ["provider_id", "provider_unique_patients"]
        features = features.merge(unique_patients, on="provider_id", how="left")

    if "specialty_concentration" in feature_list and "diagnosis_code" in claims_df.columns:
        def herfindahl_index(series):
            counts = series.value_counts(normalize=True)
            return (counts ** 2).sum()

        concentration = (
            provider_groups["diagnosis_code"]
            .apply(herfindahl_index)
            .reset_index()
        )
        concentration.columns = ["provider_id", "provider_specialty_concentration"]
        features = features.merge(concentration, on="provider_id", how="left")

    claim_std = provider_groups["claim_amount"].std().fillna(0).reset_index()
    claim_std.columns = ["provider_id", "provider_claim_amount_std"]
    features = features.merge(claim_std, on="provider_id", how="left")

    claim_median = provider_groups["claim_amount"].median().reset_index()
    claim_median.columns = ["provider_id", "provider_claim_amount_median"]
    features = features.merge(claim_median, on="provider_id", how="left")

    logger.info(
        f"Provider features computed: {features.shape[0]} providers, "
        f"{features.shape[1] - 1} features"
    )
    return features


@task(
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=2),
    tags=["features", "patient"],
)
def compute_patient_features(
    claims_df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Compute patient-level features from claims data.

    Features include utilization metrics, cost patterns,
    provider diversity, and diagnosis complexity.
    """
    logger = get_run_logger()
    logger.info(f"Computing patient features: {feature_list}")

    patient_groups = claims_df.groupby("patient_id")

    features = pd.DataFrame({"patient_id": claims_df["patient_id"].unique()})

    if "total_claims" in feature_list:
        total_claims = patient_groups["claim_id"].count().reset_index()
        total_claims.columns = ["patient_id", "patient_total_claims"]
        features = features.merge(total_claims, on="patient_id", how="left")

    if "avg_claim_amount" in feature_list:
        avg_amount = patient_groups["claim_amount"].mean().reset_index()
        avg_amount.columns = ["patient_id", "patient_avg_claim_amount"]
        features = features.merge(avg_amount, on="provider_id" if "provider_id" in features.columns else "patient_id", how="left")
        if "patient_avg_claim_amount" not in features.columns:
            avg_amount.columns = ["patient_id", "patient_avg_claim_amount"]
            features = features.merge(avg_amount, on="patient_id", how="left")

    if "unique_providers" in feature_list:
        unique_providers = patient_groups["provider_id"].nunique().reset_index()
        unique_providers.columns = ["patient_id", "patient_unique_providers"]
        features = features.merge(unique_providers, on="patient_id", how="left")

    if "diagnosis_diversity" in feature_list and "diagnosis_code" in claims_df.columns:
        diag_diversity = patient_groups["diagnosis_code"].nunique().reset_index()
        diag_diversity.columns = ["patient_id", "patient_diagnosis_diversity"]
        features = features.merge(diag_diversity, on="patient_id", how="left")

    if "claim_frequency_days" in feature_list and "claim_date" in claims_df.columns:
        claims_sorted = claims_df.sort_values(["patient_id", "claim_date"])
        claims_sorted["prev_claim_date"] = claims_sorted.groupby("patient_id")[
            "claim_date"
        ].shift(1)
        claims_sorted["days_between"] = (
            claims_sorted["claim_date"] - claims_sorted["prev_claim_date"]
        ).dt.days
        avg_frequency = (
            claims_sorted.groupby("patient_id")["days_between"]
            .mean()
            .reset_index()
        )
        avg_frequency.columns = ["patient_id", "patient_avg_days_between_claims"]
        features = features.merge(avg_frequency, on="patient_id", how="left")

    total_amount = patient_groups["claim_amount"].sum().reset_index()
    total_amount.columns = ["patient_id", "patient_total_amount"]
    features = features.merge(total_amount, on="patient_id", how="left")

    max_amount = patient_groups["claim_amount"].max().reset_index()
    max_amount.columns = ["patient_id", "patient_max_claim_amount"]
    features = features.merge(max_amount, on="patient_id", how="left")

    logger.info(
        f"Patient features computed: {features.shape[0]} patients, "
        f"{features.shape[1] - 1} features"
    )
    return features


@task(
    retries=1,
    retry_delay_seconds=10,
    tags=["features", "temporal"],
)
def compute_temporal_features(
    claims_df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Compute time-based features for each claim record.

    Features include rolling window aggregates, day-of-week patterns,
    and monthly trend indicators.
    """
    logger = get_run_logger()
    logger.info(f"Computing temporal features: {feature_list}")

    df = claims_df.copy()

    if "claim_date" not in df.columns:
        logger.warning("No claim_date column found, skipping temporal features")
        return df[["claim_id"]].assign(
            **{f: np.nan for f in feature_list}
        )

    df = df.sort_values("claim_date")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df = df.set_index("claim_date")

    result = df[["claim_id"]].copy()

    if "rolling_7d_claims" in feature_list:
        rolling_7d = df.groupby("patient_id")["claim_id"].rolling("7D").count()
        rolling_7d = rolling_7d.reset_index(level=0, drop=True)
        result["rolling_7d_claims"] = rolling_7d.values

    if "rolling_30d_amount" in feature_list:
        rolling_30d = df.groupby("patient_id")["claim_amount"].rolling("30D").sum()
        rolling_30d = rolling_30d.reset_index(level=0, drop=True)
        result["rolling_30d_amount"] = rolling_30d.values

    if "day_of_week_pattern" in feature_list:
        result["day_of_week"] = df.index.dayofweek
        result["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    if "monthly_trend" in feature_list:
        result["month"] = df.index.month
        result["quarter"] = df.index.quarter
        result["year_month"] = df.index.to_period("M").astype(str)

    result = result.reset_index(drop=True)

    logger.info(
        f"Temporal features computed: {result.shape[0]} rows, "
        f"{result.shape[1] - 1} features"
    )
    return result


@task(tags=["features"])
def combine_features(
    claims_df: pd.DataFrame,
    provider_features: pd.DataFrame,
    patient_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all feature sets into a unified feature table.

    Joins provider-level, patient-level, and temporal features
    onto the base claims data with proper alignment.
    """
    logger = get_run_logger()

    combined = claims_df[["claim_id", "patient_id", "provider_id"]].copy()

    combined = combined.merge(provider_features, on="provider_id", how="left")
    combined = combined.merge(patient_features, on="patient_id", how="left")

    temporal_cols = [c for c in temporal_features.columns if c != "claim_id"]
    if temporal_cols:
        temporal_subset = temporal_features[["claim_id"] + temporal_cols]
        combined = combined.merge(temporal_subset, on="claim_id", how="left")

    combined["_feature_timestamp"] = datetime.utcnow().isoformat()

    null_rates = combined.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.5].index.tolist()
    if high_null_cols:
        logger.warning(f"High null rate columns (>50%): {high_null_cols}")

    logger.info(
        f"Combined feature table: {combined.shape[0]} rows, "
        f"{combined.shape[1]} columns"
    )
    return combined


@task(tags=["features", "quality"])
def validate_features(
    feature_df: pd.DataFrame,
    min_features: int = 5,
) -> dict[str, Any]:
    """
    Validate the feature table for completeness and correctness.

    Checks for minimum feature count, null rates, infinite values,
    and feature variance.
    """
    logger = get_run_logger()

    validation = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_rows": len(feature_df),
        "total_columns": len(feature_df.columns),
        "checks": [],
        "passed": True,
    }

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_count = len(numeric_cols)
    if feature_count < min_features:
        validation["checks"].append(
            {"check": "min_features", "passed": False, "detail": f"{feature_count} < {min_features}"}
        )
        validation["passed"] = False
    else:
        validation["checks"].append({"check": "min_features", "passed": True})

    for col in numeric_cols:
        inf_count = np.isinf(feature_df[col].dropna()).sum()
        if inf_count > 0:
            validation["checks"].append(
                {"check": f"no_inf_{col}", "passed": False, "detail": f"{inf_count} infinite values"}
            )
            validation["passed"] = False

    for col in numeric_cols:
        variance = feature_df[col].var()
        if variance == 0:
            validation["checks"].append(
                {"check": f"nonzero_variance_{col}", "passed": False, "detail": "zero variance"}
            )

    overall_null = feature_df[numeric_cols].isnull().mean().mean()
    validation["overall_null_rate"] = float(overall_null)
    validation["checks"].append(
        {
            "check": "overall_null_rate",
            "passed": overall_null < 0.2,
            "detail": f"{overall_null:.4f}",
        }
    )

    passed_count = sum(1 for c in validation["checks"] if c["passed"])
    total_count = len(validation["checks"])
    logger.info(f"Feature validation: {passed_count}/{total_count} checks passed")

    return validation


@flow(
    name="feature_engineering_pipeline",
    description="Compute derived features from Silver claims data",
    retries=1,
    retry_delay_seconds=120,
    timeout_seconds=1800,
)
def feature_engineering_pipeline(
    config_path: str = "configs/config.yaml",
    silver_claims_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Feature engineering pipeline that runs after the claims ETL pipeline.

    Computes provider-level, patient-level, and temporal features,
    combines them into a unified feature table, and validates the output.
    """
    logger = get_run_logger()
    start_time = datetime.utcnow()

    config = load_config(config_path)
    paths = config["paths"]
    feature_config = config.get("features", {})

    if silver_claims_path is None:
        silver_dir = Path(paths["silver"]) / "claims"
        silver_claims_path = _get_latest_parquet(str(silver_dir))

    if silver_claims_path is None:
        logger.error("No Silver claims data found, running claims pipeline first")
        from flows.claims_pipeline import claims_etl_pipeline

        claims_result = claims_etl_pipeline(config_path=config_path, strict_quality=False)
        silver_claims_path = claims_result["stages"]["silver"]

    logger.info(f"Loading Silver claims from: {silver_claims_path}")
    claims_df = pd.read_parquet(silver_claims_path)

    if "claim_date" in claims_df.columns:
        claims_df["claim_date"] = pd.to_datetime(claims_df["claim_date"], errors="coerce")

    provider_features = compute_provider_features(
        claims_df,
        feature_config.get("provider", ["total_claims", "avg_claim_amount"]),
    )

    patient_features = compute_patient_features(
        claims_df,
        feature_config.get("patient", ["total_claims", "avg_claim_amount"]),
    )

    temporal_features = compute_temporal_features(
        claims_df,
        feature_config.get("temporal", ["rolling_7d_claims"]),
    )

    combined = combine_features(
        claims_df, provider_features, patient_features, temporal_features
    )

    validation = validate_features(combined)

    output_dir = Path(paths["gold"]) / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = str(output_dir / f"claim_features_{batch_id}.parquet")
    combined.to_parquet(output_path, compression="snappy")

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    logger.info(
        f"Feature pipeline complete: {combined.shape[1]} features, "
        f"{combined.shape[0]} rows, {duration:.1f}s"
    )

    return {
        "output_path": output_path,
        "feature_count": combined.shape[1],
        "row_count": combined.shape[0],
        "validation": validation,
        "duration_seconds": duration,
    }


if __name__ == "__main__":
    results = feature_engineering_pipeline()
    print(f"\nFeature pipeline completed")
    print(f"Output: {results['output_path']}")
    print(f"Features: {results['feature_count']}, Rows: {results['row_count']}")
    print(f"Duration: {results['duration_seconds']:.1f}s")
