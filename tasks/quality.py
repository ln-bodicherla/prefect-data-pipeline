"""
Data quality validation tasks for the Prefect pipeline.

Provides composable quality checks: schema validation, completeness,
freshness monitoring, distribution analysis, and quality reporting.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from prefect import task, get_run_logger


@task(
    retries=1,
    tags=["quality", "schema"],
)
def validate_schema(
    df: pd.DataFrame,
    expected_columns: dict[str, str],
    strict: bool = False,
) -> dict[str, Any]:
    """
    Validate dataframe schema against expected column definitions.

    Checks column presence, data types, and optionally rejects
    unexpected columns in strict mode.
    """
    logger = get_run_logger()
    results = {
        "check": "schema_validation",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": True,
        "details": [],
    }

    actual_columns = set(df.columns)
    expected_cols = set(expected_columns.keys())

    missing_columns = expected_cols - actual_columns
    if missing_columns:
        results["passed"] = False
        results["details"].append(
            {"issue": "missing_columns", "columns": sorted(missing_columns)}
        )
        logger.warning(f"Missing columns: {sorted(missing_columns)}")

    if strict:
        non_meta = {c for c in actual_columns if not c.startswith("_")}
        extra_columns = non_meta - expected_cols
        if extra_columns:
            results["details"].append(
                {"issue": "extra_columns", "columns": sorted(extra_columns)}
            )
            logger.info(f"Extra columns (strict mode): {sorted(extra_columns)}")

    type_mapping = {
        "string": ["object", "string"],
        "str": ["object", "string"],
        "int": ["int64", "int32", "Int64", "Int32"],
        "int64": ["int64", "Int64"],
        "float": ["float64", "float32"],
        "float64": ["float64"],
        "bool": ["bool", "boolean"],
        "datetime": ["datetime64[ns]", "datetime64[ns, UTC]"],
        "datetime64": ["datetime64[ns]", "datetime64[ns, UTC]"],
        "category": ["category"],
    }

    type_mismatches = []
    for col, expected_type in expected_columns.items():
        if col not in df.columns:
            continue

        actual_type = str(df[col].dtype)
        acceptable_types = type_mapping.get(expected_type, [expected_type])

        if actual_type not in acceptable_types:
            type_mismatches.append(
                {
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type,
                }
            )

    if type_mismatches:
        results["details"].append(
            {"issue": "type_mismatches", "mismatches": type_mismatches}
        )
        logger.warning(f"Type mismatches: {len(type_mismatches)} columns")

    passed = results["passed"] and len(type_mismatches) == 0
    results["passed"] = passed

    logger.info(
        f"Schema validation: {'PASSED' if passed else 'FAILED'} "
        f"({len(expected_columns)} columns checked)"
    )
    return results


@task(
    retries=1,
    tags=["quality", "completeness"],
)
def validate_completeness(
    df: pd.DataFrame,
    max_null_rate: float = 0.05,
    critical_columns: Optional[list[str]] = None,
    critical_max_null_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Validate data completeness by checking null rates per column.

    Applies stricter thresholds for critical columns that must
    have zero or near-zero null rates.
    """
    logger = get_run_logger()
    results = {
        "check": "completeness_validation",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": True,
        "column_null_rates": {},
        "failures": [],
    }

    for col in df.columns:
        if col.startswith("_"):
            continue

        null_rate = float(df[col].isnull().mean())
        results["column_null_rates"][col] = round(null_rate, 6)

        is_critical = critical_columns and col in critical_columns
        threshold = critical_max_null_rate if is_critical else max_null_rate

        if null_rate > threshold:
            results["passed"] = False
            results["failures"].append(
                {
                    "column": col,
                    "null_rate": round(null_rate, 6),
                    "threshold": threshold,
                    "is_critical": is_critical,
                }
            )

    overall_null_rate = float(
        df[[c for c in df.columns if not c.startswith("_")]].isnull().mean().mean()
    )
    results["overall_null_rate"] = round(overall_null_rate, 6)
    results["total_rows"] = len(df)
    results["total_nulls"] = int(df.isnull().sum().sum())

    logger.info(
        f"Completeness: {'PASSED' if results['passed'] else 'FAILED'} "
        f"(overall null rate: {overall_null_rate:.4f}, "
        f"{len(results['failures'])} column failures)"
    )
    return results


@task(
    retries=1,
    tags=["quality", "freshness"],
)
def validate_freshness(
    df: pd.DataFrame,
    date_column: str,
    max_age_hours: int = 24,
    reference_time: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Validate that data is within expected recency window.

    Checks the maximum date value in the specified column against
    the reference time to ensure data is not stale.
    """
    logger = get_run_logger()
    ref_time = reference_time or datetime.utcnow()

    results = {
        "check": "freshness_validation",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": True,
        "details": {},
    }

    if date_column not in df.columns:
        results["passed"] = False
        results["details"]["error"] = f"Column '{date_column}' not found"
        return results

    date_series = pd.to_datetime(df[date_column], errors="coerce")
    non_null_dates = date_series.dropna()

    if non_null_dates.empty:
        results["passed"] = False
        results["details"]["error"] = "No valid dates found"
        return results

    max_date = non_null_dates.max()
    min_date = non_null_dates.min()

    if hasattr(max_date, "to_pydatetime"):
        max_date_py = max_date.to_pydatetime()
    else:
        max_date_py = max_date

    if max_date_py.tzinfo is not None:
        max_date_py = max_date_py.replace(tzinfo=None)

    age = ref_time - max_date_py
    age_hours = age.total_seconds() / 3600

    results["details"] = {
        "max_date": str(max_date),
        "min_date": str(min_date),
        "reference_time": str(ref_time),
        "age_hours": round(age_hours, 2),
        "max_allowed_hours": max_age_hours,
        "date_range_days": (max_date - min_date).days,
        "null_date_count": int(date_series.isnull().sum()),
    }

    if age_hours > max_age_hours:
        results["passed"] = False
        logger.warning(
            f"Freshness check FAILED: data is {age_hours:.1f}h old "
            f"(max: {max_age_hours}h)"
        )
    else:
        logger.info(
            f"Freshness check PASSED: data is {age_hours:.1f}h old "
            f"(max: {max_age_hours}h)"
        )

    return results


@task(
    retries=1,
    tags=["quality", "distribution"],
)
def validate_distribution(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    reference_stats: Optional[dict[str, dict]] = None,
    max_drift_pct: float = 0.25,
) -> dict[str, Any]:
    """
    Validate statistical distributions of numeric columns.

    Checks for distribution drift by comparing current statistics
    against reference baselines (mean, std, min, max, quantiles).
    """
    logger = get_run_logger()
    results = {
        "check": "distribution_validation",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": True,
        "column_stats": {},
        "drift_alerts": [],
    }

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [c for c in numeric_columns if not c.startswith("_")]

    for col in numeric_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        stats = {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }
        results["column_stats"][col] = stats

        if reference_stats and col in reference_stats:
            ref = reference_stats[col]
            if "mean" in ref and ref["mean"] != 0:
                drift = abs(stats["mean"] - ref["mean"]) / abs(ref["mean"])
                if drift > max_drift_pct:
                    results["drift_alerts"].append(
                        {
                            "column": col,
                            "metric": "mean",
                            "current": stats["mean"],
                            "reference": ref["mean"],
                            "drift_pct": round(drift, 4),
                        }
                    )
                    results["passed"] = False

            if "std" in ref and ref["std"] != 0:
                std_drift = abs(stats["std"] - ref["std"]) / abs(ref["std"])
                if std_drift > max_drift_pct * 2:
                    results["drift_alerts"].append(
                        {
                            "column": col,
                            "metric": "std",
                            "current": stats["std"],
                            "reference": ref["std"],
                            "drift_pct": round(std_drift, 4),
                        }
                    )

    logger.info(
        f"Distribution validation: {'PASSED' if results['passed'] else 'FAILED'} "
        f"({len(numeric_columns)} columns, {len(results['drift_alerts'])} drift alerts)"
    )
    return results


@task(
    retries=1,
    tags=["quality", "report"],
)
def generate_quality_report(
    validation_results: list[dict[str, Any]],
    output_path: Optional[str] = None,
    report_format: str = "html",
) -> str:
    """
    Generate a consolidated quality report from validation results.

    Produces an HTML report summarizing all quality checks with
    pass/fail indicators, statistics, and actionable details.
    """
    logger = get_run_logger()
    timestamp = datetime.utcnow().isoformat()

    total_checks = len(validation_results)
    passed_checks = sum(1 for r in validation_results if r.get("passed", False))
    failed_checks = total_checks - passed_checks
    overall_status = "PASSED" if failed_checks == 0 else "FAILED"

    if report_format == "html":
        html_rows = ""
        for result in validation_results:
            check_name = result.get("check", "unknown")
            status = "PASS" if result.get("passed", False) else "FAIL"
            status_color = "#28a745" if status == "PASS" else "#dc3545"
            details = json.dumps(result.get("details", result.get("failures", [])), indent=2, default=str)

            html_rows += f"""
            <tr>
                <td>{check_name}</td>
                <td style="color: {status_color}; font-weight: bold;">{status}</td>
                <td><pre style="font-size: 11px; max-height: 200px; overflow: auto;">{details}</pre></td>
            </tr>"""

        report_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ padding: 15px 25px; border-radius: 6px; text-align: center; }}
        .stat.total {{ background: #e3f2fd; }}
        .stat.passed {{ background: #e8f5e9; }}
        .stat.failed {{ background: #ffebee; }}
        .stat h3 {{ margin: 0; font-size: 28px; }}
        .stat p {{ margin: 5px 0 0 0; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #007bff; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
        tr:hover {{ background: #f8f9fa; }}
        .status-banner {{ padding: 15px; border-radius: 6px; text-align: center; font-size: 18px; font-weight: bold; margin: 20px 0;
            background: {"#d4edda" if overall_status == "PASSED" else "#f8d7da"};
            color: {"#155724" if overall_status == "PASSED" else "#721c24"}; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Data Quality Report</h1>
    <p>Generated: {timestamp}</p>
    <div class="status-banner">Overall Status: {overall_status}</div>
    <div class="summary">
        <div class="stat total"><h3>{total_checks}</h3><p>Total Checks</p></div>
        <div class="stat passed"><h3>{passed_checks}</h3><p>Passed</p></div>
        <div class="stat failed"><h3>{failed_checks}</h3><p>Failed</p></div>
    </div>
    <table>
        <thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead>
        <tbody>{html_rows}</tbody>
    </table>
</div>
</body>
</html>"""

        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(
                output_dir / f"quality_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            )

        with open(output_path, "w") as f:
            f.write(report_html)

        logger.info(f"Quality report generated: {output_path}")
        return output_path

    elif report_format == "json":
        report = {
            "timestamp": timestamp,
            "overall_status": overall_status,
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
            },
            "results": validation_results,
        }

        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(
                output_dir / f"quality_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Quality report generated: {output_path}")
        return output_path

    else:
        raise ValueError(f"Unsupported report format: {report_format}")
