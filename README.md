# Prefect Data Pipeline

Modern data orchestration using Prefect for healthcare claims processing. Implements a Bronze/Silver/Gold medallion architecture with native Python flows, automatic retries, intelligent caching, and built-in observability.

## Architecture

This pipeline follows the medallion (multi-hop) architecture pattern:

```
Source Systems --> Bronze (Raw) --> Silver (Cleaned) --> Gold (Aggregated)
                     |                  |                    |
              Raw Parquet +      Deduplicated,         Business-level
              ingestion         type-enforced,         aggregates &
              metadata          validated records      KPIs
```

### Pipeline Stages

- **Extract**: Pull raw claims data from source systems (CSV, API, database) with configurable retries and backoff
- **Bronze Ingestion**: Land raw data as Parquet files with ingestion metadata (timestamps, source lineage, batch IDs)
- **Silver Transform**: Clean, deduplicate, enforce schemas, and apply business rules to produce curated datasets
- **Gold Aggregate**: Compute business-level aggregates, KPIs, and derived metrics for analytics consumption
- **Quality Gates**: Data quality validation at each stage using Great Expectations-style checks

## Why Prefect Over Airflow

This project demonstrates Prefect as a modern alternative to Apache Airflow:

| Feature | Prefect | Airflow |
|---------|---------|---------|
| API Style | Native Python decorators | DAG definition files |
| Local Development | `python flow.py` | Docker Compose + scheduler |
| Retries | Built-in with backoff | Requires configuration |
| Caching | Native task caching | External caching needed |
| Async Support | First-class async/await | Limited async support |
| Dynamic Workflows | Native Python control flow | Complex with dynamic DAGs |

**Decision**: Airflow was chosen for the production deployment due to its mature ecosystem, enterprise support contracts, and existing infrastructure. Prefect remains a strong candidate for greenfield projects.

## Project Structure

```
prefect-data-pipeline/
├── flows/
│   ├── claims_pipeline.py       # Main ETL orchestration flow
│   └── feature_pipeline.py      # Feature engineering flow
├── tasks/
│   ├── transformations.py       # Reusable transformation tasks
│   └── quality.py               # Data quality validation tasks
├── configs/
│   └── config.yaml              # Pipeline configuration
├── data/
│   ├── raw/                     # Source data landing zone
│   ├── bronze/                  # Raw ingested Parquet files
│   ├── silver/                  # Cleaned and validated data
│   └── gold/                    # Business aggregates
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Locally

```bash
# Execute the claims pipeline
python flows/claims_pipeline.py

# Execute the feature pipeline
python flows/feature_pipeline.py

# Run with Prefect server for observability
prefect server start
python flows/claims_pipeline.py
```

### Deploy with Schedule

```bash
prefect deployment build flows/claims_pipeline.py:claims_etl_pipeline \
    --name "daily-claims-etl" \
    --cron "0 6 * * *" \
    --apply

prefect agent start -q default
```

## Configuration

Pipeline behavior is controlled via `configs/config.yaml`:

```yaml
prefect:
  flow_name: "claims_etl"
  retries: 2
  retry_delay: 300

quality:
  max_null_rate: 0.05
  min_row_count: 1000
  max_duplicate_rate: 0.01
```

## Data Quality

Quality gates enforce data integrity at every stage:

- **Schema Validation**: Column presence, data types, and constraints
- **Completeness Checks**: Null rate thresholds per column
- **Freshness Monitoring**: Data recency within expected windows
- **Distribution Validation**: Statistical distribution drift detection
- **Uniqueness Enforcement**: Duplicate rate thresholds with configurable dedup strategies

## Monitoring

Prefect provides native observability:

- **Flow Runs Dashboard**: Visual execution history with status tracking
- **Task-Level Metrics**: Duration, retries, cache hits per task
- **Notifications**: Slack/email alerts on failure or SLA breach
- **Artifact Tracking**: Quality reports stored as Prefect artifacts

## Tech Stack

- **Orchestration**: Prefect 2.x
- **Processing**: Pandas, NumPy
- **Storage**: Parquet (PyArrow)
- **Quality**: Great Expectations
- **Analytics DB**: DuckDB
- **HTTP Client**: httpx (async)
