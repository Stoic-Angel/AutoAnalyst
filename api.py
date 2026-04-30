
"""
FastAPI Backend - Connects React Frontend to Python PDF/CSV Processing
Run with: uvicorn api:app --reload --port 8000
"""

import os
import json
import shutil
import re
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from xtractor import HybridPDFExtractor
from identifier import HuggingFaceKPIDetector


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        ser = (
            out[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"^\$(.*)$", r"\1", regex=True)
        )
        coerced = pd.to_numeric(ser, errors="coerce")
        if coerced.notna().sum() > max(3, len(out) * 0.45):
            out[col] = coerced
    return out


def _looks_like_identifier(name: str, series: pd.Series) -> bool:
    n = len(series.dropna())
    if n == 0:
        return False
    name_l = name.lower()
    hints = ("id", "uuid", "sku", "key", "imdb", "ticker", "isin", "asin")
    if any(h in name_l for h in hints):
        return True
    return float(series.nunique()) / float(n) > 0.9


def _column_roles(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, datetime_cols: list) -> dict[str, str]:
    roles = {}
    for c in datetime_cols:
        roles[c] = "temporal"
    for c in numeric_cols:
        if c in roles:
            continue
        cl = c.lower()
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if (
            len(vals) >= 3
            and ("year" in cl or cl.endswith("_yr") or "released" in cl)
            and float(vals.min()) >= 1800
            and float(vals.max()) <= 2105
            and (((vals.round() - vals).abs()) < 1e-9).mean() > 0.75
        ):
            roles[c] = "temporal_marker"
            continue
        if (
            len(vals) >= max(10, len(df) * 0.25)
            and (((vals.round() - vals).abs()) < 1e-9).mean() > 0.88
            and float(vals.min()) >= 1850
            and float(vals.max()) <= 2105
            and float(vals.max()) - float(vals.min()) <= 260
            and ("year" in cl or "released" in cl or float(vals.max()) - float(vals.min()) <= 120)
        ):
            roles[c] = "temporal_marker"
            continue
        roles[c] = "metric_numeric"
    for c in categorical_cols:
        if _looks_like_identifier(c, df[c]):
            roles[c] = "identifier_or_key"
        else:
            roles[c] = "dimension"
    return roles


def _pick_dimension_numeric(numeric_cols: list, roles: dict[str, str]) -> tuple[str | None, str | None]:
    KEY = (
        "revenue", "profit", "income", "sales", "return", "eps", "ebitda", "rating", "score",
        "votes", "gross", "views", "subscriber", "volume", "margin", "price", "amount", "total",
    )
    skip = {"temporal", "temporal_marker"}
    cand = [c for c in numeric_cols if roles.get(c) not in skip]

    def key_fn(c):
        lc = c.lower()
        return (0 if any(k in lc for k in KEY) else 1, c)

    cand = sorted(set(cand), key=key_fn)
    if not cand:
        return None, None
    primary = cand[0]
    rest = [c for c in cand if c != primary][:1]
    return primary, (rest[0] if rest else None)



def pick_display_metric_for_aggregate(
    primary_metric: str | None,
    numeric_cols: list,
    roles: dict[str, str],
) -> tuple[str | None, bool]:
    if primary_metric and roles.get(primary_metric) not in {"temporal", "temporal_marker"}:
        return primary_metric, False
    for c in numeric_cols:
        if roles.get(c) == "metric_numeric":
            return c, False
    for c in numeric_cols:
        if roles.get(c) == "temporal_marker":
            return c, True
    return None, False


def _pick_dimension_categorical(df: pd.DataFrame, categorical_cols: list, roles: dict[str, str]) -> str | None:
    ranked = []
    for col in categorical_cols:
        if roles.get(col) != "dimension":
            continue
        vc = df[col].nunique()
        ranked.append((0 if 2 <= vc <= 80 else 1, vc, col))
    ranked.sort(key=lambda z: (z[0], z[1]))
    return ranked[0][2] if ranked else None




def _metric_behavior(col: str) -> str:
    cl = col.lower()
    if any(x in cl for x in ["revenue", "sales", "profit", "income", "subscriber", "user", "growth"]):
        return "higher_is_better"
    if any(x in cl for x in ["cost", "expense", "loss", "error", "churn"]):
        return "lower_is_better"
    return "neutral"


app = FastAPI(
    title="AutoAnalyst API",
    description="AI-powered PDF analysis and KPI detection",
    version="1.0.0"
)

# Enable CORS for React frontend (running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and extracted data
UPLOAD_DIR = Path("uploads")
EXTRACTED_DIR = Path("xtracted")
UPLOAD_DIR.mkdir(exist_ok=True)
EXTRACTED_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AutoAnalyst API is running"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or CSV file for processing.
    Returns extraction results and analysis.
    """
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are allowed")
    
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # Check file type and process accordingly
        if filename_lower.endswith('.csv'):
            return await process_csv(file_path, file.filename)
        else:
            return await process_pdf(file_path, file.filename)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if file_path.exists():
            os.remove(file_path)


async def process_pdf(file_path: Path, filename: str):
    """Process PDF files"""
    # Step 1: Extract data from PDF using xtractor.py
    extractor = HybridPDFExtractor(str(file_path))
    extracted_data = extractor.extract_all()
    extractor.save_results(str(EXTRACTED_DIR))
    
    # Step 2: Detect KPIs using identifier.py (with fallback)
    doc_type = "business report"
    detected_kpis = {"semantic_kpis": [], "entities": {}, "numerical_kpis": []}
    detector = None
    
    try:
        detector = HuggingFaceKPIDetector(str(EXTRACTED_DIR / "raw_data.json"))
        doc_type = detector.detect_document_type()
        detected_kpis = detector.extract_kpis_with_huggingface()
    except Exception as kpi_error:
        print(f"⚠️ KPI detection failed (using fallback): {kpi_error}")
        detected_kpis = {
            "semantic_kpis": [
                {"kpi": "Data Analysis", "relevance_score": 0.85, "category": "fallback"},
                {"kpi": "Key Metrics", "relevance_score": 0.75, "category": "fallback"},
                {"kpi": "Insights", "relevance_score": 0.70, "category": "fallback"}
            ],
            "entities": {},
            "numerical_kpis": []
        }
    
    response = {
        "success": True,
        "filename": filename,
        "metadata": extracted_data["metadata"],
        "extraction_info": extracted_data["extraction_info"],
        "document_type": doc_type,
        "kpis": detected_kpis,
        "charts": (chart_payload := prepare_chart_data(detector, extracted_data)),
        "analysis_context": chart_payload.get("analyze_context"),
    }
    
    return JSONResponse(content=response)


async def process_csv(file_path: Path, filename: str):
    print(f"📊 Processing CSV file: {filename}")
    
    df = pd.read_csv(file_path, low_memory=False)
    df = _coerce_numeric_columns(df)
    
    num_rows, num_cols = df.shape
    columns = df.columns.tolist()
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    numeric_stats = {}
    for col in numeric_cols[:5]:
        s = df[col]
        numeric_stats[col] = {
            "mean": float(s.mean()) if not pd.isna(s.mean()) else 0,
            "min": float(s.min()) if not pd.isna(s.min()) else 0,
            "max": float(s.max()) if not pd.isna(s.max()) else 0,
            "sum": float(s.sum()) if not pd.isna(s.sum()) else 0
        }
    
    charts = prepare_csv_charts(df)
    analysis_ctx = charts.get("analyze_context")

    roles = (analysis_ctx or {}).get("column_roles") or {}
    for col in list(numeric_stats.keys()):
        if roles.get(col) == "temporal_marker":
            numeric_stats[col].pop("sum", None)
            numeric_stats[col]["median"] = float(df[col].median())

    detected_kpis = {
        "semantic_kpis": [
            {"kpi": f"{num_cols} Columns", "relevance_score": 0.95, "category": "data"},
            {"kpi": f"{num_rows} Rows", "relevance_score": 0.90, "category": "data"},
            {"kpi": f"{len(numeric_cols)} Numeric Fields", "relevance_score": 0.85, "category": "data"}
        ],
        "entities": {},
        "numerical_kpis": [{"kpi": col, **stats} for col, stats in numeric_stats.items()]
    }
    
    response = {
        "success": True,
        "filename": filename,
        "metadata": {
            "rows": num_rows,
            "columns": num_cols,
            "column_names": columns[:10],
            "numeric_columns": numeric_cols[:8],
            "categorical_columns": categorical_cols[:8]
        },
        "extraction_info": {
            "total_rows": num_rows,
            "total_columns": num_cols,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols)
        },
        "document_type": "CSV Dataset",
        "kpis": detected_kpis,
        "charts": charts,
        "numeric_stats": numeric_stats,
        "analysis_context": analysis_ctx,
    }
    
    return JSONResponse(content=response)


class SmartDataAnalyzer:
    """
    Professional dashboard analyzer following best practices:
    1. KPIs with status indicators
    2. Time-based trends
    3. Distribution analysis
    4. Comparison layers
    5. Actionable insights
    6. Data tables with export
    """
    
    COLORS = [
        "rgba(99, 102, 241, 0.8)",   # Indigo
        "rgba(16, 185, 129, 0.8)",   # Emerald
        "rgba(245, 158, 11, 0.8)",   # Amber
        "rgba(239, 68, 68, 0.8)",    # Red
        "rgba(139, 92, 246, 0.8)",   # Purple
        "rgba(14, 165, 233, 0.8)",   # Sky
        "rgba(236, 72, 153, 0.8)",   # Pink
        "rgba(34, 197, 94, 0.8)",    # Green
    ]
    
    def __init__(self, df: pd.DataFrame):
        self.df = _coerce_numeric_columns(df)
        self.kpis = []
        self.trends = []
        self.distributions = []
        self.comparisons = []
        self.insights = []
        self.filters = []
        self.table_data = None
        self.column_roles = {}
        self.primary_metric = None
        self.secondary_metric = None
        self.dimension_col = None
        self.timeline_kind = "none"
        self.row_order_comparison = True
        self._analyze_columns()
    
    def _analyze_columns(self):
        self.numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.datetime_cols = []

        if self.categorical_cols:
            for col in list(self.categorical_cols):
                if self.df[col].dtype == bool:
                    self.df[col] = (
                        self.df[col].map({True: "true", False: "false"}).astype("object")
                    )
            self.categorical_cols = self.df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        parsed_temporal = []
        for col in list(self.categorical_cols):
            try:
                sample = self.df[col].dropna().head(min(140, len(self.df)))
                if len(sample) == 0:
                    continue
                parsed = pd.to_datetime(sample.astype(str), errors="coerce")
                if parsed.notna().mean() > 0.52:
                    parsed_temporal.append(col)
                    self.datetime_cols.append(col)
                    self.categorical_cols.remove(col)
            except Exception:
                pass

        self.column_roles = _column_roles(
            self.df, self.numeric_cols, self.categorical_cols, self.datetime_cols
        )

        parsed_first = parsed_temporal
        year_markers = [
            c
            for c in self.numeric_cols
            if self.column_roles.get(c) == "temporal_marker" and c not in self.datetime_cols
        ]

        dated = []
        for c in self.datetime_cols:
            if self.column_roles.get(c) == "temporal":
                dated.append(c)
        for c in parsed_first:
            if c not in dated:
                dated.append(c)

        merged = dated + year_markers
        seen = set()
        self.datetime_cols = []
        for c in merged:
            if c not in seen:
                seen.add(c)
                self.datetime_cols.append(c)

        explicit_date = len(dated) > 0
        explicit_year_numeric = len(year_markers) > 0
        self.timeline_kind = (
            "parsed_dates" if explicit_date else ("year_column" if explicit_year_numeric else "none")
        )
        self.row_order_comparison = self.timeline_kind == "none"

        self.primary_metric, self.secondary_metric = _pick_dimension_numeric(
            self.numeric_cols, self.column_roles
        )
        self.dimension_col = _pick_dimension_categorical(
            self.df, list(self.categorical_cols), self.column_roles
        )
    
    def analyze(self) -> dict:
        self._generate_kpis()
        self._generate_trends()
        self._generate_distributions()
        self._generate_comparisons()
        self._generate_insights()
        self._generate_filters()
        self._generate_table()
        summary = self._generate_summary()

        assumptions = []
        limitations = []
        if self.timeline_kind == "parsed_dates":
            assumptions.append(
                "Time series charts use detected date/datetime columns in their raw column order."
            )
        elif self.timeline_kind == "year_column":
            assumptions.append(
                "Trends aggregated by numeric year/release-year style columns."
            )
        elif self.timeline_kind == "none":
            limitations.append(
                "No trustworthy date/time column detected; sequential charts use dataset row order, not calendar time."
            )
        limitations.append(
            "First-half versus second-half statistics compare halves of whatever order the CSV was stored (sort beforehand for time)."
        )

        agg_hint_col, catalogs_like_counts_only = pick_display_metric_for_aggregate(
            self.primary_metric, self.numeric_cols, self.column_roles
        )
        if catalogs_like_counts_only:
            assumptions.append(
                "No additive business measure inferred; comparisons and stacked bars emphasize row volumes by category/time.",
            )

        metric_label = (
            self.primary_metric
            or agg_hint_col
            or (self.numeric_cols[0] if self.numeric_cols else None)
            or "unknown_metric"
        )
        dim_hint = (
            self.dimension_col
            or (self.categorical_cols[0] if self.categorical_cols else None)
        )
        if dim_hint:
            assumptions.append(f"Ranking and category visuals prefer dimension '{dim_hint}' over high-cardinality ID-like columns.")

        analyze_context = {
            "timeline_kind": self.timeline_kind,
            "row_order_only": bool(self.timeline_kind == "none"),
            "comparison_basis": ("by_time_or_year" if not self.row_order_comparison else "row_split_not_time"),
            "column_roles": {k: self.column_roles[k] for k in sorted(self.column_roles)},
            "primary_metric_hint": metric_label,
            "primary_dimension_hint": dim_hint,
            "assumptions": assumptions,
            "limitations": limitations,
        }

        return {
            "kpis": self.kpis,
            "trends": self.trends,
            "distributions": self.distributions,
            "comparisons": self.comparisons,
            "insights": self.insights,
            "filters": self.filters,
            "tableData": self.table_data,
            "allVisualizations": self.trends + self.distributions,
            "summary": summary,
            "analyze_context": analyze_context,
        }

    def _compare_label(self):
        return (
            "first vs second row chunk (CSV order)"
            if self.row_order_comparison
            else "time-aligned windows"
        )
    
    def _generate_kpis(self):
        """Generate KPI cards with status indicators"""
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        
        # Total Records KPI
        self.kpis.append({
            "label": "Total Records",
            "value": n_rows,
            "status": "good" if n_rows > 100 else "warning" if n_rows > 10 else "neutral"
        })
        
        ordered_metrics = []
        if self.primary_metric:
            ordered_metrics.append(self.primary_metric)
        if self.secondary_metric and self.secondary_metric not in ordered_metrics:
            ordered_metrics.append(self.secondary_metric)
        for c in self.numeric_cols:
            if self.column_roles.get(c) in {"temporal", "temporal_marker"}:
                continue
            if c not in ordered_metrics and len(ordered_metrics) < 8:
                ordered_metrics.append(c)
        metric_iter = ordered_metrics[:5]

        for col in [
            c for c in self.numeric_cols if self.column_roles.get(c) == "temporal_marker"
        ][:3]:
            data = self.df[col].dropna()
            if len(data) < 3:
                continue
            med = round(float(data.median()))
            self.kpis.append({
                "label": col[:18],
                "value": med,
                "formatHint": "year",
                "change": None,
                "changeType": "neutral",
                "status": "neutral",
                "sparkline": None,
                "description": "Median calendar year (not additive)",
            })

        for i, col in enumerate(metric_iter):
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            total = float(data.sum())
            mean = float(data.mean())
            
            sparkline = None
            change = None
            change_type = "neutral"
            status = "neutral"

            lbl = self._compare_label()
            
            if len(data) >= 6:
                mid = len(data) // 2
                period1 = data.iloc[:mid].mean()
                period2 = data.iloc[mid:].mean()
                
                if period1 != 0:
                    change = ((period2 - period1) / abs(period1)) * 100
                    change_type = "positive" if change > 0 else "negative"
                    bh = _metric_behavior(col)
                    if self.row_order_comparison:
                        status = "neutral"
                    elif bh == "higher_is_better":
                        status = "good" if change > 0 else "bad"
                    elif bh == "lower_is_better":
                        status = "bad" if change > 0 else "good"
                    else:
                        status = "neutral"
                
                if len(data) > 20:
                    step = len(data) // 20
                    sparkline = data.iloc[::step].head(20).tolist()
                else:
                    sparkline = data.tolist()[-20:]
            
            col_lower = col.lower()
            kpi = {
                "label": col[:18],
                "change": change,
                "changeType": change_type,
                "status": status,
                "sparkline": sparkline,
                "description": f"Δ mean between {lbl}",
            }
            
            lc = col_lower
            currency_like = any(
                x in lc for x in ["price", "cost", "revenue", "sales", "amount", "budget", "gross"]
            )
            deny_money = any(x in lc for x in ["rating", "score", "vote"])

            if currency_like and not deny_money:
                kpi["value"] = total
                kpi["prefix"] = "$"
            elif any(x in lc for x in ["rate", "percent", "ratio", "margin", "pct"]):
                kpi["value"] = mean
                kpi["suffix"] = "%"
            elif any(x in col_lower for x in ['count', 'qty', 'quantity', 'num', 'orders', 'users', 'votes']):
                kpi["value"] = int(total)
            else:
                kpi["value"] = round(mean, 4) if abs(mean) < 1e9 else round(mean, 2)
            
            self.kpis.append(kpi)
    
    def _generate_trends(self):
        if self.timeline_kind == "none" and self.numeric_cols and len(self.df) > 10:
            row_order_metrics = [
                c for c in self.numeric_cols
                if self.column_roles.get(c) != "temporal_marker"
            ]
            pm = self.primary_metric or (row_order_metrics[0] if row_order_metrics else None)
            if not pm:
                return
            data = self.df[pm].dropna()
            if len(data) > 4:
                if len(data) > 30:
                    step = len(data) // 30
                    sampled = data.iloc[::step].head(30).tolist()
                else:
                    sampled = data.tolist()
                note = (
                    "Values sampled by stored CSV row order; not inferred as calendar time."
                )
                self.trends.append({
                    "type": "line",
                    "title": f"{pm} (row-order sample)",
                    "description": note,
                    "data": {
                        "labels": list(range(1, len(sampled) + 1)),
                        "datasets": [{
                            "label": pm[:15],
                            "data": sampled,
                            "borderColor": self.COLORS[0],
                            "backgroundColor": self.COLORS[0].replace("0.8", "0.1"),
                            "tension": 0.4,
                            "fill": True
                        }]
                    }
                })
            return

        if not self.datetime_cols or not self.numeric_cols:
            return
        
        time_col = self.datetime_cols[0]
        value_candidates = [
            c for c in self.numeric_cols
            if c != time_col and self.column_roles.get(c) != "temporal_marker"
        ]
        prioritized = []
        for pivot in ([self.primary_metric, self.secondary_metric] + value_candidates):
            if pivot and pivot not in prioritized and pivot in self.df.columns and pivot != time_col:
                prioritized.append(pivot)
        value_cols = prioritized[:3] if prioritized else (
            value_candidates[:3] if value_candidates else []
        )
        value_cols = [c for c in value_cols if c != time_col]
        
        try:
            if value_cols:
                if time_col in self.df.select_dtypes(include=['number']).columns:
                    grouped = self.df.groupby(time_col)[value_cols].mean().sort_index()
                else:
                    temp_df = self.df.copy()
                    temp_df['_date'] = pd.to_datetime(temp_df[time_col], errors='coerce')
                    temp_df = temp_df.dropna(subset=['_date'])
                    if len(temp_df) == 0:
                        return
                    grouped = temp_df.groupby('_date')[value_cols].mean().sort_index()

                if len(grouped) < 2:
                    return

                labels = [str(x)[:10] for x in grouped.index.tolist()[-20:]]
                datasets = []
                for i, col in enumerate(value_cols):
                    if col in grouped.columns:
                        values = grouped[col].tolist()[-20:]
                        datasets.append({
                            "label": col[:15],
                            "data": values,
                            "borderColor": self.COLORS[i],
                            "backgroundColor": self.COLORS[i].replace("0.8", "0.1"),
                            "tension": 0.4,
                            "fill": i == 0
                        })

                if datasets:
                    self.trends.append({
                        "type": "line",
                        "title": f"Metrics over {time_col}",
                        "description": "Average of numeric measure(s) inside each bucket.",
                        "data": {"labels": labels, "datasets": datasets},
                    })
            else:
                if time_col in self.df.select_dtypes(include=['number']).columns:
                    counted = self.df.groupby(time_col).size().sort_index()
                else:
                    temp_df = self.df.copy()
                    temp_df['_date'] = pd.to_datetime(temp_df[time_col], errors='coerce')
                    temp_df = temp_df.dropna(subset=['_date'])
                    if len(temp_df) == 0:
                        return
                    counted = temp_df.groupby('_date').size()

                if len(counted) < 2:
                    return
                slice_tail = counted[-20:]
                self.trends.append({
                    "type": "line",
                    "title": f"Catalog volume vs {time_col}",
                    "description": "Counts rows per bucket (catalog-style file has no additive measure beyond headcount here).",
                    "data": {
                        "labels": [str(i)[:10] for i in slice_tail.index.tolist()],
                        "datasets": [{
                            "label": "Row count",
                            "data": slice_tail.astype(float).tolist(),
                            "borderColor": self.COLORS[0],
                            "backgroundColor": self.COLORS[0].replace("0.8", "0.1"),
                            "tension": 0.4,
                            "fill": True
                        }]
                    }
                })
        except Exception as e:
            print(f"Trend generation error: {e}")
    
    def _generate_distributions(self):
        dim_candidates = [
            c for c in self.categorical_cols
            if self.column_roles.get(c) == "dimension"
        ]
        for col in (dim_candidates[:3] or self.categorical_cols[:3]):
            value_counts = self.df[col].value_counts()
            
            if len(value_counts) < 2 or len(value_counts) > 15:
                continue
            
            top_n = value_counts.head(6)
            
            chart_type = "doughnut" if len(top_n) <= 5 else "bar"
            
            self.distributions.append({
                "type": chart_type,
                "title": f"By {col}",
                "description": f"Category frequency for {col}",
                "data": {
                    "labels": [str(l)[:18] for l in top_n.index.tolist()],
                    "datasets": [{
                        "label": "Count" if chart_type == "bar" else None,
                        "data": top_n.values.tolist(),
                        "backgroundColor": self.COLORS[:len(top_n)]
                    }]
                }
            })
        
        cat_col = (
            self.dimension_col
            if self.dimension_col
            else (dim_candidates[0] if dim_candidates else None)
        )
        agg_col, use_row_counts = pick_display_metric_for_aggregate(
            self.primary_metric, self.numeric_cols, self.column_roles
        )

        if cat_col and (agg_col is not None or use_row_counts):
            try:
                if use_row_counts:
                    grouped = self.df.groupby(cat_col).size().sort_values(ascending=False)
                    bar_note = "Row count per category"
                    bar_label = "Titles"
                else:
                    grouped = self.df.groupby(cat_col)[agg_col].sum().sort_values(ascending=False)
                    bar_note = f"Σ {agg_col} per category"
                    bar_label = agg_col[:22]
                if len(grouped) >= 2:
                    top_5 = grouped.head(5)
                    
                    self.distributions.append({
                        "type": "bar",
                        "title": f"Mix by {cat_col}",
                        "description": bar_note,
                        "data": {
                            "labels": [str(l)[:15] for l in top_5.index.tolist()],
                            "datasets": [{
                                "label": bar_label,
                                "data": top_5.values.tolist(),
                                "backgroundColor": self.COLORS[1]
                            }]
                        }
                    })
            except Exception:
                pass
    
    def _generate_comparisons(self):
        """Generate comparison analysis"""
        label = self._compare_label()
        focus_metrics = []
        if self.primary_metric:
            focus_metrics.append(self.primary_metric)
        if self.secondary_metric and self.secondary_metric not in focus_metrics:
            focus_metrics.append(self.secondary_metric)
        for col in self.numeric_cols:
            if self.column_roles.get(col) in {"temporal", "temporal_marker"}:
                continue
            if col not in focus_metrics and len(focus_metrics) < 6:
                focus_metrics.append(col)

        for col in focus_metrics[:2]:
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            
            mid = len(data) // 2
            period1 = data.iloc[:mid]
            period2 = data.iloc[mid:]
            
            current = float(period2.sum())
            previous = float(period1.sum())
            
            if previous != 0:
                change = ((current - previous) / abs(previous)) * 100
                
                self.comparisons.append({
                    "title": f"{col} — half-to-half ({label})",
                    "current": current,
                    "previous": previous,
                    "change": change
                })

        dim_candidates = [
            c for c in self.categorical_cols
            if self.column_roles.get(c) == "dimension"
        ]
        agg_col, use_row_counts_cmp = pick_display_metric_for_aggregate(
            self.primary_metric, self.numeric_cols, self.column_roles
        )
        cat_col = self.dimension_col or (dim_candidates[0] if dim_candidates else None)

        if cat_col and (agg_col is not None or use_row_counts_cmp):
            try:
                if use_row_counts_cmp:
                    grp = self.df.groupby(cat_col).size()
                    if len(grp) < 2 or len(grp) > 10:
                        raise ValueError()
                    total = grp.sum()
                    items = []
                    for idx in grp.index[:5]:
                        v = int(grp.loc[idx])
                        pct = (v / total * 100) if total else 0
                        items.append({
                            "label": str(idx)[:20],
                            "value": float(v),
                            "change": float(pct - (100 / len(grp))),
                        })
                    self.comparisons.append({
                        "title": f"Title mix by {cat_col}",
                        "items": items
                    })
                else:
                    grouped = self.df.groupby(cat_col)[agg_col].agg(['sum', 'mean', 'count'])
                    if len(grouped) >= 2 and len(grouped) <= 10:
                        items = []
                        total = grouped['sum'].sum()
                        
                        for idx in grouped.index[:5]:
                            row = grouped.loc[idx]
                            pct = (row['sum'] / total * 100) if total != 0 else 0
                            items.append({
                                "label": str(idx)[:20],
                                "value": float(row['sum']),
                                "change": float(pct - (100 / len(grouped))),
                            })
                        
                        self.comparisons.append({
                            "title": f"{agg_col} by {cat_col}",
                            "items": items
                        })
            except Exception:
                pass
    
    def _generate_insights(self):
        """Generate actionable insights with priority levels"""
        insights_list = []
        
        focus = []
        for c in ([self.primary_metric, self.secondary_metric] + self.numeric_cols):
            if c and c not in focus:
                focus.append(c)
            if len(focus) >= 6:
                break
        
        for col in focus[:4]:
            if self.column_roles.get(col) == "temporal_marker":
                continue
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            
            mid = len(data) // 2
            p1_mean = data.iloc[:mid].mean()
            p2_mean = data.iloc[mid:].mean()
            
            if p1_mean != 0:
                change = ((p2_mean - p1_mean) / abs(p1_mean)) * 100
                
                if abs(change) > 10:
                    direction = "increased" if change > 0 else "declined"
                    col_lower = col.lower()
                    
                    is_revenue_like = any(x in col_lower for x in ['revenue', 'sales', 'profit', 'income'])
                    is_cost_like = any(x in col_lower for x in ['cost', 'expense', 'loss'])

                    if self.row_order_comparison:
                        priority = "medium"
                        action = "Sort or filter by real time field before acting on this drift"
                        text_extra = f" {col} mean {direction} ~{abs(change):.1f}% between CSV halves (row order only)"
                    elif is_revenue_like:
                        priority = "low" if change > 0 else "high"
                        action = "Maintain current strategy" if change > 0 else "Investigate root cause"
                        text_extra = f"{col} has {direction} by {abs(change):.1f}% across windowed means"
                    elif is_cost_like:
                        priority = "high" if change > 0 else "low"
                        action = "Review cost drivers" if change > 0 else "Cost optimization working"
                        text_extra = f"{col} has {direction} by {abs(change):.1f}% across windowed means"
                    else:
                        priority = "medium"
                        action = "Monitor closely"
                        text_extra = f"{col} has {direction} by {abs(change):.1f}% across windowed means"

                    insights_list.append({
                        "icon": "📈" if change > 0 else "📉",
                        "text": text_extra,
                        "priority": priority,
                        "action": action
                    })
        
        # Distribution insights
        for col in self.categorical_cols[:2]:
            vc = self.df[col].value_counts()
            if len(vc) >= 2:
                top_pct = (vc.iloc[0] / vc.sum()) * 100
                if top_pct > 50:
                    insights_list.append({
                        "icon": "🎯",
                        "text": f"'{vc.index[0]}' dominates {col} at {top_pct:.0f}%",
                        "priority": "medium",
                        "action": "Consider diversification"
                    })
        
        # Data quality insights
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 5:
            insights_list.append({
                "icon": "⚠️",
                "text": f"Dataset has {missing_pct:.1f}% missing values",
                "priority": "high",
                "action": "Data cleanup recommended"
            })
        
        # Outlier detection (skip naive IQR wording on catalog years — long tails are usually real history)
        for col in self.numeric_cols[:3]:
            data = self.df[col].dropna()
            if len(data) > 20:
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
                outlier_pct = (outliers / len(data)) * 100

                role = self.column_roles.get(col)
                if role == "temporal_marker" and outlier_pct > 5:
                    insights_list.append({
                        "icon": "📅",
                        "text": f"{col} spans many eras ({outlier_pct:.0f}% of rows sit outside the middle 50% of years)",
                        "priority": "low",
                        "action": "Expected for long-running catalogs; not automatically a data defect",
                    })
                elif role != "temporal_marker" and outlier_pct > 5:
                    insights_list.append({
                        "icon": "🔍",
                        "text": f"{outliers} outliers detected in {col} ({outlier_pct:.1f}%)",
                        "priority": "medium",
                        "action": "Review for data errors"
                    })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights_list.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        self.insights = insights_list[:6]
    
    def _generate_filters(self):
        dims = [
            col for col in self.categorical_cols
            if self.column_roles.get(col) == "dimension"
        ]
        for col in dims[:3]:
            unique_vals = self.df[col].dropna().unique()
            if 2 <= len(unique_vals) <= 20:
                for val in unique_vals[:10]:
                    self.filters.append({
                        "label": f"{col}: {str(val)[:20]}",
                        "value": f"{col}:{val}"
                    })
    
    def _generate_table(self):
        """Generate complete data table"""
        # Select columns
        cols = []
        if self.categorical_cols:
            cols.extend(self.categorical_cols[:2])
        if self.numeric_cols:
            cols.extend(self.numeric_cols[:4])
        
        cols = cols[:8]  # Max 8 columns
        
        if cols:
            # Format values for display
            rows = []
            for _, row in self.df[cols].head(100).iterrows():
                formatted_row = []
                for val in row:
                    if pd.isna(val):
                        formatted_row.append("—")
                    elif isinstance(val, float):
                        formatted_row.append(round(val, 2))
                    else:
                        formatted_row.append(val)
                rows.append(formatted_row)
            
            self.table_data = {
                "title": "Data Explorer",
                "headers": [c[:20] for c in cols],
                "rows": rows
            }
    
    def _generate_summary(self) -> str:
        rows, cols = self.df.shape
        return f"Analyzed {rows:,} records across {cols} fields"


def prepare_csv_charts(df: pd.DataFrame) -> dict:
    """Generate professional dashboard from CSV data"""
    analyzer = SmartDataAnalyzer(df)
    return analyzer.analyze()


def prepare_chart_data(detector, extracted_data: dict) -> dict:
    """
    Convert PDF extracted data into professional dashboard format.
    """
    COLORS = [
        "rgba(99, 102, 241, 0.8)",
        "rgba(16, 185, 129, 0.8)",
        "rgba(245, 158, 11, 0.8)",
        "rgba(239, 68, 68, 0.8)",
        "rgba(139, 92, 246, 0.8)",
    ]
    
    # Extract metadata
    metadata = extracted_data.get('extraction_info', {})
    total_pages = metadata.get('total_pages', 0)
    total_tables = metadata.get('total_tables', 0)
    total_text = metadata.get('total_text_blocks', 0)
    total_images = metadata.get('total_images', 0)
    
    # Extract numbers and text from document
    all_numbers = []
    text_lengths = []
    
    if 'text' in extracted_data:
        for text_block in extracted_data['text']:
            content = text_block.get('content', '')
            text_lengths.append(len(content))
            
            numbers = re.findall(r'[\d,]+\.?\d*', content)
            for num_str in numbers:
                try:
                    num = float(num_str.replace(',', ''))
                    if 0 < num < 1e12:
                        all_numbers.append(num)
                except:
                    pass
    
    # === KPIs ===
    kpis = [
        {
            "label": "Total Pages",
            "value": total_pages,
            "status": "good" if total_pages > 0 else "neutral"
        },
        {
            "label": "Data Tables",
            "value": total_tables,
            "status": "good" if total_tables > 0 else "neutral"
        },
        {
            "label": "Text Sections",
            "value": total_text,
            "status": "good" if total_text > 0 else "warning"
        },
        {
            "label": "Images",
            "value": total_images,
            "status": "neutral"
        }
    ]
    
    if all_numbers:
        avg_val = sum(all_numbers) / len(all_numbers)
        kpis.append({
            "label": "Numbers Found",
            "value": len(all_numbers),
            "status": "good",
            "sparkline": sorted(all_numbers)[-20:] if len(all_numbers) > 5 else None
        })
        kpis.append({
            "label": "Avg Value",
            "value": avg_val,
            "status": "neutral"
        })
    
    # === TRENDS ===
    trends = []
    
    if text_lengths and len(text_lengths) > 2:
        trends.append({
            "type": "line",
            "title": "Content Density by Page",
            "description": "Character count across pages",
            "data": {
                "labels": [f"Page {i+1}" for i in range(min(len(text_lengths), 20))],
                "datasets": [{
                    "label": "Characters",
                    "data": text_lengths[:20],
                    "borderColor": COLORS[0],
                    "backgroundColor": COLORS[0].replace("0.8", "0.1"),
                    "tension": 0.4,
                    "fill": True
                }]
            }
        })
    
    if all_numbers and len(all_numbers) > 5:
        sorted_nums = sorted(all_numbers)[-30:]
        trends.append({
            "type": "line",
            "title": "Extracted Values Distribution",
            "description": f"{len(all_numbers)} numbers found",
            "data": {
                "labels": list(range(1, len(sorted_nums) + 1)),
                "datasets": [{
                    "label": "Value",
                    "data": sorted_nums,
                    "borderColor": COLORS[1],
                    "backgroundColor": COLORS[1].replace("0.8", "0.1"),
                    "tension": 0.4,
                    "fill": True
                }]
            }
        })
    
    # === DISTRIBUTIONS ===
    distributions = []
    
    composition = {k: v for k, v in [
        ("Tables", total_tables),
        ("Text Blocks", total_text),
        ("Images", total_images)
    ] if v > 0}
    
    if len(composition) > 1:
        distributions.append({
            "type": "doughnut",
            "title": "Document Composition",
            "description": "Content type breakdown",
            "data": {
                "labels": list(composition.keys()),
                "datasets": [{
                    "data": list(composition.values()),
                    "backgroundColor": COLORS[:len(composition)]
                }]
            }
        })
    
    if text_lengths and len(text_lengths) > 1:
        distributions.append({
            "type": "bar",
            "title": "Content by Page",
            "description": "Character distribution",
            "data": {
                "labels": [f"P{i+1}" for i in range(min(len(text_lengths), 10))],
                "datasets": [{
                    "label": "Chars",
                    "data": text_lengths[:10],
                    "backgroundColor": COLORS[0]
                }]
            }
        })
    
    # === COMPARISONS ===
    comparisons = []
    
    if len(text_lengths) >= 4:
        mid = len(text_lengths) // 2
        first_half = sum(text_lengths[:mid])
        second_half = sum(text_lengths[mid:])
        
        if first_half > 0:
            change = ((second_half - first_half) / first_half) * 100
            comparisons.append({
                "title": "Content Distribution",
                "current": second_half,
                "previous": first_half,
                "change": change
            })
    
    # === INSIGHTS ===
    insights = []
    
    # Document structure insight
    insights.append({
        "icon": "📄",
        "text": f"Document has {total_pages} pages with {total_text} text sections",
        "priority": "low",
        "action": None
    })
    
    if all_numbers:
        max_val = max(all_numbers)
        min_val = min(all_numbers)
        insights.append({
            "icon": "🔢",
            "text": f"Found {len(all_numbers)} numerical values ranging from {min_val:,.0f} to {max_val:,.0f}",
            "priority": "medium",
            "action": "Review for key metrics"
        })
    
    if total_tables > 0:
        insights.append({
            "icon": "📋",
            "text": f"Contains {total_tables} structured data tables",
            "priority": "medium",
            "action": "Tables may contain important data"
        })
    
    if total_images > 5:
        insights.append({
            "icon": "🖼️",
            "text": f"Document is image-heavy ({total_images} images)",
            "priority": "low",
            "action": None
        })
    
    # Content balance insight
    if text_lengths:
        avg_length = sum(text_lengths) / len(text_lengths)
        max_page = text_lengths.index(max(text_lengths)) + 1
        insights.append({
            "icon": "📊",
            "text": f"Page {max_page} has the most content ({max(text_lengths):,} chars)",
            "priority": "low",
            "action": "May contain key information"
        })
    
    # Enhance with detector data
    if detector:
        try:
            if hasattr(detector, 'numerical_data') and detector.numerical_data:
                pattern_stats = {}
                for item in detector.numerical_data:
                    pattern = item.get('pattern_type', 'general')
                    if pattern not in pattern_stats:
                        pattern_stats[pattern] = []
                    pattern_stats[pattern].append(item['value'])
                
                if len(pattern_stats) > 1:
                    distributions.append({
                        "type": "bar",
                        "title": "Value Patterns",
                        "description": "By detected pattern type",
                        "data": {
                            "labels": [p.title() for p in pattern_stats.keys()],
                            "datasets": [{
                                "label": "Avg Value",
                                "data": [sum(v)/len(v) if v else 0 for v in pattern_stats.values()],
                                "backgroundColor": COLORS[:len(pattern_stats)]
                            }]
                        }
                    })
                    
                    dominant_pattern = max(pattern_stats.keys(), key=lambda k: len(pattern_stats[k]))
                    insights.append({
                        "icon": "🎯",
                        "text": f"Most common pattern: {dominant_pattern} ({len(pattern_stats[dominant_pattern])} values)",
                        "priority": "medium",
                        "action": None
                    })
        except Exception as e:
            print(f"⚠️ Error enhancing charts: {e}")
    
    return {
        "kpis": kpis,
        "trends": trends,
        "distributions": distributions,
        "comparisons": comparisons,
        "insights": insights,
        "filters": [],
        "tableData": None,
        "allVisualizations": trends + distributions,
        "summary": f"Analyzed {total_pages} pages with {total_text} text blocks and {total_tables} tables",
        "analyze_context": {
            "source": "pdf_text_layout",
            "comparison_basis": "page_and_block_sequence",
            "limitations": [
                "Sequences follow PDF extraction order (pages/text blocks); they are not validated financial periods.",
                "Aggregated numeric scrapes can include headings, captions, axes, or page numbers unrelated to KPIs.",
            ],
            "assumptions": [
                "When structured tables exist, prefer opening them manually; automatic layout merge was not validated here.",
            ],
        },
    }


@app.get("/api/status")
async def get_status():
    """Get API status and available endpoints"""
    return {
        "status": "running",
        "endpoints": {
            "POST /api/upload": "Upload PDF for analysis",
            "GET /api/status": "Check API status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
