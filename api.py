
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
    
    # Step 3: Prepare response for React frontend
    response = {
        "success": True,
        "filename": filename,
        "metadata": extracted_data["metadata"],
        "extraction_info": extracted_data["extraction_info"],
        "document_type": doc_type,
        "kpis": detected_kpis,
        "charts": prepare_chart_data(detector, extracted_data)
    }
    
    return JSONResponse(content=response)


async def process_csv(file_path: Path, filename: str):
    """Process CSV files"""
    print(f"📊 Processing CSV file: {filename}")
    
    # Read CSV with pandas
    df = pd.read_csv(file_path)
    
    # Get basic info
    num_rows, num_cols = df.shape
    columns = df.columns.tolist()
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculate statistics for numeric columns
    numeric_stats = {}
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        numeric_stats[col] = {
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
            "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else 0
        }
    
    # Create chart data
    charts = prepare_csv_charts(df, numeric_cols, categorical_cols)
    
    # Create KPIs from data
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
            "column_names": columns[:10],  # First 10 column names
            "numeric_columns": numeric_cols[:5],
            "categorical_columns": categorical_cols[:5]
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
        "numeric_stats": numeric_stats
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
        self.df = df
        self.kpis = []
        self.trends = []
        self.distributions = []
        self.comparisons = []
        self.insights = []
        self.filters = []
        self.table_data = None
        self._analyze_columns()
    
    def _analyze_columns(self):
        """Categorize columns by data type"""
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = []
        
        # Detect date columns
        for col in self.categorical_cols[:]:
            try:
                sample = self.df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() > len(sample) * 0.5:
                        self.datetime_cols.append(col)
                        self.categorical_cols.remove(col)
            except:
                pass
        
        # Check for year/date columns in numeric
        for col in self.numeric_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['year', 'date', 'month', 'quarter', 'week']):
                vals = self.df[col].dropna()
                if len(vals) > 0 and 'year' in col_lower:
                    if vals.min() > 1900 and vals.max() < 2100:
                        self.datetime_cols.append(col)
    
    def analyze(self) -> dict:
        """Run complete analysis"""
        self._generate_kpis()
        self._generate_trends()
        self._generate_distributions()
        self._generate_comparisons()
        self._generate_insights()
        self._generate_filters()
        self._generate_table()
        
        return {
            "kpis": self.kpis,
            "trends": self.trends,
            "distributions": self.distributions,
            "comparisons": self.comparisons,
            "insights": self.insights,
            "filters": self.filters,
            "tableData": self.table_data,
            "allVisualizations": self.trends + self.distributions,
            "summary": self._generate_summary()
        }
    
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
        
        # Analyze each numeric column for KPIs
        for i, col in enumerate(self.numeric_cols[:5]):
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            total = float(data.sum())
            mean = float(data.mean())
            median = float(data.median())
            
            # Calculate period-over-period change
            sparkline = None
            change = None
            change_type = "neutral"
            status = "neutral"
            
            if len(data) >= 6:
                # Split into periods for comparison
                mid = len(data) // 2
                period1 = data.iloc[:mid].mean()
                period2 = data.iloc[mid:].mean()
                
                if period1 != 0:
                    change = ((period2 - period1) / abs(period1)) * 100
                    change_type = "positive" if change > 0 else "negative"
                    
                    # Determine status based on column semantics
                    col_lower = col.lower()
                    if any(x in col_lower for x in ['revenue', 'sales', 'profit', 'income']):
                        status = "good" if change > 0 else "bad"
                    elif any(x in col_lower for x in ['cost', 'expense', 'loss', 'error']):
                        status = "bad" if change > 0 else "good"
                    else:
                        status = "good" if change > 5 else "bad" if change < -5 else "neutral"
                
                # Create sparkline
                if len(data) > 20:
                    step = len(data) // 20
                    sparkline = data.iloc[::step].head(20).tolist()
                else:
                    sparkline = data.tolist()[-20:]
            
            # Determine display format
            col_lower = col.lower()
            kpi = {
                "label": col[:18],
                "change": change,
                "changeType": change_type,
                "status": status,
                "sparkline": sparkline,
                "description": f"vs previous period"
            }
            
            # Format value based on column type
            if any(x in col_lower for x in ['price', 'cost', 'revenue', 'sales', 'amount', 'value', 'total']):
                kpi["value"] = total
                kpi["prefix"] = "$"
            elif any(x in col_lower for x in ['rate', 'percent', 'ratio', 'margin', 'pct']):
                kpi["value"] = mean
                kpi["suffix"] = "%"
            elif any(x in col_lower for x in ['count', 'qty', 'quantity', 'num', 'orders', 'users']):
                kpi["value"] = int(total)
            else:
                kpi["value"] = mean if abs(mean) < 1000 else total
            
            self.kpis.append(kpi)
    
    def _generate_trends(self):
        """Generate time-based trend visualizations"""
        if not self.datetime_cols or not self.numeric_cols:
            # Create sequential trend if no date column
            if self.numeric_cols and len(self.df) > 10:
                col = self.numeric_cols[0]
                data = self.df[col].dropna()
                
                if len(data) > 30:
                    step = len(data) // 30
                    sampled = data.iloc[::step].head(30).tolist()
                else:
                    sampled = data.tolist()
                
                self.trends.append({
                    "type": "line",
                    "title": f"{col} Trend",
                    "description": "Sequential pattern analysis",
                    "data": {
                        "labels": list(range(1, len(sampled) + 1)),
                        "datasets": [{
                            "label": col[:15],
                            "data": sampled,
                            "borderColor": self.COLORS[0],
                            "backgroundColor": self.COLORS[0].replace("0.8", "0.1"),
                            "tension": 0.4,
                            "fill": True
                        }]
                    }
                })
            return
        
        time_col = self.datetime_cols[0]
        value_cols = self.numeric_cols[:3]
        
        try:
            # Group by time period
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
            
            # Multi-metric trend
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
                    "title": f"Performance Over {time_col}",
                    "description": "Key metrics trend over time",
                    "data": {"labels": labels, "datasets": datasets}
                })
        except Exception as e:
            print(f"Trend generation error: {e}")
    
    def _generate_distributions(self):
        """Generate distribution/breakdown charts"""
        # Category distributions
        for col in self.categorical_cols[:3]:
            value_counts = self.df[col].value_counts()
            
            if len(value_counts) < 2 or len(value_counts) > 15:
                continue
            
            top_n = value_counts.head(6)
            
            chart_type = "doughnut" if len(top_n) <= 5 else "bar"
            
            self.distributions.append({
                "type": chart_type,
                "title": f"By {col}",
                "description": f"Distribution across {col} categories",
                "data": {
                    "labels": [str(l)[:18] for l in top_n.index.tolist()],
                    "datasets": [{
                        "label": "Count" if chart_type == "bar" else None,
                        "data": top_n.values.tolist(),
                        "backgroundColor": self.COLORS[:len(top_n)]
                    }]
                }
            })
        
        # Top performers if we have category + numeric
        if self.categorical_cols and self.numeric_cols:
            cat_col = self.categorical_cols[0]
            num_col = self.numeric_cols[0]
            
            try:
                grouped = self.df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                if len(grouped) >= 2:
                    top_5 = grouped.head(5)
                    
                    self.distributions.append({
                        "type": "bar",
                        "title": f"Top {cat_col}",
                        "description": f"Ranked by {num_col}",
                        "data": {
                            "labels": [str(l)[:15] for l in top_5.index.tolist()],
                            "datasets": [{
                                "label": num_col[:15],
                                "data": top_5.values.tolist(),
                                "backgroundColor": self.COLORS[1]
                            }]
                        }
                    })
            except:
                pass
    
    def _generate_comparisons(self):
        """Generate comparison analysis"""
        # Period-over-period comparison for numeric columns
        for col in self.numeric_cols[:2]:
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
                    "title": f"{col} - Period Comparison",
                    "current": current,
                    "previous": previous,
                    "change": change
                })
        
        # Category comparison if available
        if self.categorical_cols and len(self.numeric_cols) >= 1:
            cat_col = self.categorical_cols[0]
            num_col = self.numeric_cols[0]
            
            try:
                grouped = self.df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count'])
                if len(grouped) >= 2 and len(grouped) <= 10:
                    items = []
                    total = grouped['sum'].sum()
                    
                    for idx in grouped.index[:5]:
                        row = grouped.loc[idx]
                        pct = (row['sum'] / total * 100) if total != 0 else 0
                        items.append({
                            "label": str(idx)[:20],
                            "value": float(row['sum']),
                            "change": pct - (100 / len(grouped))  # vs equal distribution
                        })
                    
                    self.comparisons.append({
                        "title": f"{num_col} by {cat_col}",
                        "items": items
                    })
            except:
                pass
    
    def _generate_insights(self):
        """Generate actionable insights with priority levels"""
        insights_list = []
        
        # Trend insights
        for col in self.numeric_cols[:3]:
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
                    
                    # Determine if this is good or bad
                    is_revenue_like = any(x in col_lower for x in ['revenue', 'sales', 'profit', 'income'])
                    is_cost_like = any(x in col_lower for x in ['cost', 'expense', 'loss'])
                    
                    if is_revenue_like:
                        priority = "low" if change > 0 else "high"
                        action = "Maintain current strategy" if change > 0 else "Investigate root cause"
                    elif is_cost_like:
                        priority = "high" if change > 0 else "low"
                        action = "Review cost drivers" if change > 0 else "Cost optimization working"
                    else:
                        priority = "medium"
                        action = "Monitor closely"
                    
                    insights_list.append({
                        "icon": "📈" if change > 0 else "📉",
                        "text": f"{col} has {direction} by {abs(change):.1f}%",
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
        
        # Outlier detection
        for col in self.numeric_cols[:2]:
            data = self.df[col].dropna()
            if len(data) > 20:
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
                outlier_pct = (outliers / len(data)) * 100
                
                if outlier_pct > 5:
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
        """Generate filter options for drill-down"""
        for col in self.categorical_cols[:3]:
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


def prepare_csv_charts(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
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
        "summary": f"Analyzed {total_pages} pages with {total_text} text blocks and {total_tables} tables"
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
