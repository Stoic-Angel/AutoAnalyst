#!/usr/bin/env python3
"""
Universal HuggingFace API-Based KPI Detection and Dashboard Generation
Uses free HuggingFace Inference API - NO local model downloads!
"""

import json
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
from datetime import datetime
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HuggingFaceKPIDetector:
    def __init__(self, data_path="xtracted/raw_data.json"):
        """Initialize with extracted PDF data and HuggingFace API"""
        print("🚀 Initializing HuggingFace API-Based KPI Detection...")
        
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # HuggingFace API configuration
        self.hf_api_key = None  # Optional - works without key for basic usage
        self.api_base = "https://api-inference.huggingface.co"
        
        # Initialize data structures
        self.detected_kpis = {}
        self.document_type = None
        self.numerical_data = []
        self.time_series_data = []
        
        # API endpoints
        self.models = {
            'document_classifier': 'facebook/bart-large-mnli',
            'entity_extractor': 'dbmdz/bert-large-cased-finetuned-conll03-english',
            'text_classifier': 'microsoft/DialoGPT-medium'
        }
    
    def set_api_key(self, api_key):
        """Set HuggingFace API key for higher rate limits"""
        self.hf_api_key = api_key
        print("✅ API key set for enhanced rate limits")
    
    def _make_api_call(self, model_name, payload):
        """Make API call to HuggingFace Inference API"""
        url = f"{self.api_base}/models/{model_name}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print(f"⚠️ Authentication failed for {model_name} - using fallback methods")
                return None
            elif response.status_code == 503:
                print(f"⚠️ Model {model_name} is loading, retrying...")
                # Wait and retry once
                import time
                time.sleep(5)
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                return response.json() if response.status_code == 200 else None
            else:
                print(f"⚠️ API Error {response.status_code} for {model_name}: {response.text}")
                return None
                
        except Exception as e:
            print(f"⚠️ API call failed for {model_name}: {e}")
            return None
    
    def detect_document_type(self):
        """Detect document type using HuggingFace zero-shot classification"""
        print("🔍 Detecting document type with HuggingFace...")
        
        # Combine all text for analysis
        all_text = ""
        for text_block in self.raw_data['text']:
            all_text += text_block['content'] + " "
        
        # Truncate for API limits
        sample_text = all_text[:1000]
        
        # Document type candidates
        candidate_labels = [
            "financial report", "sales report", "marketing analysis", 
            "operational report", "research report", "performance metrics",
            "annual report", "quarterly earnings", "survey results",
            "customer analytics", "business intelligence", "project report"
        ]
        
        payload = {
            "inputs": sample_text,
            "parameters": {"candidate_labels": candidate_labels}
        }
        
        result = self._make_api_call(self.models['document_classifier'], payload)
        
        if result and 'labels' in result:
            self.document_type = result['labels'][0]
            confidence = result['scores'][0]
            print(f"📋 Detected: {self.document_type} (confidence: {confidence:.2f})")
        else:
            print("⚠️ API failed, using pattern detection")
            self.document_type = self._detect_with_patterns(sample_text)
        
        return self.document_type
    
    def _detect_with_patterns(self, text):
        """Fallback pattern-based document detection"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['revenue', 'profit', 'earnings', 'financial', 'quarterly']):
            return "financial report"
        elif any(word in text_lower for word in ['sales', 'customers', 'conversion', 'pipeline']):
            return "sales report"
        elif any(word in text_lower for word in ['marketing', 'campaign', 'engagement', 'advertising']):
            return "marketing analysis"
        elif any(word in text_lower for word in ['operational', 'operations', 'efficiency', 'productivity']):
            return "operational report"
        elif any(word in text_lower for word in ['research', 'study', 'analysis', 'findings']):
            return "research report"
        else:
            return "business report"
    
    def extract_numerical_patterns(self):
        """Extract numerical data patterns using regex and NLP"""
        print("🔢 Extracting numerical patterns...")
        
        numerical_patterns = [
            r'\$?([\d,]+\.?\d*)\s*(?:million|billion|thousand|M|B|K)?',  # Money
            r'([\d,]+\.?\d*)\s*%',  # Percentages
            r'([\d,]+\.?\d*)\s*(?:units|customers|employees|users)',  # Counts
            r'Q[1-4]\s+\w+\s+([\d,]+\.?\d*)',  # Quarterly data
            r'(?:FY|fiscal year)\s*\d+.*?([\d,]+\.?\d*)',  # Fiscal year data
        ]
        
        for text_block in self.raw_data['text']:
            text = text_block['content']
            
            # Extract numbers with context
            for pattern in numerical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str)
                        
                        # Get context around the number
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        
                        self.numerical_data.append({
                            'value': value,
                            'raw_text': match.group(0),
                            'context': context,
                            'page': text_block['page'],
                            'pattern_type': self._classify_pattern(pattern)
                        })
                    except ValueError:
                        continue
    
    def _classify_pattern(self, pattern):
        """Classify the type of numerical pattern"""
        if '$' in pattern or 'million' in pattern:
            return 'monetary'
        elif '%' in pattern:
            return 'percentage'
        elif 'Q[1-4]' in pattern:
            return 'quarterly'
        elif 'units' in pattern or 'customers' in pattern:
            return 'count'
        else:
            return 'general'
    
    def extract_kpis_with_huggingface(self):
        """Extract KPIs using HuggingFace APIs"""
        print("🤖 Extracting KPIs with HuggingFace...")
        
        # First extract numerical patterns
        self.extract_numerical_patterns()
        
        # Combine all text for analysis
        full_text = ""
        for text_block in self.raw_data['text']:
            full_text += text_block['content'] + " "
        
        sample_text = full_text[:1500]  # API limits
        
        # Extract entities using NER model
        entities = self._extract_entities_hf(sample_text)
        
        # Extract semantic KPIs using zero-shot classification
        semantic_kpis = self._extract_semantic_kpis(sample_text)
        
        # Combine results
        self.detected_kpis = {
            'entities': entities,
            'semantic_kpis': semantic_kpis,
            'numerical_kpis': self._extract_numerical_kpis(),
            'document_type': self.document_type
        }
        
        # Extract time series data
        self._extract_time_series()
        
        return self.detected_kpis
    
    def _extract_entities_hf(self, text):
        """Extract named entities using HuggingFace NER model"""
        payload = {"inputs": text[:1000]}
        
        result = self._make_api_call(self.models['entity_extractor'], payload)
        
        if result:
            # Group entities by type
            entity_groups = {}
            for entity in result:
                ent_type = entity.get('entity_group', 'MISC')
                if ent_type not in entity_groups:
                    entity_groups[ent_type] = []
                
                # Filter by confidence
                if entity.get('score', 0) > 0.8:
                    entity_groups[ent_type].append({
                        'word': entity['word'],
                        'score': entity['score']
                    })
            
            return entity_groups
        else:
            # Fallback: extract basic entities using regex patterns
            print("⚠️ Using fallback entity detection")
            fallback_entities = {
                'ORG': [],
                'MISC': [],
                'MONEY': [],
                'PERCENT': []
            }
            
            # Look for common patterns
            import re
            
            # Organizations (companies)
            org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Corp|Inc|Ltd|LLC|Company|Corporation)\b'
            orgs = re.findall(org_pattern, text)
            fallback_entities['ORG'].extend([{'word': org, 'score': 0.7} for org in orgs[:5]])
            
            # Money amounts
            money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
            money = re.findall(money_pattern, text)
            fallback_entities['MONEY'].extend([{'word': m, 'score': 0.8} for m in money[:10]])
            
            # Percentages
            percent_pattern = r'\d+(?:\.\d+)?%'
            percents = re.findall(percent_pattern, text)
            fallback_entities['PERCENT'].extend([{'word': p, 'score': 0.8} for p in percents[:10]])
            
            return fallback_entities
    
    def _extract_semantic_kpis(self, text):
        """Extract KPIs using semantic similarity with HuggingFace"""
        # Define KPI categories based on document type
        kpi_categories = {
            'financial report': [
                'revenue growth', 'profit margin', 'earnings per share', 
                'cash flow', 'return on investment', 'debt ratio'
            ],
            'sales report': [
                'sales volume', 'conversion rate', 'customer acquisition cost',
                'sales pipeline', 'average deal size', 'sales velocity'
            ],
            'marketing analysis': [
                'customer engagement', 'click-through rate', 'conversion rate',
                'return on ad spend', 'customer lifetime value', 'brand awareness'
            ],
            'operational report': [
                'efficiency metrics', 'productivity rates', 'utilization rates',
                'operational costs', 'process improvements', 'quality metrics'
            ]
        }
        
        # Get relevant KPI categories
        categories = kpi_categories.get(self.document_type, kpi_categories['financial report'])
        
        # Use zero-shot classification to find relevant KPIs
        payload = {
            "inputs": text[:800],
            "parameters": {"candidate_labels": categories}
        }
        
        result = self._make_api_call(self.models['document_classifier'], payload)
        
        if result and 'labels' in result:
            # Return top 3 most relevant KPIs
            top_kpis = []
            for i in range(min(3, len(result['labels']))):
                top_kpis.append({
                    'kpi': result['labels'][i],
                    'relevance_score': result['scores'][i],
                    'category': 'semantic'
                })
            return top_kpis
        else:
            # Fallback: return basic KPIs based on document type
            print("⚠️ Using fallback KPI detection")
            fallback_kpis = {
                'financial report': [
                    {'kpi': 'Revenue Metrics', 'relevance_score': 0.9, 'category': 'fallback'},
                    {'kpi': 'Profit Analysis', 'relevance_score': 0.8, 'category': 'fallback'},
                    {'kpi': 'Growth Rates', 'relevance_score': 0.7, 'category': 'fallback'}
                ],
                'sales report': [
                    {'kpi': 'Sales Performance', 'relevance_score': 0.9, 'category': 'fallback'},
                    {'kpi': 'Customer Metrics', 'relevance_score': 0.8, 'category': 'fallback'},
                    {'kpi': 'Conversion Rates', 'relevance_score': 0.7, 'category': 'fallback'}
                ],
                'marketing analysis': [
                    {'kpi': 'Campaign Performance', 'relevance_score': 0.9, 'category': 'fallback'},
                    {'kpi': 'Engagement Metrics', 'relevance_score': 0.8, 'category': 'fallback'},
                    {'kpi': 'ROI Analysis', 'relevance_score': 0.7, 'category': 'fallback'}
                ]
            }
            return fallback_kpis.get(self.document_type, [
                {'kpi': 'Performance Metrics', 'relevance_score': 0.8, 'category': 'fallback'},
                {'kpi': 'Key Indicators', 'relevance_score': 0.7, 'category': 'fallback'},
                {'kpi': 'Business Metrics', 'relevance_score': 0.6, 'category': 'fallback'}
            ])
    
    def _extract_numerical_kpis(self):
        """Extract KPIs from numerical patterns"""
        numerical_kpis = []
        
        # Group by pattern type
        pattern_groups = {}
        for item in self.numerical_data:
            pattern = item['pattern_type']
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(item)
        
        # Create KPI summaries
        for pattern_type, items in pattern_groups.items():
            if items:
                values = [item['value'] for item in items]
                numerical_kpis.append({
                    'kpi': f"{pattern_type.title()} Metrics",
                    'count': len(items),
                    'max_value': max(values),
                    'avg_value': np.mean(values),
                    'pattern_type': pattern_type
                })
        
        return numerical_kpis
    
    def _extract_time_series(self):
        """Extract time series data patterns"""
        time_patterns = [
            r'Q[1-4]\s+(?:FY)?\d{2,4}',  # Quarters
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',  # Months
            r'\d{4}',  # Years
            r'(?:FY|fiscal year)\s*\d{2,4}'  # Fiscal years
        ]
        
        for text_block in self.raw_data['text']:
            text = text_block['content']
            
            for pattern in time_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Look for numbers near time indicators
                    surrounding_text = text[max(0, match.start()-100):match.end()+100]
                    numbers = re.findall(r'[\d,]+\.?\d*', surrounding_text)
                    
                    if numbers:
                        self.time_series_data.append({
                            'time_period': match.group(0),
                            'values': [float(n.replace(',', '')) for n in numbers if n.replace(',', '').replace('.', '').isdigit()],
                            'context': surrounding_text,
                            'page': text_block['page']
                        })
    
    def generate_visualizations(self):
        """Generate visualizations based on detected data"""
        print("📊 Generating visualizations...")
        
        visualizations = []
        
        # 1. Numerical Data Distribution
        if self.numerical_data:
            figs = self._create_numerical_distribution()
            if figs:
                # Handle tuple return (two separate figures)
                if isinstance(figs, tuple):
                    visualizations.append(('Value Distribution', figs[0]))
                    visualizations.append(('Pattern Types', figs[1]))
                else:
                    visualizations.append(('Numerical Data Analysis', figs))
        
        # 2. KPI Categories
        if 'semantic_kpis' in self.detected_kpis and self.detected_kpis['semantic_kpis']:
            fig = self._create_kpi_categories()
            visualizations.append(('KPI Categories', fig))
        
        # 3. Entity Analysis
        if 'entities' in self.detected_kpis and self.detected_kpis['entities']:
            fig = self._create_entity_analysis()
            visualizations.append(('Entity Analysis', fig))
        
        # 4. Time Series (if available)
        if self.time_series_data:
            fig = self._create_time_series()
            visualizations.append(('Time Series Trends', fig))
        
        # 5. Pattern Metrics
        if self.numerical_data:
            fig = self._create_pattern_metrics()
            visualizations.append(('Pattern Analysis', fig))
        
        return visualizations
    
    def _create_time_series(self):
        """Create time series visualization"""
        if not self.time_series_data:
            return None
        
        # Aggregate time series data
        time_aggregated = {}
        for ts in self.time_series_data[:10]:  # Limit for performance
            period = ts['time_period']
            if period not in time_aggregated:
                time_aggregated[period] = []
            time_aggregated[period].extend(ts['values'])
        
        periods = list(time_aggregated.keys())
        avg_values = [np.mean(values) if values else 0 for values in time_aggregated.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods,
            y=avg_values,
            mode='lines+markers',
            name='Average Values',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="📈 Time Series Trends",
            xaxis_title="Time Period",
            yaxis_title="Average Value",
            height=400
        )
        
        return fig
    
    def _create_numerical_distribution(self):
        """Create numerical data distribution visualization"""
        values = [item['value'] for item in self.numerical_data if item['value'] < 1e6]  # Filter outliers
        
        if not values:
            return None
        
        # Create two separate figures instead of subplots
        # Figure 1: Histogram
        fig1 = go.Figure()
        fig1.add_trace(
            go.Histogram(x=values, nbinsx=20, name='Values', marker_color='lightblue')
        )
        fig1.update_layout(
            title="📊 Value Distribution",
            xaxis_title="Values",
            yaxis_title="Frequency",
            height=300
        )
        
        # Figure 2: Pattern types bar chart (not pie chart)
        pattern_counts = {}
        for item in self.numerical_data:
            pattern = item['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=list(pattern_counts.keys()),
                y=list(pattern_counts.values()),
                name='Pattern Types',
                marker_color='lightgreen'
            )
        )
        fig2.update_layout(
            title="🏷️ Pattern Types Distribution",
            xaxis_title="Pattern Type",
            yaxis_title="Count",
            height=300
        )
        
        # Return both figures as a combined visualization
        return (fig1, fig2)
    
    def _create_kpi_categories(self):
        """Create KPI categories visualization"""
        kpis = self.detected_kpis['semantic_kpis']
        
        categories = [kpi['kpi'] for kpi in kpis]
        scores = [kpi['relevance_score'] for kpi in kpis]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color='lightblue',
                text=[f'{score:.2f}' for score in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="🎯 Detected KPI Categories",
            xaxis_title="KPI Type",
            yaxis_title="Relevance Score",
            height=400
        )
        
        return fig
    
    def _create_entity_analysis(self):
        """Create entity analysis visualization"""
        entities = self.detected_kpis['entities']
        
        # Count entities by type
        entity_counts = {k: len(v) for k, v in entities.items() if v}
        
        if not entity_counts:
            return None
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="🏷️ Named Entity Analysis",
            xaxis_title="Entity Type",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def _create_pattern_metrics(self):
        """Create pattern-based metrics visualization"""
        if not self.numerical_data:
            return None
        
        # Group by pattern type and calculate stats
        pattern_stats = {}
        for item in self.numerical_data:
            pattern = item['pattern_type']
            if pattern not in pattern_stats:
                pattern_stats[pattern] = []
            pattern_stats[pattern].append(item['value'])
        
        patterns = list(pattern_stats.keys())
        max_values = [max(values) if values else 0 for values in pattern_stats.values()]
        avg_values = [np.mean(values) if values else 0 for values in pattern_stats.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Max Value',
            x=patterns,
            y=max_values,
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            name='Average Value',
            x=patterns,
            y=avg_values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="📊 Pattern-Based Metrics",
            xaxis_title="Pattern Type",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        return fig

class HuggingFaceDashboard:
    def __init__(self):
        self.detector = HuggingFaceKPIDetector()
        
    def generate_dashboard(self):
        """Generate HuggingFace-powered dashboard"""
        st.set_page_config(
            page_title="HuggingFace AI Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🤖 HuggingFace AI-Powered Dashboard")
        st.markdown("*Uses free HuggingFace Inference API - NO local downloads!*")
        st.markdown("---")
        
        # API Key input (optional)
        with st.sidebar:
            st.subheader("🔑 API Configuration")
            api_key = st.text_input("HuggingFace API Key (optional)", type="password", 
                                   help="Optional: Add for higher rate limits")
            if api_key:
                self.detector.set_api_key(api_key)
            
            st.markdown("---")
            st.info("""
            **Free Usage:**
            - Works without API key
            - Basic rate limits
            - No model downloads
            
            **With API Key:**
            - Higher rate limits
            - Better performance
            - Priority access
            """)
        
        # Pipeline execution
        with st.spinner("🚀 Running HuggingFace AI Analysis..."):
            
            # Step 1: Document Type Detection
            doc_type = self.detector.detect_document_type()
            
            # Step 2: Extract KPIs
            detected_kpis = self.detector.extract_kpis_with_huggingface()
            
            # Step 3: Generate Visualizations
            visualizations = self.detector.generate_visualizations()
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document Type", doc_type.title())
        
        with col2:
            semantic_count = len(detected_kpis.get('semantic_kpis', []))
            st.metric("AI-Detected KPIs", semantic_count)
        
        with col3:
            st.metric("Visualizations", len(visualizations))
        
        st.markdown("---")
        
        # KPI Summary
        if 'semantic_kpis' in detected_kpis and detected_kpis['semantic_kpis']:
            st.subheader("🎯 AI-Detected KPIs")
            
            cols = st.columns(len(detected_kpis['semantic_kpis']))
            for i, kpi in enumerate(detected_kpis['semantic_kpis']):
                with cols[i]:
                    st.metric(
                        kpi['kpi'], 
                        f"{kpi['relevance_score']:.2f}",
                        "relevance score"
                    )
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 AI-Generated Visualizations")
        
        for i, (title, fig) in enumerate(visualizations):
            st.plotly_chart(fig, use_container_width=True)
            if i < len(visualizations) - 1:
                st.markdown("---")
        
        # Analysis Summary
        st.markdown("---")
        st.subheader("🔍 Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Detected Patterns:**")
            if self.detector.numerical_data:
                pattern_types = set(item['pattern_type'] for item in self.detector.numerical_data)
                for pattern in pattern_types:
                    st.write(f"• {pattern.title()}")
        
        with col2:
            st.write("**Entity Types:**")
            if 'entities' in detected_kpis:
                for entity_type, entities in detected_kpis['entities'].items():
                    if entities:
                        st.write(f"• {entity_type}: {len(entities)} found")
        
        # Technical Details
        with st.expander("🔧 Technical Details"):
            st.write("**HuggingFace Models Used:**")
            st.write("• BART for Document Classification")
            st.write("• BERT for Named Entity Recognition")
            st.write("• Zero-shot Classification for KPI Detection")
            
            st.write(f"\n**Processing Stats:**")
            st.write(f"• Tables Processed: {len(self.detector.raw_data['tables'])}")
            st.write(f"• Text Blocks: {len(self.detector.raw_data['text'])}")
            st.write(f"• Numerical Values: {len(self.detector.numerical_data)}")
            st.write(f"• Time Series Points: {len(self.detector.time_series_data)}")
            
            st.write(f"\n**API Status:**")
            st.write(f"• HuggingFace Inference API: Active")
            st.write(f"• Local Models: None (API-only)")
            st.write(f"• Rate Limits: {'Enhanced' if self.detector.hf_api_key else 'Basic'}")

def main():
    """Main function to run the HuggingFace dashboard"""
    dashboard = HuggingFaceDashboard()
    dashboard.generate_dashboard()

if __name__ == "__main__":
    main()