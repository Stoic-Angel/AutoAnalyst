import json
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class UniversalKPIDetector:
    def __init__(self, data_path="xtracted/raw_data.json"):
        """Initialize with extracted PDF data and load transformer models"""
        print("🚀 Initializing Universal KPI Detection Pipeline...")
        
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # Load open-source models
        self._load_models()
        
        # Initialize data structures
        self.detected_kpis = {}
        self.document_type = None
        self.numerical_data = []
        self.categorical_data = []
        self.time_series_data = []
        
    def _load_models(self):
        """Load open-source transformer models for various NLP tasks"""
        print("📥 Loading transformer models...")
        
        try:
            # 1. Named Entity Recognition (for extracting entities)
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # 2. Text Classification (for document type detection)
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium"  # Fallback to a general model
            )
            
            # 3. Sentence Transformer (for semantic similarity)
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 4. Zero-shot classification for KPI categorization
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Model loading fallback: {e}")
            # Fallback to basic NLP
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load lightweight fallback models if transformers fail"""
        print("📥 Loading fallback models...")
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️ Installing spacy English model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def detect_document_type(self):
        """Automatically detect document type using transformer models"""
        print("🔍 Detecting document type...")
        
        # Combine all text for analysis
        all_text = ""
        for text_block in self.raw_data['text']:
            all_text += text_block['content'] + " "
        
        # Truncate for model limits
        all_text = all_text[:1000]
        
        # Predefined document types for zero-shot classification
        candidate_labels = [
            "financial report", "sales report", "marketing analysis", 
            "operational report", "research report", "performance metrics",
            "annual report", "quarterly earnings", "survey results",
            "customer analytics", "business intelligence", "project report"
        ]
        
        try:
            result = self.zero_shot_classifier(all_text, candidate_labels)
            self.document_type = result['labels'][0]
            confidence = result['scores'][0]
            
            print(f"📋 Detected document type: {self.document_type} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"⚠️ Fallback document type detection: {e}")
            self._fallback_document_detection(all_text)
        
        return self.document_type
    
    def _fallback_document_detection(self, text):
        """Fallback document type detection using keyword matching"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['revenue', 'profit', 'earnings', 'financial']):
            self.document_type = "financial report"
        elif any(word in text_lower for word in ['sales', 'customers', 'conversion']):
            self.document_type = "sales report"
        elif any(word in text_lower for word in ['marketing', 'campaign', 'engagement']):
            self.document_type = "marketing analysis"
        else:
            self.document_type = "business report"
    
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
    
    def detect_kpis_with_ai(self):
        """Use AI models to detect and categorize KPIs"""
        print("🤖 Detecting KPIs with AI models...")
        
        # Combine all text for AI analysis
        full_text = ""
        for text_block in self.raw_data['text']:
            full_text += text_block['content'] + " "
        
        # Extract entities using NER
        try:
            entities = self.ner_pipeline(full_text[:2000])  # Limit for model
            
            # Group entities by type
            entity_groups = {}
            for entity in entities:
                ent_type = entity['entity_group']
                if ent_type not in entity_groups:
                    entity_groups[ent_type] = []
                entity_groups[ent_type].append(entity['word'])
            
            self.detected_kpis['entities'] = entity_groups
            
        except Exception as e:
            print(f"⚠️ Entity detection failed: {e}")
            self._fallback_entity_detection(full_text)
        
        # Detect KPI categories based on document type
        self._categorize_kpis_by_domain()
        
        # Extract time series data
        self._extract_time_series()
        
        return self.detected_kpis
    
    def _fallback_entity_detection(self, text):
        """Fallback entity detection using spaCy"""
        try:
            doc = self.nlp(text[:2000])
            entities = {
                'PERSON': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
                'ORG': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
                'MONEY': [ent.text for ent in doc.ents if ent.label_ == 'MONEY'],
                'PERCENT': [ent.text for ent in doc.ents if ent.label_ == 'PERCENT'],
                'DATE': [ent.text for ent in doc.ents if ent.label_ == 'DATE']
            }
            self.detected_kpis['entities'] = entities
        except:
            self.detected_kpis['entities'] = {}
    
    def _categorize_kpis_by_domain(self):
        """Categorize KPIs based on detected document type"""
        domain_kpis = {
            'financial report': ['revenue', 'profit', 'margin', 'earnings', 'cash flow', 'growth'],
            'sales report': ['sales volume', 'conversion rate', 'customer acquisition', 'revenue'],
            'marketing analysis': ['engagement', 'reach', 'conversions', 'cost per acquisition'],
            'operational report': ['efficiency', 'productivity', 'utilization', 'performance'],
            'research report': ['results', 'findings', 'statistics', 'correlations'],
            'business report': ['performance', 'metrics', 'results', 'trends']
        }
        
        relevant_kpis = domain_kpis.get(self.document_type, domain_kpis['business report'])
        
        # Find KPIs in text using semantic similarity
        kpi_scores = {}
        for text_block in self.raw_data['text']:
            sentences = text_block['content'].split('.')
            
            for sentence in sentences[:10]:  # Limit for performance
                if len(sentence.strip()) > 20:  # Skip short sentences
                    try:
                        # Calculate similarity with relevant KPIs
                        sentence_embedding = self.sentence_model.encode([sentence.strip()])
                        kpi_embeddings = self.sentence_model.encode(relevant_kpis)
                        
                        similarities = np.dot(sentence_embedding, kpi_embeddings.T)[0]
                        max_similarity = np.max(similarities)
                        
                        if max_similarity > 0.3:  # Threshold for relevance
                            best_kpi = relevant_kpis[np.argmax(similarities)]
                            if best_kpi not in kpi_scores:
                                kpi_scores[best_kpi] = []
                            kpi_scores[best_kpi].append({
                                'sentence': sentence.strip(),
                                'score': max_similarity,
                                'page': text_block['page']
                            })
                    except:
                        continue
        
        self.detected_kpis['semantic_kpis'] = kpi_scores
    
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
    
    def generate_adaptive_visualizations(self):
        """Generate visualizations based on detected data patterns"""
        print("📊 Generating adaptive visualizations...")
        
        visualizations = []
        
        # 1. Time Series Visualization (if time series data exists)
        if self.time_series_data:
            fig = self._create_time_series_viz()
            if fig:
                visualizations.append(('Time Series Analysis', fig))
        
        # 2. Numerical Distribution
        if self.numerical_data:
            fig = self._create_numerical_distribution()
            visualizations.append(('Numerical Data Distribution', fig))
        
        # 3. KPI Categories (if semantic KPIs found)
        if 'semantic_kpis' in self.detected_kpis and self.detected_kpis['semantic_kpis']:
            fig = self._create_kpi_categories_viz()
            visualizations.append(('KPI Categories', fig))
        
        # 4. Entity Analysis
        if 'entities' in self.detected_kpis and self.detected_kpis['entities']:
            fig = self._create_entity_analysis()
            visualizations.append(('Entity Analysis', fig))
        
        # 5. Pattern-based Metrics
        fig = self._create_pattern_metrics()
        if fig:
            visualizations.append(('Pattern Analysis', fig))
        
        return visualizations
    
    def _create_time_series_viz(self):
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
        
        # Create visualization
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
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Value Distribution', 'Pattern Types']
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=values, nbinsx=20, name='Values'),
            row=1, col=1
        )
        
        # Pattern types pie chart
        pattern_counts = {}
        for item in self.numerical_data:
            pattern = item['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(pattern_counts.keys()),
                values=list(pattern_counts.values()),
                name='Pattern Types'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="🔢 Numerical Data Analysis",
            height=400
        )
        
        return fig
    
    def _create_kpi_categories_viz(self):
        """Create KPI categories visualization"""
        kpi_data = self.detected_kpis['semantic_kpis']
        
        categories = list(kpi_data.keys())
        counts = [len(items) for items in kpi_data.values()]
        avg_scores = [np.mean([item['score'] for item in items]) for items in kpi_data.values()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['KPI Frequency', 'Average Relevance Scores']
        )
        
        # Frequency bar chart
        fig.add_trace(
            go.Bar(x=categories, y=counts, name='Frequency'),
            row=1, col=1
        )
        
        # Relevance scores
        fig.add_trace(
            go.Bar(x=categories, y=avg_scores, name='Avg Score', marker_color='orange'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="🎯 Detected KPI Categories",
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

class UniversalDashboard:
    def __init__(self):
        self.detector = UniversalKPIDetector()
        
    def generate_dashboard(self):
        """Generate universal automated dashboard"""
        st.set_page_config(
            page_title="Universal AI Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🤖 Universal AI-Powered Dashboard")
        st.markdown("*Automatically detects KPIs and generates visualizations from any document type*")
        st.markdown("---")
        
        # Pipeline execution
        with st.spinner("🚀 Running AI Analysis Pipeline..."):
            
            # Step 1: Document Type Detection
            doc_type = self.detector.detect_document_type()
            
            # Step 2: Extract Numerical Patterns
            self.detector.extract_numerical_patterns()
            
            # Step 3: AI-Powered KPI Detection
            detected_kpis = self.detector.detect_kpis_with_ai()
            
            # Step 4: Generate Adaptive Visualizations
            visualizations = self.detector.generate_adaptive_visualizations()
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document Type", doc_type.title())
        
        with col2:
            st.metric("Detected KPIs", len(detected_kpis.get('semantic_kpis', {})))
        
        with col3:
            st.metric("Visualizations", len(visualizations))
        
        st.markdown("---")
        
        # KPI Summary
        st.subheader("🎯 Detected KPIs")
        
        if 'semantic_kpis' in detected_kpis and detected_kpis['semantic_kpis']:
            cols = st.columns(len(detected_kpis['semantic_kpis']))
            for i, (kpi, data) in enumerate(detected_kpis['semantic_kpis'].items()):
                with cols[i]:
                    avg_score = np.mean([item['score'] for item in data])
                    st.metric(kpi.title(), f"{len(data)} mentions", f"{avg_score:.2f} relevance")
        
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
            st.write("**AI Models Used:**")
            st.write("• BERT for Named Entity Recognition")
            st.write("• BART for Zero-shot Classification")
            st.write("• SentenceTransformers for Semantic Similarity")
            st.write("• Custom Pattern Recognition")
            
            st.write(f"\n**Processing Stats:**")
            st.write(f"• Tables Processed: {len(self.detector.raw_data['tables'])}")
            st.write(f"• Text Blocks: {len(self.detector.raw_data['text'])}")
            st.write(f"• Numerical Values Found: {len(self.detector.numerical_data)}")
            st.write(f"• Time Series Points: {len(self.detector.time_series_data)}")

def main():
    """Main function to run the universal dashboard"""
    dashboard = UniversalDashboard()
    dashboard.generate_dashboard()

if __name__ == "__main__":
    main()