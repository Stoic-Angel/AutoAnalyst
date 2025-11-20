# AI-Powered Automatic Data Analytics Website

A fully offline, privacy-preserving, intelligent system that transforms static PDF documents into structured insights, computed KPIs, and interactive dashboards using AI, OCR, and automated analytics.

## 🚀 Overview

The AI-Powered Automatic Data Analytics Website is a complete end-to-end system designed to extract, analyze, and visualize information from PDF documents without requiring any cloud services. The framework automates the entire analytics pipeline—PDF preprocessing, text/table extraction, KPI identification, statistical computation, and dashboard generation—executing completely on the user’s local machine to ensure confidentiality and security.

This project demonstrates how artificial intelligence, document understanding, and automation can work together to enable real-time, domain-agnostic data analysis.

## 🔍 Key Features

Completely Offline & Privacy-Safe
All processing runs locally—no data leaves the system.

Automatic PDF Parsing
Uses PyMuPDF for layout parsing and Tesseract OCR for text extraction in scanned/non-digital PDFs.

AI-Driven Interpretation
Transformer-based local LLMs (via Ollama) interpret extracted text and automatically identify KPIs.

Automated Data Processing Pipeline
KPI extraction, normalization, and computation using Pandas and NumPy.

Dynamic Dashboard Generation
Interactive charts and summaries built with Plotly, Matplotlib, and Streamlit.

Domain Agnostic
Works for finance, healthcare, education, HR, research, and general document analytics.

Modular Architecture
Each stage (OCR, LLM analysis, KPI extraction, visualization) functions as an independent module.

## 🛠️ Tech Stack

Languages & Frameworks:

Python

Streamlit

JavaScript (optional, for UI enhancements)

AI & NLP:

Transformer-based LLMs via Ollama

Tesseract OCR

PyMuPDF

Data Processing:

Pandas

NumPy

Visualization:

Plotly

Matplotlib

## 📥 Installation

1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Install OCR & LLM Tools

Install Tesseract → https://github.com/tesseract-ocr/tesseract

Install Ollama → https://ollama.com/download

Pull your chosen LLM:

ollama pull llama3

## ▶️ Usage
Run the Website
streamlit run app.py

Steps

Upload any PDF document.

The system extracts text, tables, and KPIs automatically.

It processes the data and performs statistical analysis.

An interactive dashboard is generated instantly.

## 📊 Output Dashboard Includes

KPI summaries

Bar charts, line charts, pie charts

Statistical breakdowns

Data tables and extracted insights

📁 Project Structure
├── app.py  
├── modules/
│   ├── pdf_parser.py
│   ├── ocr_engine.py
│   ├── kpi_extractor.py
│   ├── data_processor.py
│   ├── dashboard_generator.py
├── assets/
├── requirements.txt
└── README.md

## 🧪 Testing

Run unit tests:

pytest

## 🌐 Applications

Finance – Invoice analysis, financial reports

Healthcare – Medical record analytics

Education – Academic report analysis

HR – Resume/KPI extraction

Research – Paper summarization & data extraction

## 🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.
