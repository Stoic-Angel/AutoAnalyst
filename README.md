# AI-Powered Automatic Data Analytics Website

A fully offline, privacy-preserving system that transforms PDF documents into structured insights, computed KPIs, and interactive dashboards using AI, OCR, and automated analytics.

## 🚀 Overview

This end-to-end system automates the entire analytics pipeline—PDF preprocessing, text/table extraction, KPI identification, statistical computation, and dashboard generation—all running locally on your machine to ensure confidentiality and security.

## 🔍 Key Features

- **Completely Offline & Privacy-Safe** - All processing runs locally; no data leaves your system
- **Automatic PDF Parsing** - Uses PyMuPDF for layout parsing and Tesseract OCR for scanned documents
- **AI-Driven Interpretation** - Transformer-based local LLMs (via Ollama) automatically identify KPIs
- **Automated Data Processing** - KPI extraction, normalization, and computation using Pandas and NumPy
- **Dynamic Dashboard Generation** - Interactive charts and summaries with Plotly, Matplotlib, and Streamlit
- **Domain Agnostic** - Works for finance, healthcare, education, HR, research, and general document analytics
- **Modular Architecture** - Each stage (OCR, LLM analysis, KPI extraction, visualization) functions independently

## 🛠️ Tech Stack

**Languages & Frameworks:** Python, Streamlit, JavaScript (optional)

**AI & NLP:** Transformer-based LLMs via Ollama, Tesseract OCR, PyMuPDF

**Data Processing:** Pandas, NumPy

**Visualization:** Plotly, Matplotlib

## 📥 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Mac/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OCR & LLM Tools**
   - Install [Tesseract](https://github.com/tesseract-ocr/tesseract)
   - Install [Ollama](https://ollama.com/download)
   - Pull your chosen LLM: `ollama pull llama3`

## ▶️ Usage

Run the application:
```bash
streamlit run identifier.py
```

**Workflow:**
1. Upload any PDF document
2. The system automatically extracts text, tables, and KPIs
3. Data is processed and statistical analysis is performed
4. An interactive dashboard is generated with:
   - KPI summaries
   - Bar charts, line charts, pie charts
   - Statistical breakdowns
   - Data tables and extracted insights

## 📁 Project Structure

```
├── identifier.py  
├── xtractor.py
├── modules/
│   ├── pdf_parser.py
│   ├── ocr_engine.py
│   ├── kpi_extractor.py
│   ├── data_processor.py
│   ├── dashboard_generator.py
├── assets/
├── samples/
├── xtracted/
├── src/
├── requirements.txt
└── README.md
```

## 🌐 Applications

- **Finance** – Invoice analysis, financial reports
- **Healthcare** – Medical record analytics
- **Education** – Academic report analysis
- **HR** – Resume/KPI extraction
- **Research** – Paper summarization & data extraction

## 🧪 Testing

Run unit tests:
```bash
pytest
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
