import os
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
import json
from datetime import datetime



class HybridPDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.extracted_data = {
            'metadata': {},
            'tables': [],
            'text': [],
            'images': [],
            'extraction_info': {}
        }
    
    def extract_with_pdfplumber(self):
        """Extract tables and layout information using PDFplumber"""
        print("🔍 Extracting tables and layout with PDFplumber...")
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Extract metadata
                self.extracted_data['metadata'] = {
                    'pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', 'Unknown'),
                    'author': pdf.metadata.get('Author', 'Unknown'),
                    'subject': pdf.metadata.get('Subject', 'Unknown'),
                    'creator': pdf.metadata.get('Creator', 'Unknown')
                }
                
                # Extract tables from each page
                for page_num, page in enumerate(pdf.pages):
                    print(f"  📄 Processing page {page_num + 1}...")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and any(any(cell for cell in row) for row in table):
                            table_info = {
                                'page': page_num + 1,
                                'table_index': table_idx + 1,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0,
                                'data': table
                            }
                            self.extracted_data['tables'].append(table_info)
                    
                    # Extract text with positioning
                    text = page.extract_text()
                    if text and text.strip():
                        self.extracted_data['text'].append({
                            'page': page_num + 1,
                            'content': text.strip(),
                            'length': len(text)
                        })
                
                print(f"  ✅ Extracted {len(self.extracted_data['tables'])} tables and {len(self.extracted_data['text'])} text blocks")
                
        except Exception as e:
            print(f"  ❌ Error with PDFplumber: {e}")
    
    def extract_with_pymupdf(self):
        """Extract text, images, and additional data using PyMuPDF"""
        print("🔍 Extracting text and images with PyMuPDF...")
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  📄 Processing page {page_num + 1}...")
                
                # Extract text (PyMuPDF often has better text extraction)
                text = page.get_text()
                if text and text.strip():
                    # Check if we already have text from this page
                    existing_text = next((t for t in self.extracted_data['text'] if t['page'] == page_num + 1), None)
                    if existing_text:
                        # Update existing text if PyMuPDF version is better
                        if len(text.strip()) > len(existing_text['content']):
                            existing_text['content'] = text.strip()
                            existing_text['length'] = len(text.strip())
                    else:
                        # Add new text block
                        self.extracted_data['text'].append({
                            'page': page_num + 1,
                            'content': text.strip(),
                            'length': len(text.strip())
                        })
                
                # Extract images
                images = page.get_images()
                for img_idx, img in enumerate(images):
                    img_info = {
                        'page': page_num + 1,
                        'image_index': img_idx + 1,
                        'width': img[2],
                        'height': img[3],
                        'format': img[1],
                        'size_bytes': img[4],
                        'bbox': img[5] if len(img) > 5 else None
                    }
                    self.extracted_data['images'].append(img_info)
            
            doc.close()
            print(f"  ✅ Extracted {len(self.extracted_data['images'])} images")
            
        except Exception as e:
            print(f"  ❌ Error with PyMuPDF: {e}")
    
    def extract_all(self):
        """Run the complete hybrid extraction process"""
        print(f"🚀 Starting hybrid PDF extraction for: {self.pdf_path}")
        print("=" * 60)
        
        # Extract with PDFplumber first (tables and layout)
        self.extract_with_pdfplumber()
        
        # Extract with PyMuPDF (text and images)
        self.extract_with_pymupdf()
        
        # Add extraction metadata
        self.extracted_data['extraction_info'] = {
            'timestamp': datetime.now().isoformat(),
            'total_tables': len(self.extracted_data['tables']),
            'total_text_blocks': len(self.extracted_data['text']),
            'total_images': len(self.extracted_data['images']),
            'total_pages': self.extracted_data['metadata'].get('pages', 0)
        }
        
        print("=" * 60)
        print(f"🎉 Extraction complete!")
        print(f"   📊 Tables: {len(self.extracted_data['tables'])}")
        print(f"   📝 Text blocks: {len(self.extracted_data['text'])}")
        print(f"   🖼️  Images: {len(self.extracted_data['images'])}")
        print(f"   📄 Pages: {self.extracted_data['metadata'].get('pages', 0)}")
        
        return self.extracted_data
    
    def save_results(self, output_dir="xtracted"):
        """Save extracted data to organized text files"""
        print(f"\n💾 Saving results to '{output_dir}' folder...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save tables
        if self.extracted_data['tables']:
            tables_file = Path(output_dir) / "tables.txt"
            with open(tables_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(" EXTRACTED TABLES\n")
                f.write("=" * 80 + "\n\n")
                
                for table in self.extracted_data['tables']:
                    f.write(f"Table {table['table_index']} (Page {table['page']})\n")
                    f.write(f"Dimensions: {table['rows']} rows × {table['columns']} columns\n")
                    f.write("-" * 50 + "\n")
                    
                    for row in table['data']:
                        f.write(" | ".join(str(cell) if cell else "" for cell in row) + "\n")
                    f.write("\n" + "=" * 80 + "\n\n")
            
            print(f"  ✅ Tables saved to: {tables_file}")
        
        # Save text
        if self.extracted_data['text']:
            text_file = Path(output_dir) / "text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("📝 EXTRACTED TEXT\n")
                f.write("=" * 80 + "\n\n")
                
                for text_block in self.extracted_data['text']:
                    f.write(f"Page {text_block['page']} (Length: {text_block['length']} chars)\n")
                    f.write("-" * 50 + "\n")
                    f.write(text_block['content'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"  ✅ Text saved to: {text_file}")
        
        # Save images info
        if self.extracted_data['images']:
            images_file = Path(output_dir) / "images.txt"
            with open(images_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("️  EXTRACTED IMAGES\n")
                f.write("=" * 80 + "\n\n")
                
                for img in self.extracted_data['images']:
                    f.write(f"Image {img['image_index']} (Page {img['page']})\n")
                    f.write(f"Format: {img['format']}\n")
                    f.write(f"Dimensions: {img['width']} × {img['height']}\n")
                    f.write(f"Size: {img['size_bytes']} bytes\n")
                    if img['bbox']:
                        f.write(f"Bounding Box: {img['bbox']}\n")
                    f.write("-" * 50 + "\n\n")
            
            print(f"  ✅ Image info saved to: {images_file}")
        
        # Save metadata and summary
        summary_file = Path(output_dir) / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" EXTRACTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Document Metadata:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.extracted_data['metadata'].items():
                f.write(f"{key.title()}: {value}\n")
            
            f.write("\nExtraction Results:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Tables: {len(self.extracted_data['tables'])}\n")
            f.write(f"Total Text Blocks: {len(self.extracted_data['text'])}\n")
            f.write(f"Total Images: {len(self.extracted_data['images'])}\n")
            f.write(f"Total Pages: {self.extracted_data['metadata'].get('pages', 0)}\n")
            
            f.write(f"\nExtraction Timestamp: {self.extracted_data['extraction_info']['timestamp']}\n")
        
        print(f"  ✅ Summary saved to: {summary_file}")
        
        # Save raw JSON data
        json_file = Path(output_dir) / "raw_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Raw JSON data saved to: {json_file}")
        print(f"\n🎯 All results saved to '{output_dir}' folder!")

def main():
    """Main function to run the PDF extraction"""
    pdf_path = "samples/NVIDIAAn.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Create extractor and run extraction
    extractor = HybridPDFExtractor(pdf_path)
    extracted_data = extractor.extract_all()
    
    # Save results
    extractor.save_results()
    
    print("\n🎉 PDF extraction completed successfully!")
    print("Check the 'extracted' folder for all results.")

if __name__ == "__main__":
    main()
