
import zipfile
import xml.etree.ElementTree as ET
import os

def extract_text_from_docx(docx_path, output_path):
    print(f"Processing: {docx_path}")
    if not os.path.exists(docx_path):
        print("Error: File not found.")
        return

    try:
        with zipfile.ZipFile(docx_path) as zf:
            xml_content = zf.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            
            # Extract text from paragraphs
            for p in tree.iterfind('.//w:p', namespaces):
                texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            
            markdown_content = "# Extracted Thesis Content\n\n" + '\n\n'.join(paragraphs)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
            print(f"Success! Extracted {len(paragraphs)} paragraphs to {output_path}")

    except Exception as e:
        print(f"Failed to extract text: {e}")

if __name__ == "__main__":
    file_path = r"c:\Users\HOUSE\Desktop\InfractiVision\Visión computacional de cruces en rojo para mejorar el proceso de registro de infracciones de tránsito en Trujillo 2025.docx"
    output_path = r"c:\Users\HOUSE\Desktop\InfractiVision\thesis_extracted.md"
    extract_text_from_docx(file_path, output_path)
