# document_processor.py
import os
import PyPDF2 # For PDF
from docx import Document # For DOCX

def read_text_file(filepath):
    """Reads content from a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {filepath}: {e}")
        return None

def read_pdf_file(filepath):
    """Reads content from a PDF file."""
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Handle empty pages
        return text
    except Exception as e:
        print(f"Error reading PDF file {filepath}: {e}")
        return None

def read_docx_file(filepath):
    """Reads content from a DOCX file."""
    text = ""
    try:
        doc = Document(filepath)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX file {filepath}: {e}")
        return None

def get_file_content(filepath, mime_type):
    """Determines file type and extracts content."""
    if mime_type == 'text/plain':
        return read_text_file(filepath)
    elif mime_type == 'application/pdf':
        return read_pdf_file(filepath)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return read_docx_file(filepath)
    else:
        print(f"Unsupported file type for content extraction: {mime_type} for {os.path.basename(filepath)}")
        return None

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits a long text into smaller chunks with overlap."""
    if not text:
        return []

    chunks = []
    current_position = 0
    while current_position < len(text):
        end_position = min(current_position + chunk_size, len(text))
        chunk = text[current_position:end_position]
        chunks.append(chunk)
        if end_position == len(text):
            break
        current_position += (chunk_size - chunk_overlap)
        # Ensure current_position doesn't go negative
        if current_position < 0:
            current_position = 0

    return chunks