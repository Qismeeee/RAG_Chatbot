import os
from pdf2image import convert_from_path
import pytesseract
from google.cloud import vision
import io
import json
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)

# Tesseract OCR and Google Cloud API Key
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../key.json"


def extract_text_from_image_google(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response.full_text_annotation.text if response.full_text_annotation else ""


def extract_text_from_image_tesseract(image):
    return pytesseract.image_to_string(image, lang='vie')


def save_text_to_txt(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


def save_text_to_json(text, metadata, output_path):
    data = {"content": text, "metadata": metadata}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def langchain_document_loader(TMP_DIR):
    documents = []
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    txt_loader = DirectoryLoader(
        TMP_DIR, glob="*.txt", loader_cls=TextLoader, show_progress=True)
    documents.extend(txt_loader.load())

    pdf_files = [os.path.join(TMP_DIR, f)
                 for f in os.listdir(TMP_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        try:
            pdf_loader = PyPDFLoader(pdf_file)
            pdf_documents = pdf_loader.load()
            if pdf_documents and any(doc.page_content for doc in pdf_documents):
                documents.extend(pdf_documents)
                print(f"Đã xử lý PDF thường: {os.path.basename(pdf_file)}")
                for i, doc in enumerate(pdf_documents, start=1):
                    output_path = os.path.join(
                        output_dir, f"{base_filename}_page_{i}.txt")
                    save_text_to_txt(doc.page_content, output_path)
            else:
                raise ValueError("PDF có thể là dạng scan")
        except Exception:
            print(
                f"Đã phát hiện PDF dạng scan, sử dụng OCR: {os.path.basename(pdf_file)}")
            pages = convert_from_path(pdf_file)
            for page_number, page in enumerate(pages, start=1):
                temp_image_path = f"temp_page_{page_number}.png"
                page.save(temp_image_path, "PNG")

                try:
                    text = extract_text_from_image_google(temp_image_path)
                    if not text:
                        text = extract_text_from_image_tesseract(page)
                except Exception as e:
                    print(f"Lỗi với Google OCR, chuyển sang Tesseract: {e}")
                    text = extract_text_from_image_tesseract(page)

                metadata = {"source": os.path.basename(
                    pdf_file), "page_number": page_number}
                documents.append({
                    "page_content": text,
                    "metadata": metadata
                })

                output_path_json = os.path.join(
                    output_dir, f"{base_filename}_page_{page_number}.json")
                save_text_to_json(text, metadata, output_path_json)

                os.remove(temp_image_path)

    excel_loader = DirectoryLoader(
        TMP_DIR, glob="*.xlsx", loader_cls=UnstructuredExcelLoader, show_progress=True)
    documents.extend(excel_loader.load())
    docx_loader = DirectoryLoader(
        TMP_DIR, glob="*.docx", loader_cls=Docx2txtLoader, show_progress=True)
    documents.extend(docx_loader.load())

    return documents

if __name__ == "__main__":
    TMP_DIR = "../data"
    documents = langchain_document_loader(TMP_DIR)

    for doc in documents:
        if isinstance(doc, dict):
            print("Nguồn:", doc['metadata']['source'])
            print("Số trang:", doc['metadata'].get('page_number', 'N/A'))
            print("Nội dung:\n", doc['page_content'])
        else:
            print("Nguồn:", doc.metadata.get('source', 'N/A'))
            print("Nội dung:\n", doc.page_content)
        print("-----\n")
