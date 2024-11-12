import os
from pdf2image import convert_from_path
import pytesseract
from google.cloud import vision
import io

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Đường dẫn tới Google Cloud API Key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/LOQ/AppData/Local/Google/Cloud SDK/key.json"


def extract_text_from_image(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response.full_text_annotation.text if response.full_text_annotation else ""


def extract_text_from_pdf_scan(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang='vie')
    return text


def save_extracted_text(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


def process_all_pdfs_in_directory(input_directory, output_directory):
    os.makedirs(output_directory,
                exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_directory, filename)
            print(f"Đang xử lý: {pdf_path}")
            extracted_text = extract_text_from_pdf_scan(pdf_path)

            output_file_path = os.path.join(
                output_directory, f"{os.path.splitext(filename)[0]}.txt")

            save_extracted_text(extracted_text, output_file_path)
            print(f"Đã lưu văn bản từ {filename} vào {output_file_path}")


if __name__ == "__main__":
    input_directory = "data/raw/pdfs/"
    output_directory = "data/processed/extracted_texts/"

    process_all_pdfs_in_directory(input_directory, output_directory)
