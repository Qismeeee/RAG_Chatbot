from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def normalize_text(text):
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = re.sub(r'[^\w\s]', '', text)
    # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Khởi tạo mô hình Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(question_list):
    # Chuẩn hóa câu hỏi
    normalized_questions = [normalize_text(q) for q in question_list]
    
    # Tạo embedding cho từng câu hỏi
    embeddings = model.encode(normalized_questions)

    # Tính ma trận độ tương đồng cosine
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

# Ví dụ
questions = [
    "Học phí tại Trường Đại học Cần Thơ là bao nhiêu?",
    "Quy định về học phí tại Trường Đại học Cần Thơ là gì?",
    "Thời gian đăng ký học phần tại Trường Đại học Cần Thơ?",
]
similarity_matrix = calculate_similarity(questions)
print(similarity_matrix)
