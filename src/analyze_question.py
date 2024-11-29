import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import json

def normalize_question(question):
    question = question.lower()
    question = re.sub(r'[^\w\s]', '', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def get_embeddings(questions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(questions, convert_to_tensor=True)

def calculate_similarity(embeddings):
    return util.pytorch_cos_sim(embeddings, embeddings)

def group_similar_questions(questions, similarity_matrix, threshold=0.8):
    groups = []
    visited = set()

    for i, question in enumerate(questions):
        if i in visited:
            continue
        group = [question]
        visited.add(i)
        for j in range(i + 1, len(questions)):
            if j not in visited and similarity_matrix[i][j] >= threshold:
                group.append(questions[j])
                visited.add(j)
        groups.append(group)
    
    return groups

def count_group_frequencies(groups):
    return {group[0]: len(group) for group in groups}

def display_suggestions(group_frequencies):
    for question, frequency in group_frequencies.items():
        print(f"Câu hỏi: '{question}' xuất hiện {frequency} lần.")

def get_suggestions(limit=4):
    log_file_path = "D:/HK1_2024-2025/Chatbot/Chat/data/chat_logs.json"
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        data = json.load(log_file)

    questions = [normalize_question(entry["message"]) for entry in data]
    embeddings = get_embeddings(questions)
    similarity_matrix = calculate_similarity(embeddings)
    groups = group_similar_questions(questions, similarity_matrix)
    
    group_frequencies = count_group_frequencies(groups)
    sorted_questions = sorted(group_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_questions = [q[0] for q in sorted_questions[:limit]]

    return top_questions

suggestions = get_suggestions()