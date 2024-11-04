import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import ElectraTokenizer
import pickle
import kss
import re
from rouge import Rouge
from concurrent.futures import ThreadPoolExecutor, as_completed

model_file = 'model.pkl'

# 토크나이저 초기화 (KoELECTRA)
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

# 한국어 불용어 로드 함수
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

stopwords = load_stopwords('./stopwords-ko.txt')

# 데이터 로드 함수
def load_data(directory):
    texts, summaries = [], []
    print("데이터 로딩 중...")
    files = [f for f in os.listdir(directory) if f.endswith('.json') and os.path.getsize(os.path.join(directory, f)) >= 1024]

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(load_file, os.path.join(directory, file)): file for file in files}
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="파일 로딩"):
            content, summary = future.result()
            if content and summary:
                texts.append(content)
                summaries.append(summary)

    return texts, summaries

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if 'summary' in data:
            content = data['content']
            content = remove_special_phrases(content)
            return content, data['summary']
    return None, None

# 특수 형식의 문구 제거 함수
def remove_special_phrases(text):
    text = re.sub(r'^\(.*?\)\s.*?기자\s=\s', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'※.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s.*?기자\(.*?\)$', '', text, flags=re.MULTILINE)
    return text

# 텍스트 전처리 함수
def preprocess_text(texts):
    processed_texts = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(preprocess_single_text, text) for text in texts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="텍스트 전처리"):
            processed_texts.append(future.result())
    return processed_texts

def preprocess_single_text(text):
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    text = ' '.join(tokenizer.tokenize(text))
    return text

# 문서 요약 함수
def summarize_document(text, vectorizer, top_n=3):
    sentences = kss.split_sentences(text)
    sentence_vectors = vectorizer.transform(sentences).toarray()
    doc_vector = np.mean(sentence_vectors, axis=0)
    similarities = linear_kernel(doc_vector.reshape(1, -1), sentence_vectors).flatten()
    sentence_scores = [(score, sentence) for score, sentence in zip(similarities, sentences)]
    sentence_scores.sort(reverse=True)
    summary = ' '.join([sentence for _, sentence in sentence_scores[:top_n]])
    return summary

# 모델 학습 및 저장
def train_and_save_model(texts, model_path=model_file):
    print("모델 학습 및 저장 중...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=stopwords)
    vectorizer.fit(texts)

    with open(model_path, 'wb') as model_file:
        pickle.dump(vectorizer, model_file)
    print("모델 저장 완료")

# 요약 성능 평가 함수
def evaluate_summary(reference_summaries, generated_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    return scores

# 실행
directory = 'datas'
texts, reference_summaries = load_data(directory)
texts = preprocess_text(texts)
train_and_save_model(texts)
vectorizer = pickle.load(open(model_file, 'rb'))

# 요약 생성 및 평가
generated_summaries = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(summarize_document, text, vectorizer) for text in texts]
    for future in tqdm(as_completed(futures), total=len(futures), desc="요약 생성 중"):
        generated_summaries.append(future.result())

evaluation_scores = evaluate_summary(reference_summaries, generated_summaries)
print("요약 평가 결과:", evaluation_scores)