# tfidf_summarizer.py
import numpy as np
import pickle
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import sent_tokenize
from collections import OrderedDict
import nltk

nltk.download('punkt_tab')

# 미리 저장된 vectorizer를 로드합니다.
with open('model.pkl', 'rb') as model_file:
    vectorizer = pickle.load(model_file)

# TF-IDF 기반 요약 함수
def summarize_document(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= 5:
        return None, "Text must have more than 5 sentences."
    
    sentence_vectors = vectorizer.transform(sentences).toarray()
    doc_vector = np.mean(sentence_vectors, axis=0)
    similarities = linear_kernel(doc_vector.reshape(1, -1), sentence_vectors).flatten()
    sentence_scores = sorted(enumerate(zip(similarities, sentences)), reverse=True, key=lambda x: x[1][0])
    top_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: x[0])
    ordered_sentences = [sentences[index] for index, _ in top_sentences]
    summary = ' '.join(ordered_sentences)
    
    return summary, None
