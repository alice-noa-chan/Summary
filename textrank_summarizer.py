# textrank_summarizer.py
from textrankr import TextRank
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from nltk.tokenize import sent_tokenize
from typing import List
import difflib

class SoynlpTokenizer:
    def __init__(self, texts: List[str]):
        self.eojeol_counter = self._get_eojeol_counter(texts)
        self.tokenizer = LTokenizer(scores=self.eojeol_counter)

    def _get_eojeol_counter(self, texts: List[str]):
        word_extractor = WordExtractor()
        word_extractor.train(texts)
        words = word_extractor.extract()
        return {word: score.cohesion_forward for word, score in words.items()}

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.tokenizer.tokenize(text)
        return tokens

# 문장 분할기 (nltk 사용)
def split_sentences(text: str) -> List[str]:
    return sent_tokenize(text)

# 가장 유사한 원문 문장 찾기
def find_most_similar_sentence(summary_sentence: str, original_sentences: List[str]) -> str:
    most_similar = difflib.get_close_matches(summary_sentence, original_sentences, n=1, cutoff=0.1)
    return most_similar[0] if most_similar else summary_sentence

# Textrank summarization function
def summarize_textrank(text: str, num_sentences: int = 3) -> List[str]:
    texts = split_sentences(text)
    
    tokenizer = SoynlpTokenizer(texts)
    textrank = TextRank(tokenizer)
    
    summarized = textrank.summarize(text, num_sentences, verbose=False)
    
    original_sentences = split_sentences(text)
    summarized_with_originals = [find_most_similar_sentence(summary, original_sentences) for summary in summarized]
    
    return summarized_with_originals
