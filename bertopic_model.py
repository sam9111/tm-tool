import torch
from transformers import BertModel, BertTokenizerFast
import pickle
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Mapping, Tuple, Union
import stanza
import re

import dill as pickle


file_path = "./data/"

tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model = BertModel.from_pretrained("setu4993/LaBSE")


def save(trained_model):
    with open(file_path+"topic_model", 'wb') as f:
        pickle.dump(trained_model, f)


def load():
    with open(file_path+"topic_model", 'rb') as f:
        trained_model = pickle.load(f)
    return trained_model


def tokenize_ta(text, return_tensors="pt", *args, **kwargs):
    return indic_tokenize.trivial_tokenize(text)


stopwords = ['அங்கு',
             'அங்கே',
             'அடுத்த',
             'அதனால்',
             'அதன்',
             'அதற்கு',
             'அதிக',
             'அதில்',
             'அது',
             'அதே',
             'அதை',
             'அந்த',
             'அந்தக்',
             'அந்தப்',
             'அன்று',
             'அல்லது',
             'அவன்',
             'அவரது',
             'அவர்',
             'அவர்கள்',
             'அவள்',
             'அவை',
             'ஆகிய',
             'ஆகியோர்',
             'ஆகும்',
             'இங்கு',
             'இங்கே',
             'இடத்தில்',
             'இடம்',
             'இதனால்',
             'இதனை',
             'இதன்',
             'இதற்கு',
             'இதில்',
             'இது',
             'இதை',
             'இந்த',
             'இந்தக்',
             'இந்தத்',
             'இந்தப்',
             'இன்னும்',
             'இப்போது',
             'இரு',
             'இருக்கும்',
             'இருந்த',
             'இருந்தது',
             'இருந்து',
             'இவர்',
             'இவை',
             'உன்',
             'உள்ள',
             'உள்ளது',
             'உள்ளன',
             'எந்த',
             'என',
             'எனக்',
             'எனக்கு',
             'எனப்படும்',
             'எனவும்',
             'எனவே',
             'எனினும்',
             'எனும்',
             'என்',
             'என்ன',
             'என்னும்',
             'என்பது',
             'என்பதை',
             'என்ற',
             'என்று',
             'என்றும்',
             'எல்லாம்',
             'ஏன்',
             'ஒரு',
             'ஒரே',
             'ஓர்',
             'கொண்ட',
             'கொண்டு',
             'கொள்ள',
             'சற்று',
             'சிறு',
             'சில',
             'சேர்ந்த',
             'தனது',
             'தன்',
             'தவிர',
             'தான்',
             'நான்',
             'நாம்',
             'நீ',
             'பற்றி',
             'பற்றிய',
             'பல',
             'பலரும்',
             'பல்வேறு',
             'பின்',
             'பின்னர்',
             'பிற',
             'பிறகு',
             'பெரும்',
             'பேர்',
             'போது',
             'போன்ற',
             'போல',
             'போல்',
             'மட்டுமே',
             'மட்டும்',
             'மற்ற',
             'மற்றும்',
             'மிக',
             'மிகவும்',
             'மீது',
             'முதல்',
             'முறை',
             'மேலும்',
             'மேல்',
             'யார்',
             'வந்த',
             'வந்து',
             'வரும்',
             'வரை',
             'வரையில்',
             'விட',
             'விட்டு',
             'வேண்டும்',
             'வேறு']

vectorizer_model = CountVectorizer(
    stop_words=stopwords, analyzer='word',
    tokenizer=tokenize_ta
)


def set_topic_model(top_n_words, nr_topics, low_memory):

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=False,
        embedding_model=model,
        top_n_words=top_n_words,
        nr_topics=nr_topics,
        low_memory=low_memory,
    )

    return topic_model


def embeddings(docs):

    model.eval()
    n_docs = len(docs)
    batch_size = 8
    embeds = torch.zeros((n_docs, model.config.hidden_size))
    for i in range(0, n_docs, batch_size):
        batch = docs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.pooler_output
        embeds[i:i+batch_size] = batch_embeddings

    with open(file_path+"embeddings.pkl", "wb") as f:
        pickle.dump(embeds, f)


def load_embeddings():
    with open(file_path+"embeddings.pkl", "rb") as f:
        embeds = pickle.load(f)

        return embeds.detach().numpy()


def clean(content):
    if not content:
        return ""
    CLEAN_HTML = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    CLEAN_ALPHA = re.compile(".*?[A-Za-z0-9].*?")
    CLEAN_PUNCT = re.compile(r"[%\"&.;:`!\'?,\"()\[\]-|’‘-“”-]")
    CLEAN_WHITE = re.compile(r"\s{2,}")

    content = re.sub(CLEAN_HTML, " ", content)
    content = re.sub(CLEAN_ALPHA, " ", content)
    content = re.sub(CLEAN_PUNCT, "", content)
    content = re.sub(CLEAN_WHITE, " ", content)
    content = content.strip()
    return content


def run(data, top_n_words, nr_topics, low_memory):

    data = data['text'].apply(clean).tolist()

    embeddings(data)

    embeds = load_embeddings()
    topic_model = set_topic_model(
        top_n_words, nr_topics, low_memory)

    trained_model = topic_model.fit(data, embeds)

    save(trained_model)


def get_topic_info():
    trained_model = load()
    return trained_model.get_topic_info()


def get_topics():
    trained_model = load()
    return trained_model.topics_


def reduce_outliers(docs):

    trained_model = load()

    docs = docs['text'].apply(clean).tolist()

    new_topics = trained_model.reduce_outliers(
        docs, topics=trained_model.topics_, strategy="c-tf-idf")

    trained_model.update_topics(
        docs, topics=new_topics, vectorizer_model=vectorizer_model)

    save(trained_model)


def search_topic(query, top_n):

    trained_model = load()

    similar_topics, similarity = trained_model.find_topics(query, top_n=top_n)

    result = []
    for topic in similar_topics:
        result.append(
            (topic, trained_model.get_representative_docs(topic))
        )

    return result


class TamilPOS():
    """
    Extract Topic Keywords based on their Part-of-Speech using stanza library for Tamil.
    """

    def __init__(self,
                 top_n_words: int = 10,
                 pos_patterns: List[str] = None):
        self.top_n_words = top_n_words

        if pos_patterns is None:
            self.pos_patterns = [
                'NOUN',
                'PROPN',
            ]
        else:
            self.pos_patterns = pos_patterns

        # load stanza pipeline for Tamil
        self.nlp = stanza.Pipeline(lang='ta', processors='tokenize,pos')

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        topic_to_keywords = {}
        for topic_id, topic_words in topics.items():
            # filter candidate documents that contain at least one keyword from the topic
            mask = documents['text'].str.contains(
                '|'.join([word[0] for word in topic_words]), regex=True)
            candidate_docs = documents[mask]

            # extract candidate keywords from candidate_docs based on POS patterns
            candidate_keywords = []
            for doc in candidate_docs['text']:
                doc_keywords = []
                # get POS tags for each word in the document
                doc_words = self.nlp(doc).sentences[0].words
                for word in doc_words:
                    if word.upos in self.pos_patterns:
                        doc_keywords.append(word.text)
                candidate_keywords.extend(doc_keywords)
            # count the frequency of each keyword and keep the top n
            candidate_keyword_counts = pd.Series(
                candidate_keywords).value_counts().head(self.top_n_words)
            # normalize keyword counts
            candidate_keyword_counts = candidate_keyword_counts / \
                candidate_docs.shape[0]
            # assign c-TF-IDF scores to keywords
            keyword_scores = [(word, topic_model.get_topic(topic_id)[word])
                              for word in candidate_keyword_counts.index]
            # sort keywords by their respective c-TF-IDF scores
            sorted_keyword_scores = sorted(
                keyword_scores, key=lambda x: x[1], reverse=True)
            # add top n keywords to topic_to_keywords dict
            topic_to_keywords[topic_id] = sorted_keyword_scores[:self.top_n_words]
        return topic_to_keywords


def apply_pos_removal(docs, tags):

    trained_model = load()

    docs = docs['text'].apply(clean).tolist()

    trained_model.update_topics(
        docs, representation_model=TamilPOS(pos_patterns=tags), vectorizer_model=vectorizer_model)

    save(trained_model)
