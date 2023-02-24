import numpy as np
import spacy
from spacy.lang.en import English
import re
import nltk as nt
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer


def get_summarized_article(article):
    print("Entered get_summarized article")
    clean_text = clean(article)
    print(clean_text)
    sentences = nt.sent_tokenize(clean_text)
    print(sentences)
    final_text = remove_stopwords(sentences)
    print(final_text)
    if len(final_text) == 0: return None
    # Vectorizing the article
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(final_text)  # Get the Document Term Matrix
    dt_matrix = dt_matrix.toarray()
    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T

    a = td_matrix.shape
    if a[1] <= 2:
        return None

    # Implementing Singular value Decomposition
    num_sentences = 3
    num_topics = 3

    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)
    print(u.shape, s.shape, vt.shape)

    term_topic_mat, singular_values, topic_document_mat = u, s, vt
    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
    top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
    top_sentence_indices.sort()
    return ''.join(np.array(sentences)[top_sentence_indices])


def clean(text):
    print("Entered preprocessor_final")
    print(text)
    if isinstance((text), (str)):
        #text = re.sub('<[^>]*>', ' ', text)
        #text = re.sub(r'<[^<>]*>', ' ', text)
        #text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
        #text = re.sub(r'http.*', ' ', text)
        #text = re.sub(r'\[[^\[\]]*\]', ' ', text)
        text = re.sub('[\W]+', ' ', text.lower())
        return text

    if isinstance((text), (list)):
        return_list = []
        for i in range(len(text)):
            #temp_text = re.sub('<[^>]*>', ' ', temp_text[i])
            #temp_text = re.sub('<[^<>]*>', ' ', temp_text[i])
            #temp_text = re.sub('\[([^\[\]]*)\]\([^\(\)]*\)', ' ', temp_text[i])
            #temp_text = re.sub('\[[^\[\]]*\]', ' ', temp_text[i])
            temp_text = re.sub('<[^>]*>', ' ', temp_text[i])
            #temp_text = re.sub('[\W]+', ' ', temp_text.lower())
            return_list.append(temp_text)
        return return_list
    else:
        pass


def remove_stopwords(text_list):
    print("Entered remove_stopwords")
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    return_list = []
    for i in range(len(text_list)):
        if text_list[i] not in spacy_stopwords:
            return_list.append(text_list[i].lower())
    return return_list


def low_rank_svd(matrix, singular_count=1):
    print("Entered low_rank_svd")
    print(matrix.shape)
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt
