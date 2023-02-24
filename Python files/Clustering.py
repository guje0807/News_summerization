import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import spacy
from spacy.lang.en import English
import re
import nltk as nt
from nltk.tokenize import WhitespaceTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


def w_tokenizer(text):
    tokenizer = WhitespaceTokenizer()
    # Use tokenize method
    tokenized_list = tokenizer.tokenize(text)
    return (tokenized_list)


def remove_stopwords(text_list):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    return_list = []
    for i in range(len(text_list)):
        if text_list[i] not in spacy_stopwords:
            return_list.append(text_list[i])
    return(return_list)


def preprocessor_final(text):
    if isinstance((text), (str)):
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('[\W]+', '', text.lower())
        return text
    if isinstance((text), (list)):
        return_list = []
        for i in range(len(text)):
            temp_text = re.sub('<[^>]*>', '', text[i])
            temp_text = re.sub('[\W]+', '', temp_text.lower())
            return_list.append(temp_text)
        return(return_list)
    else:
        pass


def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})


def pre_proces(news):
    estimators = [('tokenizer', pipelinize(w_tokenizer)), ('stopwordremoval', pipelinize(remove_stopwords)),
                  ('preprocessor', pipelinize(preprocessor_final))]
    pipe = Pipeline(estimators)

    norm_corpus = pipe.transform(news)

    main_corpus = []
    for i in norm_corpus:
        doc = " "
        print(i)
        if len(i) == 0:
            pass
        for j in i:
            doc += ' ' + j
            #print('doc:',doc)
        main_corpus.append(doc)
    #print("main corpus",main_corpus)
    return main_corpus


def cluster(df):
    df = df.dropna()
    main_corpus = pre_proces(list(df['news']))
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=4)
    cv_matrix = cv.fit_transform(main_corpus)
    Num_Clusters = 3
    km = KMeans(n_clusters=Num_Clusters, max_iter=100, n_init=50, random_state=42)
    km.fit(cv_matrix)
    df['kmeans_clusters'] = km.labels_
    news_clusters = (df[['title', 'kmeans_clusters']]).sort_values(by=['kmeans_clusters'], ascending=False)
    news_clusters = news_clusters.copy(deep=True)
    feature_names = cv.get_feature_names()
    topn_features = 10
    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    topics = []
    for cluster_num in range(Num_Clusters):
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
        news = news_clusters[news_clusters['kmeans_clusters'] == cluster_num]['title'].values.tolist()
        topics.append(news[0])
    return topics






