import numpy as np

def cos_similarity(v1,v2):
    dot_product = np.dot(v1,v2)
    l2_norm = np.sqrt(sum(np.square(v1)))*np.sqrt(sum(np.square(v2)))
    similarity = dot_product / l2_norm

    return similarity

from sklearn.feature_extraction.text import TfidfVectorizer

doc_list = ['if you take the blue pill, the story ends' ,
            'if you take the red pill, you stay in Wonderland',
            'if you take the red pill, I show you how deep the rabbit hole goes']

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)

print(feature_vect_simple)

feature_vect_dense = feature_vect_simple.todense()

vect1 = np.array(feature_vect_dense[0]).reshape(-1,)
vect2 = np.array(feature_vect_dense[1]).reshape(-1,)

similarity_simple12 = cos_similarity(vect1,vect2)

print("문장 1, 문장 2의 유사도 : {0:.3f}".format(similarity_simple12))

vect3 = np.array(feature_vect_dense[2]).reshape(-1,)

similarity_simple13 = cos_similarity(vect1,vect3)

print("문장 1, 문장 3의 유사도 : {0:.3f}".format(similarity_simple13))

similarity_simple23 = cos_similarity(vect2,vect3)

print("문장 2, 문장 3의 유사도 : {0:.3f}".format(similarity_simple23))

from sklearn.metrics.pairwise import cosine_similarity
similarity_simple_pair = cosine_similarity(feature_vect_simple,feature_vect_simple)
print(similarity_simple_pair)


######################################################
######################################################

import pandas as pd
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

all_files = glob.glob(os.path.join("OpinosisDataset1.0","topics","*.data"))
filename_list = []
opinion_text = []

for file_ in all_files:
    df = pd.read_table(file_,index_col=None,header=0,encoding='latin1')
    filename_ = file_.split('/')[-1]
    filename = filename_.split('.')[0]
    filename_list.append(filename)
    opinion_text.append(df.to_string())

from nltk.stem import WordNetLemmatizer
import nltk
import string

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


document_df = pd.DataFrame({'filename':filename_list,'opnion_text':opinion_text})
tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words = 'english', ngram_range = (1,2), min_df = 0.05, max_df = 0.85)
feature_vect = tfidf_vect.fit_transform(document_df['opnion_text'])
km_cluster = KMeans(n_clusters=3,max_iter=10000,random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_center = km_cluster.cluster_centers_
document_df['label'] = cluster_label

from sklearn.metrics.pairwise import cosine_similarity

hotel_index = document_df[document_df['label']==1].index
comparison_docname = document_df.iloc[hotel_index[0]]['filename']
similarity_pair = cosine_similarity(feature_vect[hotel_index[0]],feature_vect[hotel_index])
print(similarity_pair)

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sorted_index = similarity_pair.argsort()[:,::-1]
sorted_index = sorted_index[:,1:]
hotel_sorted_indexes = hotel_index[sorted_index.reshape(-1)]
hotel_1_sim_value = np.sort(similarity_pair.reshape(-1))[::-1]
hotel_1_sim_value = hotel_1_sim_value[1:]

hotel_1_sim_df = pd.DataFrame()
hotel_1_sim_df['filename'] = document_df.iloc[hotel_sorted_indexes]['filename']
hotel_1_sim_df['similarity'] = hotel_1_sim_value
print(document_df)
sns.barplot(x='similarity',y='filename',data=hotel_1_sim_df)
plt.title(comparison_docname)
plt.show()