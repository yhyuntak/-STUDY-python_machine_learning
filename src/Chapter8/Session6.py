from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

categories = ['rec.motorcycles','rec.sport.baseball','comp.graphics',
              'comp.windows.x', 'sci.electronics', 'sci.med', 'soc.religion.christian', 'talk.politics.mideast' ]

news_df = fetch_20newsgroups(subset="all",remove=('headers','footers','quotes'),random_state=0,categories=categories)
count_vect = CountVectorizer(max_df=0.95,max_features=1000,min_df=2,stop_words='english',ngram_range=(1,2))
feat_vect = count_vect.fit_transform(news_df.data)
lda = LatentDirichletAllocation(n_components=8,random_state=0)
lda.fit(feat_vect)

def display_topics(model,feature_names,no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print("Topic #",topic_index)

        topic_word_indexs = topic.argsort()[::-1]
        top_indexes = topic_word_indexs[:no_top_words]

        feature_concat = " ".join([feature_names[i] for i in top_indexes])
        print(feature_concat)

feature_names = count_vect.get_feature_names()
display_topics(lda,feature_names,15)