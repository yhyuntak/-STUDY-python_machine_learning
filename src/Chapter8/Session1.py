from nltk import sent_tokenize
import nltk
nltk.download('punkt')

text_sample = 'The Matrix is everywhere its all around us, here even in this room. You can see it out your window or on your television. You feel it when you go to work, or go to church or pay your taxes.'
sentences = sent_tokenize(text=text_sample)
print(type(sentences),len(sentences))
print(sentences)

from nltk import word_tokenize
sentence = sentences[0]
words = word_tokenize(sentence)
print(words)

def tokenize_text(text):
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

word_tokens = tokenize_text(text_sample)
print(word_tokens)

################################################################################################################
################################################################################################################

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens:
    filtered_words = []
    for word in sentence :
        word = word.lower()
        if word not in stopwords :
            filtered_words.append(word)
    all_tokens.append(filtered_words)

print(all_tokens)

################################################################################################################
################################################################################################################


from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
print(stemmer.stem('working'),stemmer.stem('works'),stemmer.stem('worked'))
#
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# lemma = WordNetLemmatizer()
# print(lemma.lemmatize('amusing','v'),lemma.lemmatize('amuses','v'),lemma.lemmatize('amused','v'))
#
