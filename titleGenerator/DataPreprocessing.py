#%%
#from nltk import word_tokenize
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
# nltk.download('stopwords')
# nltk.download('wordnet')

#%%

def DataNormalize(raw_path):

    # './drive/MyDrive/4rd1/DataMining/IRTM_Final/{}_Raw.pkl'
    with open(raw_path, "rb") as fp:
        train = pickle.load(fp)

    token_set = []
    # stop_words = set(stopwords.words('english'))
    # print(stop_words) 
    # word_tokens = word_tokenize(train[0]['content']) 不去除標點符號
    for i in range(len(train)):
        word_tokens = tokenizer.tokenize(train[i]['content'])  # 去除標點
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
        # if w not in stop_words:
            filtered_sentence.append(lemmatizer.lemmatize(w))
        # filtered_sentence.append(stemmer.stem(w))
        # print(len(word_tokens))
        # print(filtered_sentence)
        token_set.append(filtered_sentence)
    return token_set

#%%

def GenTFIDF(token_set, header = None):
    IDF = {}
    showed = {}
    for i in range(len(token_set)):
        showed = {}
        for word in token_set[i]:
            if word not in IDF:
                IDF[word] = 1
            else:
                if word not in showed:
                    showed[word] = 1
                    IDF[word] += 1
                else:
                    continue
    for key in IDF.keys():
        IDF[key] = math.log(len(token_set)/IDF[key], 10)
    IDF = {k: v for k, v in sorted(IDF.items(), key=lambda item: item[1], reverse=True)}
    
    tfidf_arr = []
    for i in range(len(token_set)):
        TF = Counter(token_set[i])
        TFIDF = {}
        for key in TF.keys():
            TFIDF[key] = TF[key] * IDF[key]
            # TFIDF = {k: v for k, v in sorted(TFIDF.items(), key=lambda item: item[1], reverse=True)}
            TFIDF = sorted(TFIDF.items(), key=lambda item: item[1], reverse=True)
        tfidf_arr.append(TFIDF)  

    return tfidf_arr

#%%
def main():
    token_set = "./news/SPORTS_Raw.pkl"
    tfidf = GenTFIDF(token_set)
    print(tfidf[0])

if __name__ == "__main__":
    main()