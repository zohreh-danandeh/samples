import nltk

#1. Sentence tokenization

text = "Navigating Exchange-Traded Fund (ETF) capital markets can appear complicated without the right resources. From understanding ETF liquidity and transaction costs to general education and guidance, Goldman Sachs Asset Management (GSAM) is committed to helping our clients navigate the capital markets infrastructure within ETF trading and execution. This quick reference guide walks through key components of ETF capital markets and relevant trading practices."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)
    print()
    
 #2. Word Tokenization
 
for sentence in sentences:
 words = nltk.word_tokenize(sentence)
 print(words)
 print()
 
 #3.Text Lemmatization and Stemming
 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

def compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word, pos):
   
    print("Stemmer:", stemmer.stem(word))
    print("Lemmatizer:", lemmatizer.lemmatize(word, pos))
    print()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "seen", pos = wordnet.VERB)
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "drove", pos = wordnet.VERB)

#4.Stop words
from nltk.corpus import stopwords
print(stopwords.words("english"))

stop_words = set(stopwords.words("english"))
sentence = "Navigating Exchange-Traded Fund  capital markets can appear complicated without the right resources."

words = nltk.word_tokenize(sentence)
without_stop_words = []
for word in words:
    if word not in stop_words:
        without_stop_words.append(word)

print(without_stop_words)

#5.Regex
import re
sentence = "The development of snowboarding was inspired by skateboarding, sledding, surfing and skiing."
pattern = r"[^\w]"
print(re.sub(pattern, " ", sentence))


#6.Bag-of-words
#Step 6.1. Load the Data
with open("C:\\Users\\marco\\Desktop\\simple.txt", "r") as file:
    documents = file.read().splitlines()
    
print(documents)

# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 6.2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 6.3. Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(documents)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)

#  TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
pd1=pd.DataFrame(values.toarray(), columns = feature_names)


