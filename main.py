# Author      : Madhumitha Sukumar
# Description : 

import numpy as np
import pandas as pd
import seaborn as sns
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
from sklearn.utils import resample
from sklearn.svm import  LinearSVC
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from string import punctuation
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV


# 1. Data Exploration
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]
df.Sentiment.value_counts()
df['Sentiment'] = df['Sentiment'].replace({4:1})
df = df[['Sentiment','text']]

# Plot graph for visualisation of count.
# sns.set(style="whitegrid") 
# sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2')
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.title('Distribution of Sentiments')
# plt.show() 
# The plot shows that the data is unbalance. hence we will downsample the data to have the same count for each sentiment.

## majority class 0
df_majority = df[df['Sentiment']==0]
## minority class 1
df_minority = df[df['Sentiment']==1]
df_majority_downsampled = pd.DataFrame(resample(df_majority,
                                                 replace=False,
                                                 n_samples=len(df_minority),
                                                 random_state=1234))

df = pd.concat([df_majority_downsampled, df_minority], ignore_index=True)
df.head()

sns.set(style="whitegrid") 
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiments')
# plt.show() -- both classes have equal number of samples (248576)

# 2. Data Preprocessing
stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)
stemmer = LancasterStemmer()

corpus = df['text'].tolist()
print(len(corpus))
print(corpus[0])

final_corpus = []
final_corpus_joined = []
for i in df.index:

    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    #Convert to lowercase
    text = text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()

    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text 
            if not word in stuff_to_be_removed] 
    text1 = " ".join(text)
    final_corpus.append(text)
    final_corpus_joined.append(text1)

data_cleaned = pd.DataFrame()
data_cleaned["text"] = final_corpus_joined
data_cleaned["Sentiment"] = df["Sentiment"].values
# print(data_cleaned['Sentiment'].value_counts())
# print(data_cleaned.head())  

