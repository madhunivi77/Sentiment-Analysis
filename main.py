# Author      : Madhumitha Sukumar
# Description : This python script that involves the use of machine learning techniques 
# to analyze and classify textual data based on the sentiment expressed. The project aims
# to build a predictive model capable of determining whether a given text conveys positive,
# negative, or neutral sentiment.

# importing necessary libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import LancasterStemmer
import re
from sklearn.utils import resample
from wordcloud import WordCloud
from sklearn.svm import  LinearSVC
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from string import punctuation
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# 1. Data Exploration
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]
df.Sentiment.value_counts()
df['Sentiment'] = df['Sentiment'].replace({4:1})
df = df[['Sentiment','text']]

# Plot graph for visualisation of count.
sns.set(style="whitegrid") 
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiments')
plt.show() 
# The plot shows that the data is unbalance. hence we will downsample the data to have the same count for each sentiment.

# majority class 0
df_majority = df[df['Sentiment']==0]
## minority class 1
df_minority = df[df['Sentiment']==1]

df_majority_downsampled = pd.DataFrame(resample(df_majority,
                                                 replace=False,
                                                 n_samples=len(df_minority),
                                                 random_state=1234))
df = pd.concat([df_majority_downsampled, df_minority], ignore_index=True)
df.head()

# Plot graph again after downsampling
sns.set(style="whitegrid") 
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='Set2')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiments')
plt.show() 
# Plot showcases that both classes have equal number of samples (248576)


# 2. Data Preprocessing
stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)
stemmer = LancasterStemmer()

# Convert the 'text' column of the dataframe into a list
corpus = df['text'].tolist()
print(len(corpus))
# print(corpus[0])

# Initialize empty lists to store the final processed corpus
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
print(data_cleaned['Sentiment'].value_counts())
print(data_cleaned.head())  

# 3. Exploratory Data Analysis
data_eda = pd.DataFrame()
data_eda['text'] = final_corpus
data_eda['Sentiment'] = df["Sentiment"].values
print(data_eda.head())

# Storing positive data seperately
positive = data_eda[data_eda['Sentiment'] == 1]
positive_list = positive['text'].tolist()

# Storing negative data seperately
negative = data_eda[data_eda['Sentiment'] == 0]
negative_list = negative['text'].tolist()

# Creating a single string containing all words
positive_all = " ".join([word for sent in positive_list for word in sent ])
negative_all = " ".join([word for sent in negative_list for word in sent ])


# Creating a wordcloud for positive reviews
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(positive_all)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Creating a wordcloud for negative reviews
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_all)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# INFERENCE 
# Positive data has words like Thank, love , LOL, Haha etc
# Negative data has words like hate,lone, sad, tired, suck ,sorry etc
# Some of the words are still common in both such as Lol, quot, work ,today etc

# Gets the number of occurences of all words in the dataframe
def get_count(data):
    dic = {}
    for i in data:
        for j in i:
            if j not in dic:
                dic[j]=1
            else:
                dic[j]+=1    
    return(dic)
count_corpus = get_count(positive_list)
count_corpus = pd.DataFrame({"word":count_corpus.keys(),"count":count_corpus.values()})
count_corpus = count_corpus.sort_values(by = "count", ascending = False)

# Plots a histogram  showing how many times each word appears in the corpus(positive)
import seaborn as sns
plt.figure(figsize = (10,5))
sns.barplot(x = count_corpus["word"][:20], y = count_corpus["count"][:20])
plt.title('one words in positive data')
plt.show()

# Plots a histogram  showing how many times each word appears in the corpus(negative)
count_corpus = get_count(negative_list)
count_corpus = pd.DataFrame({"word":count_corpus.keys(),"count":count_corpus.values()})
count_corpus = count_corpus.sort_values(by = "count", ascending = False)
import seaborn as sns
plt.figure(figsize = (10,5))
sns.barplot(x = count_corpus["word"][:20], y = count_corpus["count"][:20])
plt.title('one words in negative data')
plt.show()

# 4. Text Vectorization 
tfidf = TfidfVectorizer()
classifier = CalibratedClassifierCV(LinearSVC(dual = True), method='sigmoid')
vector = tfidf.fit_transform(data_cleaned['text'])
y = data_cleaned['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(vector, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=42,
                                                    stratify = y)

#Evaluate and display performance metrics for a machine learning model.
def metrics(y_train,y_train_pred,y_test,y_test_pred):
    '''
    Evaluate and display performance metrics for a machine learning model.

    Parameters:
    y_train (array-like): True labels of the training dataset.
    y_train_pred (array-like): Predicted labels of the training dataset.
    y_test (array-like): True labels of the testing dataset.
    y_test_pred (array-like): Predicted labels of the testing dataset.

    This function calculates and prints the accuracy of the model on both the 
    training and testing datasets. It also displays a normalized confusion matrix 
    and a classification report for both datasets, providing a detailed view of the
    model's performance in terms of precision, recall, and F1-score for each class.

    The function does not return any value; it only prints and displays the metrics.
    '''
    # Training Results
    print("training accuracy = ",round(accuracy_score(y_train,y_train_pred),2)*100)
    ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,normalize = 'all')
    print(classification_report(y_train,y_train_pred))
    plt.show()

    # Testing Results
    print("testing accuracy = ",round(accuracy_score(y_test,y_test_pred),2)*100)
    ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred,normalize = 'all')
    print(classification_report(y_test,y_test_pred))
    plt.show()

from sklearn.naive_bayes import MultinomialNB
# Multinomial NB 
NB = MultinomialNB()
NB.fit(X_train,y_train)
y_train_pred = NB.predict(X_train)
y_test_pred = NB.predict(X_test)
metrics(y_train,y_train_pred,y_test,y_test_pred)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Linear Support Vector Machine
svc = LinearSVC()
svc.fit(X_train,y_train)
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
metrics(y_train,y_train_pred,y_test,y_test_pred)

from sklearn.linear_model import LogisticRegression
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
metrics(y_train, y_train_pred, y_test, y_test_pred)

# Sample text testing of the model 
def preprocess_text(text, stop_words, stemmer, lemmatizer):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    # Convert to list from string
    text = text.split()
    # Lemmatization
    text = [lemmatizer.lemmatize(word) for word in text if not word in stop_words]
    # Join the list to make a string
    text = " ".join(text)
    return text

sample_text = "I love this product! It's amazing."
# Preprocess the sample text
stop_words = set(stopwords.words('english') + list(punctuation))
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
processed_sample = preprocess_text(sample_text, stop_words, stemmer, lemmatizer)

# Vectorize the sample text using the same TF-IDF vectorizer
sample_text_vector = tfidf.transform([processed_sample])

# Predict using the trained models
nb_prediction = NB.predict(sample_text_vector)
svc_prediction = svc.predict(sample_text_vector)
lr_prediction = lr.predict(sample_text_vector)

# Print the predictions
print(f"Naive Bayes Prediction: {'Positive' if nb_prediction[0] == 1 else 'Negative'}")
print(f"Linear SVC Prediction: {'Positive' if svc_prediction[0] == 1 else 'Negative'}")
print(f"Logistic Regression Prediction: {'Positive' if lr_prediction[0] == 1 else 'Negative'}")