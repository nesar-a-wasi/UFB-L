import pdb
from nltk.corpus import stopwords
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
import re
import io
import os
import random
from pathlib import Path


#importing necessary libraries
import numpy as np, pandas as pd, random, matplotlib.pyplot as plt
import string

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from collections import OrderedDict
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn import svm

import text_normalizer as tn


def classification_report_csv(report,domain_name,classifer_name):
    lines = report.split('\n')
    for line in lines[2:-5]:
        print(line)
        row = {}
        report_data = []
        row_data = line.split()
        row['Domain'] = domain_name
        row['classifier'] = str(classifer_name)
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])*100
        row['recall'] = float(row_data[2])*100
        row['f1_score'] = float(row_data[3])*100
        report_data.append(row)
        row_data = lines[5].split()
        # print(list(report_data[0].values()))
        row['accuracy'] = float(row_data[1])*100
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv('classification_report.csv', index = False, mode='a',header=False)

def print_results(test_y,pred_test,classifer_name,domain):
    
    
    from sklearn.metrics import classification_report
    print(classification_report(test_y, pred_test,digits=4))
    # report = classification_report(test_y,pred_test,digits=4)
    report = classification_report(test_y, pred_test,digits=4)
    classification_report_csv(report,domain,classifer_name)



    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(test_y, pred_test)
    print(confusion_matrix)
    
    
    tn, fp, fn, tp = np.ravel(confusion_matrix).tolist()
    print("True Negatives: ",tn)
    print("False Positives: ",fp)   
    print("False Negatives: ",fn)
    print("True Positives: ",tp)
    #Precision 
    Precision = tp*100/(tp+fp) 
    print("Precision {:0.2f} %".format(Precision))
    #Recall 
    Recall = tp*100/(tp+fn) 
    print("Recall {:0.2f} %".format(Recall))
    #F1 Score
    f1 = (2*Precision*Recall)/(Precision + Recall)
    print("F1 Score {:0.2f} %".format(f1))
    #Accuracy
    Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
    print("Accuracy {:0.2f} %".format(Accuracy))

def word_extraction(text):
    #convert to tokens 
    tokens = word_tokenize(text) 
    # remove punctuation
    table = str.maketrans('', '', string.punctuation) 
    tokens = [w.translate(table) for w in tokens]  
    #remove non-alphabetic words
    tokens = [word for word in tokens if word.isalpha()] 
    stop_words = set(stopwords.words('english')) #list of english stop words 
    tokens = [w for w in tokens if not w in stop_words] #remove english stop words 
    tokens = [word for word in tokens if len(word) > 1] #remove one-lettered words 
    tokens = [w.lower() for w in tokens] #convert all words to lower case 
    return (tokens)

#define a function that averages the word vectors for each word in each line of the corpus
def document_vector(doc,word_list):
    """Create document vectors by averaging word vectors. """
    doc = [word2vec[word] for word in doc if word in word_list]
    if len(doc)>0:
        a = np.mean (doc, axis = 0)
    else:
        a = np.zeros(100)
    return a

#define a function that converts every sentence in the corpus the input to tokens and returns 
#the document-vector representation
def glovectorize_mean (x_train_df,word_list): 
    x_extract =[]
    for sentence in x_train_df:
        tok = word_extraction(sentence)
        #capture errors due to empty array 
        if len (tok) ==  0:
            a = np.zeros(100)
            x_extract.append (a)
        elif len(tok) == 1:
            try:
                a = document_vector(sentence,word_list)
                if pd.isna(a):
                    a = np.zeros(100)
                    x_extract.append (a)

                else:
                    x_extract.append (a)


            except :
                a = np.zeros(100)
                x_extract.append (a)

        else: 
            # x_extract.append (a)
            a = document_vector(word_extraction(sentence),word_list)
            x_extract.append (a)
           
        #Stack arrays in sequence vertically
        y = np.vstack(x_extract)
    return y


def process_tweet(review_text):
    import nltk
    # takens_list = []
    takens_list = nltk.word_tokenize(review_text)
    return takens_list

def count_tweets(tweets):
    word_l = []
    # cnt=0
    for tweet in tweets:
        for word in process_tweet(tweet):
            # cnt = cnt+1
            # print(cnt)
            if word not in word_l:
                word_l.append(word)
                

    return word_l

def classifers(x_extract,train_y,x_test_extract,test_y):
    
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='linear'))
    clf.fit(x_extract, train_y)
    pred_test = clf.predict(x_test_extract)

    print_results(test_y,pred_test,"SVM",curr_domain)

    
dataset_path = "dataset\\"

main_class ='books'
negative_classes = ['automotive', 'beauty', 'camera_&_photo', 'cell_phones_&_service', 'dvd', 'health_&_personal_care', 'jewelry_&_watches',
 'kitchen_&_housewares', 'sports_&_outdoors', 'toys_&_games']
not_in_negative_train_classes=["apparel","electronics","music","baby","grocery","magazines","outdoor_living","software","video","gourmet_food"]


train_positive = pd.read_csv (dataset_path+main_class+"//"+main_class+" train_positive_samples.csv",header=None,names={"Review"})
train_positive = train_positive['Review']
train_class_label=np.ones(len(train_positive),dtype=int)

train_negative = pd.DataFrame()

for i in range(0,len(negative_classes)):
    temp = pd.read_csv (dataset_path+negative_classes[i]+'/'+negative_classes[i]+' train_negative_samples.csv',header=None,names={"Review"})
    train_negative = pd.concat([train_negative,temp['Review']])
del i,temp

train_class_negative_label = np.full((len(train_negative)),-1)
train_class_label = np.concatenate((train_class_label,train_class_negative_label))


training_data = train_positive.copy()
training_data= training_data.append(train_negative.squeeze())



corpus = np.array(training_data)
norm_corpus = tn.normalize_corpus(corpus)


test_positive = pd.read_csv (dataset_path+main_class+'/'+main_class+' test_positive_samples.csv',header=None,names={"Review"})
test_positive = test_positive['Review']


test_negative = pd.DataFrame()

for i in range(0,len(negative_classes)):
    temp = pd.read_csv (dataset_path+negative_classes[i]+'/'+negative_classes[i]+' test_negative_samples.csv',header=None,names={"Review"})
    test_negative = pd.concat([test_negative,temp['Review']])
del i,temp

test_negative_not_in_train = pd.DataFrame()

for i in range(0,len(not_in_negative_train_classes)):
    temp = pd.read_csv (dataset_path+not_in_negative_train_classes[i]+'/'+not_in_negative_train_classes[i]+' test_negative_samples_not_in_train.csv',header=None,names={"Review"})
    test_negative_not_in_train = pd.concat([test_negative_not_in_train,temp['Review']])
del i,temp

testing_data = test_positive.copy()
testing_data= testing_data.append(test_negative.squeeze())
testing_data= testing_data.append(test_negative_not_in_train.squeeze())


positive_class_test_vector = np.full(len(test_positive),1)
negative_class_test_vector = np.full(len(test_negative)+len(test_negative_not_in_train),-1)
test_class_label = np.concatenate((positive_class_test_vector,negative_class_test_vector))

#PRinting Header row in csv file
import csv
head_row=['Domain','Classifier','Class','Precision','Recall','F1_score','Accuracy']
file = open('classification_report.csv', 'a', newline ='')
writer = csv.DictWriter(file, fieldnames = head_row)
writer.writeheader()
file.close()


word_embeddings = pd.read_csv('path_to_glove\\glove.6B.100d.txt',
                               header=None, sep=' ', index_col=0,
                               encoding='utf-8', quoting=3)

curr_domain='books'



# word_list = word_embeddings.index.values.tolist()
word_list = count_tweets(training_data)

word2vec = OrderedDict(zip(word_list, word_embeddings.values))

x_extract = glovectorize_mean(training_data,word_list)
# Tansform test data
x_test_extract  = glovectorize_mean(testing_data,word_list)

classifers(x_extract,train_class_label,x_test_extract,test_class_label)
