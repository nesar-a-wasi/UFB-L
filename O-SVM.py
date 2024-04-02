import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import os
import nltk
import random
import text_normalizer as tn

def classification_report_csv(report,domain_name,classifer_name):
    lines = report.split('\n')
    for line in lines[2:-5]:
        print(line)
        row = {}
        report_data = []
        row_data = line.split()
        row['Domain'] = domain_name
        row['classifier'] = classifer_name
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

def print_results(test_y,pred_test):
    from sklearn.metrics import classification_report
    print(classification_report(test_y, pred_test,digits=4))
    report = classification_report(test_y, pred_test,digits=4)
    
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

dataset_path = "dataset\\"
main_class ='automotive'
negative_classes = ["kitchen_&_housewares","books","baby","beauty","grocery","music","outdoor_living","software","electronics","magazines"]
not_in_negative_train_classes=['camera_&_photo', 'apparel', 'toys_&_games', 'gourmet_food', 'video', 'cell_phones_&_service', 'health_&_personal_care', 'dvd', 'sports_&_outdoors', 'jewelry_&_watches']

train_positive = pd.read_csv (dataset_path+main_class+"//"+main_class+" train_positive_samples.csv",header=None)
train_positive = train_positive[0]
train_class_label=np.ones(len(train_positive),dtype=int)

train_negative = pd.DataFrame()

for i in range(0,len(negative_classes)):
    temp = pd.read_csv (dataset_path+negative_classes[i]+'/'+negative_classes[i]+' train_negative_samples.csv',header=None)
    train_negative = pd.concat([train_negative,temp[0]])
del i,temp

train_class_negative_label = np.full((len(train_negative)),-1)
train_class_label = np.concatenate((train_class_label,train_class_negative_label))

training_data = train_positive.copy()
training_data= training_data.append(train_negative.squeeze())

corpus = np.array(training_data)
norm_corpus = tn.normalize_corpus(corpus)

test_positive = pd.read_csv (dataset_path+main_class+'/'+main_class+' test_positive_samples.csv',header=None)
test_positive = test_positive[0]

test_negative = pd.DataFrame()

for i in range(0,len(negative_classes)):
    temp = pd.read_csv (dataset_path+negative_classes[i]+'/'+negative_classes[i]+' test_negative_samples.csv',header=None)
    test_negative = pd.concat([test_negative,temp[0]])
del i,temp

test_negative_not_in_train = pd.DataFrame()

for i in range(0,len(not_in_negative_train_classes)):
    temp = pd.read_csv (dataset_path+not_in_negative_train_classes[i]+'/'+not_in_negative_train_classes[i]+' test_negative_samples_not_in_train.csv',header=None)
    test_negative_not_in_train = pd.concat([test_negative_not_in_train,temp[0]])
del i,temp

testing_data = test_positive.copy()
testing_data= testing_data.append(test_negative.squeeze())
testing_data= testing_data.append(test_negative_not_in_train.squeeze())


################## Unigram ##################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_unigrams = TfidfVectorizer(stop_words ='english',ngram_range = (1, 1))
X_train_unigrams = vectorizer_unigrams.fit_transform(norm_corpus).toarray()


###### Cleaning and tokenizing testing data 
X_test_corpus = np.array(testing_data).squeeze()

X_test_corpus_norm = tn.normalize_corpus(X_test_corpus.squeeze())

X_test_unigram = vectorizer_unigrams.transform(X_test_corpus_norm).toarray()
positive_class_test_vector = np.full(len(test_positive),1)
negative_class_test_vector = np.full(len(test_negative)+len(test_negative_not_in_train),-1)
test_class_label = np.concatenate((positive_class_test_vector,negative_class_test_vector))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto')
clf.fit(X_train_unigrams, train_class_label)
pred_test = clf.predict(X_test_unigram)
print_results(test_class_label,pred_test)

