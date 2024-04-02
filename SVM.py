import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import os
import nltk
import random
import text_normalizer as tn

#function to convert CSV file data to Libsvm format
def csvtolibsvm(csvfile,outfile,target_label_index,header_Flag):
    import sys
    import csv
    from collections import defaultdict
    
    def construct_line( label, line ):
        new_line = []
        if float( label ) == 0.0:
            label = "0"
        new_line.append( label )
    
        for i, item in enumerate( line ):
            if item == '' or float( item ) == 0.0:
                continue
            new_item = "%s:%s" % ( i + 1, item )
            new_line.append( new_item )
        new_line = " ".join( new_line )
        new_line += "\n"
        return new_line

    # ---
    input_file = csvfile
    output_file = outfile
    
    try:
        label_index = int( target_label_index )
    except IndexError:
        label_index = 0
    
    try:
        skip_headers = header_Flag
    except IndexError:
        skip_headers = 0
    
    i = open( input_file, 'r' )
    o = open( output_file, 'w' )
    
    reader = csv.reader( i )
    
    if skip_headers:
        headers = next(reader)
    
    for line in reader:
        if label_index == -1:
            label = '1'
        else:
            label = line.pop( label_index )
    
        new_line = construct_line( label, line )
        o.write( new_line )
    o.close()
    return

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
# main_class ='AlarmClock'
# negative_classes = ["Baby","Bag","CableModem","Dumbbell","Flashlight","Gloves","GPS","GraphicsCard","Headphone","HomeTheaterSystem"]
# not_in_negative_train_classes = ["Jewelry","Keyboard","Magazine_Subscriptions","Movies_TV","Projector","RiceCooker","Sandal","Vacuum","Video_Games"]
main_class ='baby'
negative_classes = ['automotive', 'beauty', 'camera_&_photo', 'cell_phones_&_service', 'dvd', 'health_&_personal_care', 'jewelry_&_watches',
 'kitchen_&_housewares', 'sports_&_outdoors', 'toys_&_games']
not_in_negative_train_classes=["apparel","electronics","music","books","grocery","magazines","outdoor_living","software","video","gourmet_food"]

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
del X_test_corpus

positive_class_test_vector = np.full(len(test_positive),1)
negative_class_test_vector = np.full(len(test_negative)+len(test_negative_not_in_train),-1)
test_class_label = np.concatenate((positive_class_test_vector,negative_class_test_vector))

del test_negative,test_positive,test_negative_not_in_train
del positive_class_test_vector, negative_class_test_vector
del train_negative,train_positive,train_class_negative_label

############## Begin -- preparing data for svm in document space 
from pandas import DataFrame
svm_train_data = DataFrame(X_train_unigrams)
svm_train_data.insert(column='class_label',value=train_class_label,loc=0)

svm_train_data.to_csv('training_data_svm_with_class.csv',index=False)
csvtolibsvm('training_data_svm_with_class.csv','training_data_svm_libsvm_format.data',0,'True')

svm_test_data = DataFrame(X_test_unigram)
svm_test_data.insert(column='class_label',value=test_class_label,loc=0)

svm_test_data.to_csv('testing_data_svm_with_class.csv',index=False)
csvtolibsvm('testing_data_svm_with_class.csv','testing_data_svm_libsvm_format.data',0,'True')

from libsvm.svmutil import *
y, x = svm_read_problem('training_data_svm_libsvm_format.data')
# options:
# -s svm_type : set type of SVM (default 0)
# 	0 -- C-SVC
# 	1 -- nu-SVC
# 	2 -- one-class SVM
# 	3 -- epsilon-SVR
# 	4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
# 	0 -- linear: u'*v
# 	1 -- polynomial: (gamma*u'*v + coef0)^degree
# 	2 -- radial basis function: exp(-gamma*|u-v|^2)
# 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)


m = svm_train(y, x, '-s 0 -t 0')
y_test, x_test = svm_read_problem('testing_data_svm_libsvm_format.data')
p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
print_results(y_test,p_label)    