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
    
############################
####Similarity Functions####
############################
def Cos_similarity(v_1,v_2):
    #check if the vecors are nonzero vectors
    if not any(v_2):
        return 1
    if not any(v_1):
        return 1
         
    from scipy import spatial
    calculated_cosine_similarity = 1 - spatial.distance.cosine(v_1, v_2)
    return calculated_cosine_similarity

def dice_Similarity(v_1,v_2):
    
    comp_dot = np.dot(v_1,v_2)
    computed_squares =  np.dot(v_1,v_1)+np.dot(v_2,v_2)
    calculated_dice_similarity = 2 * comp_dot/computed_squares 
    
    return calculated_dice_similarity

def jaccard_Similarity(v_1,v_2):
    
    comp_dot = np.dot(v_1,v_2)
    computed_squares = np.dot(v_1,v_1)+np.dot(v_2,v_2)-comp_dot
    calculated_jacard_similarity = comp_dot/computed_squares 
    
    return calculated_jacard_similarity

def gow_similarity(v_1,v_2):
       #check if the vecors are nonzero vectors
    if not any(v_2):
        return 1
    if not any(v_1):
        return 1
          
    zero_Vec=np.zeros(len(v_1))
    from scipy.spatial import distance
    sum_tmps=0
    for i in range(0,len(v_1)):
        tmp_term_1=v_1[i]/distance.euclidean(v_1,zero_Vec)
        tmp_term_2=v_2[i]/distance.euclidean(v_2,zero_Vec)
        sum_tmps+=np.abs(tmp_term_1-tmp_term_2)
        
    sum_tmps= sum_tmps/len(v_1)
    calculated_gow_similarity=1-sum_tmps
    return calculated_gow_similarity


def Lor_similarity(v_1,v_2):
    sum_tmps=0
    for i in range(0,len(v_1)):
        sum_tmps+=np.log(1+np.abs(v_1[i]-v_2[i]))
        
    calculated_lor_similarity=1-sum_tmps
    return calculated_lor_similarity

##################################################
#####Rocchilio method for calculating centroids###
##################################################
def calculate_centriods(matrix,positive_docs,number_of_docs):
    import math as ma
    
    alpha=16
    beta=4
    
    centers=[]
    
    for i in range(0,len(matrix[0])):
#        print(i)
        positive_term=0
        for j in range(0,positive_docs):
            if matrix[j][i]==0:
                continue
            else:
                positive_term=positive_term + matrix[j][i]/ma.sqrt(matrix[j][i])
        positive_term=alpha*positive_term/positive_docs
       
        negative_term=0
        for j in range(positive_docs,number_of_docs):
            if matrix[j][i]==0:
                continue
            else:
                negative_term=negative_term + matrix[j][i]/ma.sqrt(matrix[j][i])
        negative_term=negative_term*beta/abs(number_of_docs-positive_docs)
        
        centers.append(positive_term-negative_term)
         
    return centers

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

main_class ='camera_&_photo'
negative_classes = ['grocery', 'kitchen_&_housewares', 'toys_&_games', 'gourmet_food', 'video', 'cell_phones_&_service', 'health_&_personal_care',
 'dvd', 'sports_&_outdoors', 'jewelry_&_watches']
not_in_negative_train_classes=["apparel","books","baby","automotive","beauty","music","outdoor_living","software","electronics","magazines"]


train_positive = pd.read_csv (dataset_path+main_class+'/'+main_class+' train_positive_samples.csv',header=None)
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
X_train_unigrams = vectorizer_unigrams.fit_transform(norm_corpus)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
mutual_info=mutual_info_classif(X_train_unigrams, train_class_label)
mutual_info = pd.Series(mutual_info)
mutual_info.index = vectorizer_unigrams.get_feature_names()
kk=mutual_info.sort_values(ascending = False)
ll=list(kk.index[0:])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_unigrams = TfidfVectorizer(stop_words ='english',ngram_range = (1, 1),vocabulary=ll)
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

center_of_positive_onegram_train=calculate_centriods(X_train_unigrams,len(train_positive),len(corpus)) #for unigram

del train_negative,train_positive,train_class_negative_label

cos_similairty_train_onegram=[]
lor_similarity_train_onegram=[]
gow_similairty_train_onegram=[]
dice_similarity_train_onegram=[]
jaccard_similairty_train_onegram=[]


from scipy import spatial
for doc in range(0,len(corpus)):
    cos_similairty_train_onegram.append(spatial.distance.cosine(center_of_positive_onegram_train,X_train_unigrams[doc]))
    lor_similarity_train_onegram.append(Lor_similarity(center_of_positive_onegram_train,X_train_unigrams[doc]))
    gow_similairty_train_onegram.append(gow_similarity(center_of_positive_onegram_train,X_train_unigrams[doc]))
    dice_similarity_train_onegram.append(spatial.distance.dice(center_of_positive_onegram_train,X_train_unigrams[doc]))
    jaccard_similairty_train_onegram.append(spatial.distance.jaccard(center_of_positive_onegram_train,X_train_unigrams[doc]))
    

from pandas import DataFrame
training_data_df=DataFrame()

training_data_df.insert(column='cosine_uni',value=cos_similairty_train_onegram,loc=0)
training_data_df.insert(column='Lor_uni',value=lor_similarity_train_onegram,loc=1)
training_data_df.insert(column='Dice_uni',value=dice_similarity_train_onegram,loc=2)
training_data_df.insert(column='Jaccard_uni',value=jaccard_similairty_train_onegram,loc=3)
training_data_df.insert(column='gow_uni',value=gow_similairty_train_onegram,loc=4)

training_data_df.insert(column='class_label',value=train_class_label,loc=0)

training_data_df.to_csv('training_cbs_with_class.csv',index=False)
csvtolibsvm('training_cbs_with_class.csv','training_cbs_libsvm_format.data',0,'True')



cos_similairty_test_onegram=[]
Lor_similarity_mat_test_onegram=[]
gow_similairty_mat_test_onegram=[]
Dice_similarity_test_onegram=[]
Jaccard_similairty_test_onegram=[]

from scipy import spatial
for doc in range(0,len(X_test_unigram)):
    cos_similairty_test_onegram.append(spatial.distance.cosine(center_of_positive_onegram_train,X_test_unigram[doc]))
    Lor_similarity_mat_test_onegram.append(Lor_similarity(center_of_positive_onegram_train,X_test_unigram[doc]))
    gow_similairty_mat_test_onegram.append(gow_similarity(center_of_positive_onegram_train,X_test_unigram[doc]))
    Dice_similarity_test_onegram.append(spatial.distance.dice(center_of_positive_onegram_train,X_test_unigram[doc]))
    Jaccard_similairty_test_onegram.append(spatial.distance.jaccard(center_of_positive_onegram_train,X_test_unigram[doc]))
    
from pandas import DataFrame
testing_data_df=DataFrame()

testing_data_df.insert(column='cosine_uni',value=cos_similairty_test_onegram,loc=0)
testing_data_df.insert(column='Lor_uni',value=Lor_similarity_mat_test_onegram,loc=1)
testing_data_df.insert(column='Dice_uni',value=Dice_similarity_test_onegram,loc=2)
testing_data_df.insert(column='Jaccard_uni',value=Jaccard_similairty_test_onegram,loc=3)
testing_data_df.insert(column='gow_uni',value=gow_similairty_mat_test_onegram,loc=4)

testing_data_df.insert(column='class_label',value=test_class_label,loc=0)
testing_data_df.to_csv('testing_cbs_with_class.csv',index=False)
csvtolibsvm('testing_cbs_with_class.csv','testing_cbs_libsvm_format.data',0,'True')


#instead of the following line
# from svmutil import *
####### use the following line
from libsvm.svmutil import *
y, x = svm_read_problem('training_cbs_libsvm_format.data')

 
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


m = svm_train(y, x, '-s 0 -t 2')
y_test, x_test = svm_read_problem('testing_cbs_libsvm_format.data')
p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

print_results(y_test,p_label)   
