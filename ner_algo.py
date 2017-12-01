import json
import csv, codecs, cStringIO
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score



#### SECTION I: Initializing and cleaning the dataset

# Fetching data from ner.csv
data = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
data.head()

# Drop null rows and check if any null values remaining
data.dropna(inplace=True)
print
print("Remaining Nulls: " + str(data[data.isnull().any(axis=1)].size))

# Fetch a smaller sample of data for testing (takes less computational time). 
# Change this for production

# arbirtrary size for now, just need code to run somwhat quickly
# randomize these? yuh
data_small = data[:20000]
data_valid = data[20001:30000]

# tag is the response variable
preds = list(data.columns.values)
preds.remove('tag')
y_small = data_small['tag']
x_small = data_small[preds]

# Not really useful for initial testing, but will be helpful for model tuning
# Split data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x_small, y_small, test_size=0.2, random_state=0)

## HARDCODE: Start ##

# Training data variables. All possible options of each indicator variable
pos_list = list(set(x_train['pos']))      # Word's part of speech options
shape_list = list(set(x_train['shape']))  # Shape of each word
word_list = list(set(data_small['word'])) # All the words in the small data set [Change to large later on]

# Different name entity tags available
tag_list = list(set(y_train.values)) 

## HARDCODE: End ##    



#### SECTION II: [ONLY RUN THIS SECTION ONCE] PreProcessing the Shape and Part-of-Speech dictionaries

# Init empty dictionaries to use for making predictions. Dict of Dicts
shape_probs = {} # {Key= shape : vale={key= entity-tage: value= probability}}
pos_probs = {}   # {Key= part-of-speech : vale={key= entity-tage: value= probability}}
word_probs = {}  # {Key= unique-word : vale={key= entity-tage: value= probability}}

alpha = 1.0

# Upload dictionary to specified csv file
# You need to create an empty csv file first with the correct name (name = indicator name in the original data file. Ex. shape.csv)
def csv_from_dict(csv_name, dict_name):
    # Truncate content of file if there is anything there
    f = open(csv_name, "w+")
    f.close()
    # Upload new content to the selected empty csv file
    with open(csv_name, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_name.items():
           # writer.writerow([key.encode('utf-8').strip(), value]) # This works for all dicts
           writer.writerow([key, value]) # This doesn't work for the words dictionary

# Creates a dict of dicts as follows -> {Key= indicator_variable : vale={key= entity-tage: value= probability}}
def create_entity_dict(indicator_list, indicator_name, indicator_dict_name):
    for item in indicator_list:
        tag_prob_dict = {}
        for tag in tag_list:
            count = 0
            for i in data_small[data_small[indicator_name] == item]['tag']:
                if i == tag:
                    count += 1
            tag_prob_dict.update({str(tag) : (1.0*count + alpha)/(len(data_small[data_small[indicator_name] == item]) + alpha*len(indicator_list))})
        indicator_dict_name[item] = tag_prob_dict
    csv_from_dict(indicator_name+'.csv', indicator_dict_name)

# Populate the shape dictionary from the small training data set
# create_entity_dict(shape_list,'shape', shape_probs) # Uncomment to recreate CSV file
    
# Populate the part-of-speech dictionary from the small training data set
# create_entity_dict(pos_list,'pos', pos_probs) # Uncomment to recreate CSV file

# Populate the words dictionary from the small training data set
# create_entity_dict(word_list,'word', word_probs) # Uncomment to recreate CSV file



#### SECTION III: Create Dicts from CSV files

def dict_from_csv(csv_name):
    with open(csv_name, 'rb') as csv_file:
        init_dict = dict(csv.reader(csv_file))
        new_dict = {}
        for key, value in init_dict.iteritems():
            new_dict[unicode(key, 'utf-8')] = ast.literal_eval(value)
    return new_dict

# Populate the pos, word, and shape dicts from their CSV files
pos_probs = dict_from_csv('pos.csv')
shape_probs = dict_from_csv('shape.csv')
word_probs = dict_from_csv('word.csv')

# Check if the dictionaries have correct probabilities
def test_dict_probs(indicator_dict, dict_name):    
    for indic_key, indic_val in indicator_dict.iteritems():
        sum_prob = 0.0
        for tag, prob in indic_val.iteritems():
            sum_prob += prob
            if sum_prob > 1:
                print "prob_sum: ", sum_prob
                print "PROB SUM ERROR AT: "+dict_name+" for key"+indic_key
                print "______________________"
                break;
    print "done with "+dict_name
    print "__________________"

# Uncomment this to run individual tests
# test_dict_probs(pos_probs, "pos")
# test_dict_probs(shape_probs, "shape")
# test_dict_probs(word_probs, "word")



#### SECTION IV: Single Indicator Variable Prediction Algorithm

# Baseline Accuracy
num_O = len(data[data['tag'] == 'O'])
percent = 1.0*num_O/len(data)
print '-----------------------------------------------'
print "Baseline accuracy for all 'O' predictor: " + "%.4f" % percent
print '-----------------------------------------------'

# Generic function that takes in a single predictive indicator and trains/validates the modal
def train_validate_model(data_set, indicator_name, indicator_dict_name):
    # training and validation prediction
    
    pred_train = []
    pred_valid = []

    count_correct = 0
    data_set_len = len(data_set)
    for i in range(data_set_len):
        try:
            # get the key corresponding to the max value in dict
            dict_use = indicator_dict_name[data_set.iloc[i][indicator_name]] # this might be buggy, is 'pos' correct?
            pred_tag = max(dict_use.iterkeys(), key=(lambda key: dict_use[key]))
        except:
            pred_tag = 'O'
            do='nothing' # figure out this case later
        pred_train.append(pred_tag)
        if data_set.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / data_set_len
    print "Train Accuracy using " + indicator_name + ': ' + str(train_accuracy)

    # validation prediction
    count_correct = 0
    data_valid_len = len(data_valid)
    for i in range(data_valid_len):
        try:
            # get the key corresponding to the max value in dict
            dict_use = indicator_dict_name[data_valid.iloc[i][indicator_name]]
            pred_tag = max(dict_use.iterkeys(), key=(lambda key: dict_use[key]))
        except:
            pred_tag = 'O' # figure out this case later
        pred_valid.append(pred_tag)
        if data_valid.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    valid_accuracy = 1.0*count_correct / data_valid_len
    print "Validation Accuracy using " + indicator_name + ': ' + str(valid_accuracy)

# # Make predictions based off of "shape"
# train_validate_model(data_small,'shape',shape_probs)
# print '-----------------------------------------------'

# # Make predictions based off of "words"
# train_validate_model(data_small,'word',word_probs)
# print '-----------------------------------------------'

# # Make predictions based off of "part-of-speech"
# train_validate_model(data_small,'pos',pos_probs)
# print '-----------------------------------------------'


#### SECTION V: Generic Multiple Indicator Variables's Prediction Algorithm

# then we can incorporate multiple features in the algorithm by multiply probabilities and taking a max
# or adding log probabilities. This is more complicated, but will evertually be the basis of the final model. 

def multi_var_ner(shape_coeff, pos_coeff, word_coeff):

    pred_train = []
    pred_valid = []

    # # training prediction
    count_correct = 0
    for i in range(len(data_small)):
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:  
            # try, except to ignore one value when a word has not been seen before!
            try:      
                pos_p = pos_probs[data_small.iloc[i]['pos']][tag]
            except:
                pos_p = 1.0
            try:
                shape_p = shape_probs[data_small.iloc[i]['shape']][tag]
            except:
                shape_p = 1.0
            try:
                word_p = word_probs[data_small.iloc[i]['word']][tag]
            except:
                word_p = 1.0
            prob = pos_p * shape_p * word_p 
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        pred_tag = max_tag
        pred_train.append(pred_tag)
        if data_small.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_small)
    print "Train Accuracy using "+pos_coeff+" pos, "+shape_coeff+" shape, and "+word_coeff+" word: " + str(train_accuracy)

    # validation prediction
    count_correct = 0
    for i in range(len(data_valid)):
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:        
            # try, except to ignore one value when a word has not been seen before!
            try:      
                pos_p = pos_probs[data_valid.iloc[i]['pos']][tag]
            except:
                pos_p = 1.0
            try:
                shape_p = shape_probs[data_valid.iloc[i]['shape']][tag]
            except:
                shape_p = 1.0
            try:
                word_p = word_probs[data_valid.iloc[i]['word']][tag]
            except:
                word_p = 1.0
            prob = pos_p * shape_p * word_p 
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        pred_tag = max_tag
        pred_valid.append(pred_tag)
        if data_valid.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_valid)
    print "Validation Accuracy using "+pos_coeff+" pos, "+shape_coeff+" shape, and "+word_coeff+" word: " + str(train_accuracy)
    print '-----------------------------------------------'

# all coeffs = 1, all vars are wieghted equally
# multi_var_ner(1.0, 1.0, 1.0)

# SECTION VI: TUNING 1. FIRST DO EXPLORATORY TUNING ON THE ALPHA (HALLUCINATION) PARAMETER ON INDIVIDUAL MODELS


# SECTION VII: TUNING 2. TUNE HYPERPARAMETERS TO WEIGHT OUR INDIVIDUAL MODELS
# intuition: Word provides the strongest individual model (as expected), then POS, then Shape

# multi_var_ner(shape_coeff, pos_coeff, word_coeff)


