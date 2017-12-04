import json
import csv, codecs, cStringIO
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
import spacy
import string



#### SECTION I: Initializing and cleaning the dataset

# Fetching data from ner.csv
data = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
data.head()

# Drop null rows and check if any null values remaining
data.dropna(inplace=True)
# print
# print("Remaining Nulls: " + str(data[data.isnull().any(axis=1)].size))

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
                print "PROB SUM ERROR AT: "+dict_name+" for key"+indic_key
        print "prob_sum: ", sum_prob
    print "done with "+dict_name
    print "__________________"

# Uncomment this to run individual tests
# test_dict_probs(pos_probs, "pos")
# test_dict_probs(shape_probs, "shape")
# test_dict_probs(word_probs, "word")



#### SECTION IV: Single Indicator Variable Prediction Algorithm

# Baseline Accuracy
# num_O = len(data[data['tag'] == 'O'])
# percent = 1.0*num_O/len(data)
# print '-----------------------------------------------'
# print "Baseline accuracy for all 'O' predictor: " + "%.4f" % percent
# print '-----------------------------------------------'

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

# Make predictions based off of "words"
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

    # training prediction
    count_correct = 0
    for i in range(len(data_small)):
        prob = 0.0
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
            
            if shape_coeff == pos_coeff == word_coeff == -1.0:
                prob = pos_p * shape_p * word_p
            else:
                prob = (pos_coeff*pos_p) + (shape_coeff*shape_p) + (word_coeff*word_p)

            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        pred_tag = max_tag
        pred_train.append(pred_tag)
        if data_small.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_small)
    print "Train Accuracy using "+str(pos_coeff)+" pos, "+str(shape_coeff)+" shape, and "+str(word_coeff)+" word: " + str(train_accuracy)

    # validation prediction
    count_correct = 0
    for i in range(len(data_valid)):
        prob = 0.0
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:        
            #try, except to ignore one value when a word has not been seen before!
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
            
            if shape_coeff == pos_coeff == word_coeff == -1.0:
                prob = pos_p * shape_p * word_p
            else:
                prob = (pos_coeff*pos_p) + (shape_coeff*shape_p) + (word_coeff*word_p) 
            
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        pred_tag = max_tag
        pred_valid.append(pred_tag)
        if data_valid.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_valid)
    print "Validation Accuracy using "+str(pos_coeff)+" pos, "+str(shape_coeff)+" shape, and "+str(word_coeff)+" word: " + str(train_accuracy)
    print '-----------------------------------------------'

# SECTION VII: TUNING 2. TUNE HYPERPARAMETERS TO WEIGHT OUR INDIVIDUAL MODELS
# intuition: Word provides the strongest individual model (as expected), then POS, then Shape

# print "BaseCase - Virtbre Multiplication"
# multi_var_ner(-1.0,-1.0,-1.0)

# multi_var_ner(0,0,1)

# x = 1
# for i in range(1,4):
#     for j in range(1,4):
#         for k in range(1,4):
#             multi_var_ner(i,j,k)


# SECTION VIII: NER for Facebook Messenger
# [u'mixedcase', 
# u'lowercase', 
# u'camelcase', 
# u'uppercase', 
# u'capitalized', 
# u'number', 
# u'abbreviation', 
# u'punct', 
# u'other', 
# u'ending-dot', 
# u'contains-hyphen']

def camel(s):
    return (s != s.lower() and s != s.upper())

def local_shapes(spacy_shape):
    if spacy_shape.islower():
        return u'lowercase'
    elif spacy_shape.isupper():
        return u'uppercase'
    elif not spacy_shape.isnumeric():
       return u'mixedcase' 
    elif spacy_shape[0].isupper():
        return u'capitalized'
    elif spacy_shape.isnumeric():
        return u'number'
    elif camel(spacy_shape):
        return u'camelcase'
    elif spacy_shape[len(spacy_shape)-1] == '.':
        return u'ending-dot',
    elif '-' in spacy_shape:
        return u'contains-hyphen'
    elif spacy_shape in string.punctuation:
        return u'punct'
    else:
        return u'other'

def messenger_ner(sentence):
    nlp = spacy.load('en')
    doc = nlp(sentence.decode('utf-8'))

    sentence_dict = {}
    for token in doc:
        local_shapes(token.shape_)
        sentence_dict[token] = (token.tag_, local_shapes(token.shape_))

    # print sentence_dict
    tagged_sentence = {}
    for word, value in sentence_dict.iteritems():
        word_pos, word_shape = value
        prob = 0.0
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:  
            # try, except to ignore one value when a word has not been seen before!
            try:      
                pos_p = pos_probs[word_pos][tag]
            except:
                pos_p = 1.0
            try:
                shape_p = shape_probs[word_shape][tag]
            except:
                shape_p = 1.0
            try:
                word_p = word_probs[word][tag]
            except:
                word_p = 1.0
            prob = pos_p * shape_p * word_p

            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        tagged_sentence[word] = max_tag

    return tagged_sentence

# print "Tagged: ", messenger_ner("Hi! My name is John and I go to Harvard University")

#---------------------------------------------------------------------------
# def combined_model(feature_list):
#     pred_train = []
#     pred_valid = []

#     # # training prediction
#     count_correct = 0
#     for i in range(len(data_small)):
#         #if i % 100 == 0:
#         #print(i)
#         #tag_probs = {}
#         max_prob = 0.0
#         max_tag = ''
#         for tag in tag_list:  
#             prob = 1.0
#             names = []
#             for name, probs in feature_list:
#                 names.append(name)
#                 # try, except to ignore one value when a word has not been seen before!
#                 try:      
#                     p = probs[data_small.iloc[i][name]][tag]
#                 except:
#                     p = 1.0
#                 prob *= p
#             if prob > max_prob:
#                 max_prob = prob
#                 max_tag = tag
#         #pred_tag = max(tag_probs.iterkeys(), key=(lambda key: tag_probs[key]))
#         pred_tag = max_tag
#         pred_train.append(pred_tag)
#         if data_small.iloc[i]['tag'] == pred_tag:
#             count_correct += 1
#     train_accuracy = 1.0*count_correct / len(data_small)
#     names_str = names[0]
#     for n in names[1:]: 
#         names_str = names_str + ", "
#         names_str = names_str + n
#     print "Train Accuracy using " + '(' + names_str + ')' + ": " + str(train_accuracy)

#     # validation prediction
#     count_correct = 0
#     for i in range(len(data_valid)):
#         #if i % 100 == 0:
#         #    print(i)
#         #tag_probs = {}
#         max_prob = 0.0
#         max_tag = ''
#         for tag in tag_list:  
#             prob = 1.0
#             names = []
#             for name, probs in feature_list:
#                 names.append(name)
#                 # try, except to ignore one value when a word has not been seen before!
#                 try:      
#                     p = probs[data_valid.iloc[i][name]][tag]
#                 except:
#                     p = 1.0
#                 prob *= p
#             if prob > max_prob:
#                 max_prob = prob
#                 max_tag = tag
#         #pred_tag = max(tag_probs.iterkeys(), key=(lambda key: tag_probs[key]))
#         pred_tag = max_tag
#         pred_valid.append(pred_tag)
#         if data_valid.iloc[i]['tag'] == pred_tag:
#             count_correct += 1
#     train_accuracy = 1.0*count_correct / len(data_valid)
#     names_str = names[0]
#     for n in names[1:]: 
#         names_str = names_str + ", "
#         names_str = names_str + n
#     print "Validation Accuracy using " + '(' + names_str + ')' ": " + str(train_accuracy)
#     print '-----------------------------------------------'

# Below lines test combined models for test and validation accuracy
#combined_model([('pos', pos_probs), ('shape', shape_probs), ('word', word_list)])
#combined_model([('pos', pos_probs), ('shape', shape_probs)])
#combined_model([('word', word_probs), ('shape', shape_probs)])
#combined_model([('pos', pos_probs), ('word', word_probs)])

# all coeffs = 1, all vars are wieghted equally

# SECTION VI: TUNING 1. FIRST DO EXPLORATORY TUNING ON THE ALPHA (HALLUCINATION) PARAMETER ON INDIVIDUAL MODELS

# UNCOMMENT WHEN ALPHA TESTING
#print
#print '-----------------------------------------------'
#print "Section VI: ALPHA TESTING" + " Using Part of Speech (pos) and Shape:"
#print '-----------------------------------------------'

# function to re-create initial dictionaries with different alpha values (not 1.0)
# def create_entity_dict2(indicator_list, indicator_name, indicator_dict_name, alpha):
#     print "Alpha: " + str(alpha)
#     for item in indicator_list:
#         tag_prob_dict = {}
#         for tag in tag_list:
#             count = 0
#             for i in data_small[data_small[indicator_name] == item]['tag']:
#                 if i == tag:
#                     count += 1
#             tag_prob_dict.update({str(tag) : (1.0*count + alpha)/(len(data_small[data_small[indicator_name] == item]) + alpha*len(indicator_list))})
#         indicator_dict_name[item] = tag_prob_dict
#     return indicator_dict_name

# how to use two previously made functions
# create_entity_dict(word_list,'word', word_probs)
# train_validate_model(data_small,'pos',pos_probs)

# function takes in lists (as demonstrated below) so we can adjust alpha and test results
# def test_for_alpha(indicator_lists, indicator_names, indicator_dict_names, alphas):
#     tuple_list = []
#     for i in range(len(indicator_lists)):
#         dicto = create_entity_dict2(indicator_lists[i], indicator_names[i], indicator_dict_names[i], alphas[i])
#         tuple_list.append((indicator_names[i], dicto))
#     combined_model(tuple_list)

# UNCOMMENT TO PERFORM ALPHA TESTING ON POS AND SHAPE
#test_for_alpha([pos_list, shape_list], ['pos', 'shape'], [pos_probs, shape_probs], [1.0, 1.0])
#test_for_alpha([pos_list, shape_list], ['pos', 'shape'], [pos_probs, shape_probs], [0.1, 0.1])
#test_for_alpha([pos_list, shape_list], ['pos', 'shape'], [pos_probs, shape_probs], [1.0, 0.1])
#test_for_alpha([pos_list, shape_list], ['pos', 'shape'], [pos_probs, shape_probs], [0.1, 1.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [1.0, 5.0, 10.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [1.0, 10.0, 5.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [5.0, 1.0, 10.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [5.0, 10.0, 1.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [10.0, 5.0, 1.0])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [10.0, 1.0, 5.0])

# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.2, 0.4, 0.6])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.2, 0.6, 0.4])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.4, 0.2, 0.6])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.4, 0.6, 0.2])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.6, 0.2, 0.4])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.6, 0.4, 0.2])

# print "TESTING FOR WORD ALPHA"
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.5, 0.9])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.5, 0.5])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.5, 0.1])

# print "TESTING FOR SHAPE ALPHA"
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.9, 0.5])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.5, 0.5])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.1, 0.5])

# print "TESTING FOR POS ALPHA"
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.9, 0.5, 0.5])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.5, 0.5, 0.5])
# test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [0.1, 0.5, 0.5])


# SECTION VII: TUNING 2. TUNE HYPERPARAMETERS TO WEIGHT OUR INDIVIDUAL MODELS
# intuition: Word provides the strongest individual model (as expected), then POS, then Shape

# multi_var_ner(shape_coeff, pos_coeff, word_coeff)


