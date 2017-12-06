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
import time
import collections
import sys
import warnings

# ignores the f1 warning
warnings.filterwarnings('ignore')

# ------------------------------------------------
# SECTION I: Initializing and cleaning the dataset
# ------------------------------------------------

# Fetching data from ner.csv
data = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
data.head()

# Drop null rows and check if any null values remaining
data.dropna(inplace=True)

# Fetch a smaller sample of data for testing (takes less computational time). 
# Change this for production
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

# Training data variables. All possible options of each indicator variable
pos_list = list(set(x_train['pos']))      # Word's part of speech options
shape_list = list(set(x_train['shape']))  # Shape of each word
word_list = list(set(data_small['word'])) # All the words in the small data set [Change to large later on]

# Different name entity tags available
tag_list = list(set(y_train.values)) 

print "pos_list", pos_list 
print "shape_list", shape_list
print "word_list", word_list
print "tag_list", tag_list

end = time.time()

# ------------------------------------------------------------------
# SECTION II: PreProcessing the Shape and Part-of-Speech dictionaries
# [ONLY RUN THIS SECTION ONCE]
# -------------------------------------------------------------------

# Init empty dictionaries to use for making predictions. Dict of Dicts
shape_probs = {} # {Key= shape : vale={key= entity-tage: value= probability}}
pos_probs = {}   # {Key= part-of-speech : vale={key= entity-tage: value= probability}}
word_probs = {}  # {Key= unique-word : vale={key= entity-tage: value= probability}}

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
            writer.writerow([key.encode('utf-8').strip(), value]) # This works for all dicts
            #writer.writerow([key, value]) # This doesn't work for the words dictionary

# Creates a dict of dicts as follows -> {Key= indicator_variable : vale={key= entity-tage: value= probability}}
def create_entity_dict(indicator_list, indicator_name, indicator_dict_name, alpha):
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

# Uncomment the lines below to recreate the CSV indicator prob files (AKA do not uncomment)

# # Populate the shape dictionary from the small training data set
# create_entity_dict(shape_list,'shape', shape_probs, 2.0) # Uncomment to recreate CSV file
    
# # Populate the part-of-speech dictionary from the small training data set
# create_entity_dict(pos_list,'pos', pos_probs, 2.0) # Uncomment to recreate CSV file

# # Populate the words dictionary from the small training data set
# create_entity_dict(word_list,'word', word_probs, 0.01) # Uncomment to recreate CSV file

# ----------------------------------------------------
# SECTION III: Pull the Indicator Dicts from CSV files
# ----------------------------------------------------

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

# Function that checked if the CSV files have the correct probabilities (for testing)
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


# ----------------------------------------------------------
# SECTION IV: Single Indicator Variable Prediction Algorithm
# ----------------------------------------------------------

# Baseline Accuracy
def baseline(): 
    num_O = len(data[data['tag'] == 'O'])
    percent = 1.0*num_O/len(data)
    print '-----------------------------------------------'
    print "Baseline accuracy for all 'O' predictor: " + "%.4f" % percent
    print '-----------------------------------------------'

# Generic function that takes in a single predictive indicator and trains/validates the modal
def train_validate_model(data_set, indicator_name, indicator_dict_name):
    # training prediction
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
    print '------------- Single Indicator --------------'
    print "     Training F1-Score using " + indicator_name + ': ' +  str(f1_score(data_set['tag'], pred_train, labels=tag_list, average="weighted"))

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
    print "     Validation F1-Score using " + indicator_name + ': ' +  str(f1_score(data_valid['tag'], pred_valid, labels=tag_list, average="weighted"))
    print "     Validation Accuracy using " + indicator_name + ': ' + str(valid_accuracy)
    print '-----------------------------------------------'

# --------------------------------------------------------------
# SECTION V: Generic Multi Indicator Entity Prediction Algorithm
# --------------------------------------------------------------

# then we can incorporate multiple features in the algorithm by multiply probabilities and taking a max
# or adding log probabilities. This is more complicated, but will evertually be the basis of the final model. 

# feature_list is a list of tuples of name and probs for the indicators you want to combine in the model
# i.e. [('pos', pos_probs), ('shape', shape_probs), ('word', word_list)]
def combined_model(feature_list):

    pred_train = []
    pred_valid = []

    # training prediction
    count_correct = 0
    for i in range(len(data_small)):
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:  
            prob = 1.0
            names = []
            for name, probs in feature_list:
                names.append(name)
                # try, except to ignore one value when a word has not been seen before!
                try:      
                    p = probs[data_small.iloc[i][name]][tag]
                except:
                    # p = total_tags_prob[tag] # performs worse that 1.0 (WHY?)
                    p = 1.0
                prob *= p
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        prob = 0.0
        pred_tag = max_tag
        pred_train.append(pred_tag)
        if data_small.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_small)
    names_str = names[0]
    for n in names[1:]: 
        names_str = names_str + ", "
        names_str = names_str + n
    print '------------- Combo Algorithm --------------'
    print "     Train F1-Score using " + '(' + names_str + ')' ": " +  str(f1_score(data_small['tag'], pred_train, labels=tag_list, average="weighted"))

    # validation prediction
    count_correct = 0
    for i in range(len(data_valid)):
        max_prob = 0.0
        max_tag = ''
        for tag in tag_list:  
            prob = 1.0
            names = []
            for name, probs in feature_list:
                names.append(name)
                # try, except to ignore one value when a word has not been seen before!
                try:      
                    p = probs[data_valid.iloc[i][name]][tag]
                except:
                    # p = total_tags_prob[tag] # performs worse that 1.0 (WHY?)
                    p = 1.0
                prob *= p
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        prob = 0.0
        pred_tag = max_tag
        pred_valid.append(pred_tag)
        if data_valid.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / len(data_valid)
    names_str = names[0]
    for n in names[1:]: 
        names_str = names_str + ", "
        names_str = names_str + n
    print "     Validation F1-Score using " + '(' + names_str + ')' ": " +  str(f1_score(data_valid['tag'], pred_valid, labels=tag_list, average="weighted"))
    print "     Validation Accuracy using " + '(' + names_str + ')' ": " + str(train_accuracy)
    print '--------------------------------------------'

# -------------------------------------------------
# SECTION VI: Tuning Alpha(HALLUCINATION) Parameter 
# -------------------------------------------------

# function to re-create initial dictionaries with different alpha values (not 1.0)
def create_entity_dict2(indicator_list, indicator_name, indicator_dict_name, alpha):
    print '--------------- Alpha Tuning ---------------'
    print "     Alpha: " + str(alpha)
    for item in indicator_list:
        tag_prob_dict = {}
        for tag in tag_list:
            count = 0
            for i in data_small[data_small[indicator_name] == item]['tag']:
                if i == tag:
                    count += 1
            tag_prob_dict.update({str(tag) : (1.0*count + alpha)/(len(data_small[data_small[indicator_name] == item]) + alpha*len(indicator_list))})
        indicator_dict_name[item] = tag_prob_dict
    return indicator_dict_name

# function takes in lists (as demonstrated below) so we can adjust alpha and test results
def test_for_alpha(indicator_lists, indicator_names, indicator_dict_names, alphas):
    tuple_list = []
    for i in range(len(indicator_lists)):
        dicto = create_entity_dict2(indicator_lists[i], indicator_names[i], indicator_dict_names[i], alphas[i])
        tuple_list.append((indicator_names[i], dicto))
    combined_model(tuple_list)

# --------------------------------------------------
# SECTION VII: NER for Facebook Messenger 
# --------------------------------------------------

# def camel(s):
#     return (s != s.lower() and s != s.upper())

# def local_shapes(spacy_shape):
#     if spacy_shape.islower():
#         return u'lowercase'
#     elif spacy_shape.isupper():
#         return u'uppercase'
#     elif not spacy_shape.isnumeric():
#        return u'mixedcase' 
#     elif spacy_shape[0].isupper():
#         return u'capitalized'
#     elif spacy_shape.isnumeric():
#         return u'number'
#     elif camel(spacy_shape):
#         return u'camelcase'
#     elif spacy_shape[len(spacy_shape)-1] == '.':
#         return u'ending-dot',
#     elif '-' in spacy_shape:
#         return u'contains-hyphen'
#     elif spacy_shape in string.punctuation:
#         return u'punct'
#     else:
#         return u'other'

# def messenger_ner(sentence):
#     nlp = spacy.load('en')
#     doc = nlp(sentence.decode('utf-8'))

#     sentence_dict = {}
#     for token in doc:
#         local_shapes(token.shape_)
#         sentence_dict[token] = (token.tag_, local_shapes(token.shape_))

#     # print sentence_dict
#     tagged_sentence = {}
#     for word, value in sentence_dict.iteritems():
#         word_pos, word_shape = value
#         prob = 0.0
#         max_prob = 0.0
#         max_tag = ''
#         for tag in tag_list:  
#             # try, except to ignore one value when a word has not been seen before!
#             try:      
#                 pos_p = pos_probs[word_pos][tag]
#             except:
#                 pos_p = 1.0
#             try:
#                 shape_p = shape_probs[word_shape][tag]
#             except:
#                 shape_p = 1.0
#             try:
#                 word_p = word_probs[word][tag]
#             except:
#                 word_p = 1.0
#             prob = pos_p * shape_p * word_p

#             if prob > max_prob:
#                 max_prob = prob
#                 max_tag = tag
#         tagged_sentence[word] = max_tag

#     return tagged_sentence

# Main Function, called from terminal to test all segments of the code
# Explained in the Readme.md file

# ------------------------------------------------
# SECTION VIII: Terminal Commands to run the code 
# ------------------------------------------------

# if len(sys.argv) > 1: # if user gave some input
#     if sys.argv[1] == "baseline":
#         baseline()
#     elif sys.argv[1] == "single": 
#         train_validate_model(data_small, sys.argv[2], word_probs) # Single indicator algo
#     elif sys.argv[1] == "combo":
#         if sys.argv[2] == "pos_word":
#             combined_model([('pos', pos_probs), ('word', word_probs)])
#         elif sys.argv[2] == "pos_shape":
#             combined_model([('pos', pos_probs), ('shape', shape_probs)])
#         elif sys.argv[2] == "word_shape":
#             combined_model([('shape', shape_probs), ('word', word_probs)])
#         elif sys.argv[2] == "pos_word_shape":
#             combined_model([('pos', pos_probs), ('shape', shape_probs), ('word', word_probs)])
#     elif sys.argv[1] == "alpha":
#         test_for_alpha([pos_list, shape_list, word_list], ['pos', 'shape', 'word'], [pos_probs, shape_probs, word_probs], [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])])
# else:
#     print "You have entered an incorrect command. Please check the code documentation on how to run the code."

# ---------------------------------------------------
# SECTION VIIII: Scrapbook - delete before submission 
# ---------------------------------------------------

# Performance of Combo without these is better (when except is just auto 1.0)
# Probability of each tag occuring in our FULL dataset
# total_tags_prob = {'I-art': 0.026702957257148524, 'B-nat': 0.02155310121469845, 'B-gpe': 1.560978587089311, 'B-art': 0.04138958374858021, 'I-tim': 0.5999582289454335, 'B-org': 1.9201333621979586, 'I-per': 1.654248202080351, 'B-geo': 3.5709483269166764, 'I-org': 1.573376388672987, 'I-geo': 0.7053395424066803, 'O': 84.69338806168001, 'I-eve': 0.0283242082334754, 'B-eve': 0.03318796116245602, 'I-gpe': 0.0218392043281679, 'B-tim': 1.922517554810204, 'I-nat': 0.007247945541226028, 'B-per': 1.6188667837146293}
# Probability of each tag occuring in our SMALL dataset
# total_tags_prob = {'I-art': 0.11, 'B-nat': 0.045, 'B-gpe': 2.535, 'B-art': 0.185, 'I-tim': 0.25, 'B-org': 1.895, 'O': 85.56, 'B-geo': 2.555, 'I-org': 1.395, 'I-geo': 0.395, 'I-per': 1.775, 'I-eve': 0.07, 'B-eve': 0.09, 'I-gpe': 0.13, 'B-tim': 1.56, 'I-nat': 0.025, 'B-per': 1.425}

# possible word shapes

# [u'mixedcase', 
# u'lowercase', 
# u'camelcase', 
# u'uppercase', 
# u'capitalized',
# u'number', 
# u'abbreviation', ######
# u'punct', 
# u'other', 
# u'ending-dot', 
# u'contains-hyphen']
