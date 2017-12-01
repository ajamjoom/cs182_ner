import json
import csv, codecs, cStringIO
import pandas as pd
import numpy as np
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
data[data.isnull().any(axis=1)].size

# Fetch a smaller sample of data for testing (takes less computational time). 
# Change this for production
data_small = data[:20000]
data_valid = data[20001:30000]

preds = list(data.columns.values)
preds.remove('tag')
y_small = data_small['tag']
x_small = data_small[preds]

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
           # writer.writerow([key.encode('utf-8').strip(), value])
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
            new_dict[unicode(key, 'utf-8')] = value
    return new_dict

# Populate the pos, word, and shape dicts from their CSV files
pos_probs = dict_from_csv('pos.csv')
shape_probs = dict_from_csv('shape.csv')
word_probs = dict_from_csv('word.csv')

# TEST to debug why reading from CSV outputs an error

# print "TEST: ", test_pos_probs
# print "GENERATED: ", pos_probs
# print test_pos_probs
# value = { k : test_pos_probs[k] for k in set(test_pos_probs) - set(pos_probs) }
# print "DIFFS: ", value
# print "TEST: ", test_pos_probs
# print "REAL", pos_probs

### WHY WOULD THEY NOT BE EQUAL?
# if test_pos_probs == pos_probs:
#     print "Both Dicts are the same"
# else:
#     print "fucked up shit"


#### SECTION IV: Single Indicator Variable Prediction Algorithm

pred_train = []
pred_valid = []

# Baseline Accuracy
num_O = len(data[data['tag'] == 'O'])
percent = 1.0*num_O/len(data)
print "Percent of data without a tag (Baseline accuracy if we only predict 'O': " + str(percent)

# Generic function that takes in a single predictive indicator and trains/validates the modal
def train_validate_modal(data_set, indicator_name, indicator_dict_name):
    # training prediction
    count_correct = 0
    data_set_len = len(data_set)
    for i in range(data_set_len):
        try:
            # get the key corresponding to the max value in dict
            dict_use = indicator_dict_name[data_set.iloc[i][indicator_name]] # this might be buggy, is 'pos' correct?
            pred_tag = max(dict_use.iterkeys(), key=(lambda key: dict_use[key]))
        except:
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
            dict_use = indicator_dict_name[data_valid.iloc[i][indicator_name]] # this might be buggy, is 'pos' correct?
            pred_tag = max(dict_use.iterkeys(), key=(lambda key: dict_use[key]))
        except:
            do='nothing' # figure out this case later
        pred_train.append(pred_tag)
        if data_valid.iloc[i]['tag'] == pred_tag:
            count_correct += 1
    train_accuracy = 1.0*count_correct / data_valid_len
    print "Validation Accuracy using " + indicator_name + ': ' + str(train_accuracy)

# Make predictions based off of "shape"
train_validate_modal(data_small,'shape',shape_probs)

# # Make predictions based off of "words"
train_validate_modal(data_small,'word',word_probs)

# # Make predictions based off of "part-of-speech"
train_validate_modal(data_small,'pos',pos_probs)



#### SECTION V: Multiple Indicator Variables's Prediction Algorithm

# then we can incorporate multiple features in the algorithm by multiply probabilities and taking a max
# or adding log probabilities. This is more complicated, but will evertually be the basis of the final model. 

training prediction
count_correct = 0
for i in range(len(data_small)):
    if i % 100 == 0:
        print(i)
    tag_probs = {}
    for tag in tag_list:        
        pos_p = pos_probs[data_small.iloc[i]['pos']][tag]
        shape_p = shape_probs[data_small.iloc[i]['shape']][tag]
        word_p = word_probs[data_small.iloc[i]['word']][tag]
        tag_probs[tag] = pos_p * shape_p * word_p 
    pred_tag = max(tag_probs.iterkeys(), key=(lambda key: dict_use[key]))
    pred_train.append(pred_tag)
    if data_small.iloc[i]['tag'] == pred_tag:
        count_correct += 1
train_accuracy = 1.0*count_correct / len(data_small)
print "Train Accuracy using pos, shape, and word: " + str(train_accuracy)

