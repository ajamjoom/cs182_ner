import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


def run_new_model(training_points, testing_points):
    # LOAD IN AND HANDLE DATA 
    data = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
    data.head()

    # drop null rows and check if any null values remaining
    data.dropna(inplace=True)
    data[data.isnull().any(axis=1)].size

    # set sizes for training and testing data
    data_small = data[:training_points]
    data_valid = data[training_points:training_points+testing_points]

    # split into x and y
    preds = list(data.columns.values)
    preds.remove('tag')
    y_small = data_small['tag']
    x_small = data_small[preds]

    # prepare lists
    pos_list = list(set(data['pos']))
    shape_list = list(set(data['shape']))
    word_list = list(set(data['word']))
    tag_list = list(set(data['tag'].values))


    #############################
    ## SECTION 1               ## 
    ## PRELIM VITERBI TESTING  ##
    #############################

    # INITIAL STATE PROBABILITIES
    initial_tag_probs = {}
    for tag in tag_list:
        prob = 1.0*len(data_small[data_small['tag'] == tag]) / len(data_small)
        initial_tag_probs[tag] = prob

    # BASELINE ALL 'O' CLASSIFIER
    pred_b = []
    for i in range(len(data_small)):
        pred_b.append('O')
        
    print
    print("BASELINE ALL 'O' CLASSIFIER")
    print("Accuracy Score: " + str(accuracy_score(data_small['tag'], pred_b)))
    print("F1 Score: " + str(f1_score(data_small['tag'], pred_b, labels=tag_list, average="weighted")))
    print


    # TRANSITION PROBABILITIES, from tag to tag
    transition_probs = {}
    for tag1 in tag_list:
        within_tag = {}
        data_tag1 = data_small[data_small['tag'] == tag1]
        for tag2 in list(set(data_small['prev-iob'])):
            to_tag2 = data_tag1[data_tag1['prev-iob'] == tag2]
            within_tag[tag2] = len(to_tag2)*1.0/len(data_tag1)
        transition_probs[tag1] = within_tag


    # FUNCTION TO USE TESTED RANDOM FORESTS TO PREDICT FEATURE PROBABILITIES BASED ON TAG
    def features_from_tag(test_data):
        response_1 = data_small['word']
        response_2 = data_small['pos']
        response_3 = data_small['shape']
        
        predictor = data_small['tag']
        pred_final = pd.get_dummies(predictor)
        
        classify1 = RandomForestClassifier()
        classify2 = RandomForestClassifier()
        classify3 = RandomForestClassifier()

        # train RF models for 3 features
        classify1.fit(pred_final, response_1)
        classify2.fit(pred_final, response_2)
        classify3.fit(pred_final, response_3)
        
        # GET ONE PREDICTION FOR EAST POSSIBLE TAG
        target = pd.get_dummies(pd.DataFrame(tag_list))
        
        p1 = classify1.predict_proba(target)
        p2 = classify2.predict_proba(target)
        p3 = classify3.predict_proba(target)
        
        # prepare dictionaries in the format that we need
        emission_probs = {}
        for i in range(len(tag_list)):
            word_preds = {}
            words = classify1.classes_
            for j in range(len(words)):
                p = p1[i][j]
                word_preds[words[j]] = p
                
            pos_preds = {}
            poss = classify2.classes_
            for k in range(len(poss)):
                pos_preds[poss[k]] = p2[i][k]
            
            shape_preds = {}
            shapes = classify3.classes_
            for l in range(len(shapes)):
                shape_preds[shapes[l]] = p3[i][l]
            
            emission_probs[list(set(data_small['tag']))[i]] = [word_preds, pos_preds, shape_preds]
            
        # important to return order of classes in order to map later
        return emission_probs, classify1.classes_, classify2.classes_, classify3.classes_
        
    # run the function above
    f, final_word_list, final_pos_list, final_shape_list = features_from_tag(data_valid)


    # FUNCTION THAT USES VITERBI ALGORITHM TO PREDICT 
    def viterbi_prediction():
        train_prediction = []
        sentence_indices = list(set(data_small['sentence_idx']))
        
        count = 0
        for index, row in data_small.iterrows():
            word = row['word']
            pos = row['pos']
            shape = row['shape']
            max_tag = 'O'

            # make sure we have seen the features before
            if word in list(f['O'][0].keys()):
                if pos in list(f['O'][1].keys()):
                    if shape in list(f['O'][2].keys()):
                        max_prob = -1000000
                        max_tag = 'O'
                        for tag in tag_list:
                            # p(e|x)
                            emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                            # transition model
                            prev_tag = row['prev-iob']
                            transition_prob = transition_probs[tag][prev_tag]
                            prob = emission * transition_prob
                        
                            # maximize probability
                            if prob > max_prob:
                                max_prob = prob
                                max_tag = tag
                                if max_tag == 'camelcase':
                                    count += 1
            else: 
                # if not, maximize again, just with pos and shape
                max_tag = 'O'
                max_prob = -1
                max_tag = 'O'
                for tag in tag_list:
                    # p(e|x)
                    emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                    # transition model
                    prev_tag = row['prev-iob']
                    transition_prob = transition_probs[tag][prev_tag]
                        
                    prob = emission * transition_prob
                    
                    # maximize probability
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag
                
            train_prediction.append(max_tag)
        
        print("Training Accuracy: " + str(accuracy_score(train_prediction, data_small['tag'])))
        print("Training F1 Score: " + str(f1_score(data_small['tag'], train_prediction, labels=tag_list, average="weighted")))
        valid_prediction = []
        for index, row in data_valid.iterrows():
            word = row['word']
            pos = row['pos']
            shape = row['shape']
            max_tag = 'O'

            # check if we've seen the word before
            if word in list(f['O'][0].keys()):
                if pos in list(f['O'][1].keys()):
                    if shape in list(f['O'][2].keys()):
                        max_prob = -1000000
                        max_tag = 'O'
                        for tag in tag_list:
                            # p(e|x)
                            emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                            # transition model
                            prev_tag = row['prev-iob']
                            transition_prob = transition_probs[tag][prev_tag]
                        
                            prob = emission * transition_prob
                        
                            # maximize probability
                            if prob > max_prob:
                                max_prob = prob
                                max_tag = tag

            # predict using only pos and shape
            else: 
                
                max_tag = 'O'
                max_prob = -1
                max_tag = 'O'
                for tag in tag_list:
                    # p(e|x)
                    try: 
                        emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    except: 
                        count +=1
                        try:
                            emission = 1.0*f[tag][2][shape] * initial_tag_probs[tag]
                        except:
                            try:
                                emission = 1.0*f[tag][1][pos] * initial_tag_probs[tag]
                            except:
                                emission = 1.0*initial_tag_probs
                    # transition model
                    prev_tag = row['prev-iob']
                    transition_prob = transition_probs[tag][prev_tag]
                        
                    prob = emission * transition_prob
                    
                    # maximize 
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag


                
            valid_prediction.append(max_tag) 
        print("Validation Accuracy: " + str(accuracy_score(valid_prediction, data_valid['tag'])))
        print("Validation F1 Score: " + str(f1_score(data_valid['tag'], valid_prediction, labels=tag_list, average="weighted")))

    print("VITERBI ALGORITHM TESTING")
    print(str(training_points) + " Data Points, With Prev-IOB, Markov Model")
    viterbi_prediction()



    ###############################
    ## SECTION 2                 ## 
    ## LOOSENED VITERBI TESTING  ##
    ###############################


    # # TRANSITION PROBABILITIES from prev prev tag to tag
    # THIS IS WHERE WE LOOSEN THE MODEL, by considering transition from Prev-Prev-Tag

    transition_trans_probs = {}
    for tag1 in tag_list:
        within_tag = {}
        data_tag1 = data_small[data_small['tag'] == tag1]
        for tag2 in list(set(data['prev-prev-iob'])):
            to_tag2 = data_tag1[data_tag1['prev-prev-iob'] == tag2]
            within_tag[tag2] = len(to_tag2)*1.0/len(data_tag1)
        transition_trans_probs[tag1] = within_tag
        
    predictor = data_small['tag']
    pred_final = pd.get_dummies(predictor)
    target = pd.get_dummies(pd.DataFrame(tag_list))

    def loosened_viterbi_prediction():
        train_prediction = []
        sentence_indices = list(set(data_small['sentence_idx']))
        
        count = 0
        #print(f['B-gpe'][2])
        #print(f['B-geo'][2])
        #print(f['B-per'][2])
        for index, row in data_small.iterrows():
            word = row['word']
            pos = row['pos']
            shape = row['shape']
            max_tag = 'O'
            #print(list(f['O'][1].keys()))
            if word in list(f['O'][0].keys()):
                if pos in list(f['O'][1].keys()):
                    if shape in list(f['O'][2].keys()):
                        max_prob = -1000000
                        max_tag = 'O'
                        for tag in tag_list:
                            # p(e|x)
                            emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                            # transition model
                            prev_tag = row['prev-iob']
                            prev_prev_tag = row['prev-prev-iob']
                            transition_prob1 = transition_probs[tag][prev_tag]
                            transition_prob2 = transition_trans_probs[tag][prev_prev_tag]
                            prob = emission * transition_prob1 * transition_prob2
                        
                            if prob > max_prob:
                                max_prob = prob
                                max_tag = tag

            else: 
                max_tag = 'O'
                max_prob = -1
                for tag in tag_list:
                    # p(e|x)
                    emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                    # transition model
                    prev_tag = row['prev-iob']
                    prev_prev_tag = row['prev-prev-iob']
                    transition_prob1 = transition_probs[tag][prev_tag]
                    transition_prob2 = transition_trans_probs[tag][prev_prev_tag]
                    
                    prob = emission * transition_prob1 * transition_prob2
                
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag
                
            train_prediction.append(max_tag)
            
        print("Training Accuracy: " + str(accuracy_score(train_prediction, data_small['tag'])))
        print("Training F1 Score: " + str(f1_score(data_small['tag'], train_prediction, labels=tag_list, average="weighted")))
        valid_prediction = []
        count = 0
        for index, row in data_valid.iterrows():
            word = row['word']
            pos = row['pos']
            shape = row['shape']
            max_tag = 'O'
            if word in list(f['O'][0].keys()):
                if pos in list(f['O'][1].keys()):
                    if shape in list(f['O'][2].keys()):
                        max_prob = -1
                        max_tag = 'O'
                        for tag in tag_list:
                            # p(e|x)
                            emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    
                            # transition model
                            prev_tag = row['prev-iob']
                            prev_prev_tag = row['prev-prev-iob']
                            transition_prob1 = transition_probs[tag][prev_tag]
                            transition_prob2 = transition_trans_probs[tag][prev_prev_tag]
                        
                            prob = emission * transition_prob1 * transition_prob2

                            if prob > max_prob:
                                max_prob = prob
                                max_tag = tag
            else: 
                max_tag = 'O'
                max_prob = -1
                max_tag = 'O'
                for tag in tag_list:
                    # p(e|x)
                    try: 
                        emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                    except: 
                        try:
                            emission = 1.0*f[tag][2][shape] * initial_tag_probs[tag]
                        except:
                            try:
                                emission = 1.0*f[tag][1][pos] * initial_tag_probs[tag]
                            except:
                                emission = 1.0*initial_tag_probs
                    
                    # transition model
                    prev_tag = row['prev-iob']
                    prev_prev_tag = row['prev-prev-iob']
                    transition_prob1 = transition_probs[tag][prev_tag]
                    transition_prob2 = transition_trans_probs[tag][prev_prev_tag]

                    prob = emission * transition_prob1 * transition_prob2

                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag

            valid_prediction.append(max_tag)
            
        print("Validation Accuracy: " + str(accuracy_score(valid_prediction, data_valid['tag'])))
        print("Validation F1 Score: " + str(f1_score(data_valid['tag'], valid_prediction, labels=tag_list, average="weighted")))

        
    print
    print(str(training_points) + " Data Points, With Prev-Prev-IOB, Loosened Model")
    loosened_viterbi_prediction()

run_new_model(20000,10000)

