# cs182_ner
CS182 Final Project- building an NER Model using Hidden Markov Models (Josh Kuppersmith, Abdul Jamjoom)

* Refer to the Final Paper Apendix 1 for instructions on how to run the code
  
  Appendix I: Instructions for Running Code

Github link (Public Repo): https://github.com/ajamjoom/cs182_ner

We implemented a Naive Bayes (HMM) NER tagger, a Viterbi (HMM) NER tagger, and a Facebook Messenger app that acts as a UI for our NER taggers. All the code we wrote is located in the following files: ner_algo.py, hmm_viterbi_model.py, and app.py . 

We created simple command line methods to run different parts of our code so that you can easily test and examine our results. When you run a certain command, our code outputs analytical results for the chosen algorithm and features. Our fastest and most accurate algorithm's command line running method can be found in sections 7.2.

Important Note: You need to download our training data files to be able to run the code. These files are ner.csv (training data, this file is included on canvas), shape_100k.csv, pos_100k.csv, and word_100k.csv (all are pre-run trained dictionaries for our emission model, these three files are included on our github repository and on canvas). These last 3 files are very important so that you do not have to re-train, which takes a very long time. If you are not able to download ner.csv training data file from canvas please download it from the following link https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data and place it in the CS182_ner folder. If you weren't able to download the 100K files, you need to uncomment the following lines in ner_algo.py: 94, 97, and 100. These lines re-train the data and generate the csv files discussed above, which takes a few minutes to run. 


 HMM, Naive Bayes

File: ner_algo.py

There are multiple ways to test our naive model. You can run the model with different values of alpha, different types of feature predictors, or you can just run our best model. Below are the commands you can run on the ner_algo.py file. All our Naive Bayes models are trained on 100K data points. The default values for the alpha (hallucination) parameter for our features are:
word =  0.01, shape = 2.0 , pos = 2.0 (Best Alpha values we found). Note: This model is fairly slow as it runs on 100K data points and 20K points for validation (training predictions have been commented out for the sake of speed), if you run anything that includes the word feature it will take a minute or two to run the code.

This command will run our best Naive Bayes model (HMM and Viterbi is better - check 7.2) which is a combination of the word and part-of-speech (pos) features:
python ner_algo.py best

With the command below, you can test different combinations of features for our Naive Bayes algorithm:
python ner_algo.py test <feature>

* Replace <feature> with one of the following features: 
pos, shape, word, pos_word, pos_shape, word_shape, pos_word_shape

With the command below, you can test different alpha values for our pos_word_shape Naive Model: 

python ner_algo.py alpha_test <pos_alpha> <shape_alpha> <word_alpha>

* Replace the place holders above with a float. NOTE: this test works, but takes over an hour to run since it must re-train data

HMM, Viterbi

File: hmm_viterbi_model.py

This code runs the fastest and most accurate NER tagger we have implemented. It runs both the Viterbi Prev-Tag, Markov Model and the Viterbi  Prev-Prev-Tag Loosened Model at the same time so you can compare them. This command takes around 20-30 sec to run with 20,000 training points, please run this :)

python hmm_viterbi_model.py <training_points>

* Replace <training_points> with an int between 20,000 (fast, lower performance) and 200,000 (slow, better performance)
  



