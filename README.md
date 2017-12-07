# cs182_ner
CS182 Final Project- building an NER Model using Hidden Markov Models (Josh Kuppersmith, Abdul Jamjoom)

-> Facebook Messenger API Integration
-> NER algorithm (Python)

How to run the code from your terminal:

File = hmm_viterbi_model.py
  
  1. Run HMM Viterbi Model (Fastest and most accurate tagger)
  
    python hmm_viterbi_model.py (#training points)

	i.e. Ôpython hmm_viterbi_model.py 20000Õ to run with 20000 data points
	use an int no larger than 600000
    
File = ner_algo.py

1. NER Baseline Accuracy:
    
  python ner_algo.py baseline

2. Run NER with a single indicator:

  python ner_algo.py single indicator
  
  Note: replace "indicator" with one of { pos, word, shape }

3. Run NER with a combination of indicators:
  
  python ner_algo.py combo indicators
  
  Note: replace "indicators" with one of the following { pos_word, pos_shape, word_shape, pos_word_shape }

4. Test Alpha Tuning (only for the combo pos_word_shape algorithm):
    
   python ner_algo.py alpha pos_alpha shape_alpha word_alpha
  
  Note: alpha is a float.
  
  
  



