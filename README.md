# cs182_ner
CS182 Final Project- building an NER Model using Hidden Markov Models (Josh Kuppersmith, Abdul Jamjoom)

-> Facebook Messenger API Integration
-> NER algorithm (Python)

How to run the code from your terminal:

File : ner_algo.py

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
  


