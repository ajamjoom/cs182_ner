import sys
import json
import csv, codecs, cStringIO
from datetime import datetime
import ast
import string

import os
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
from collections import Counter

app = Flask(__name__)

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "CS182 Final Project | NER with HMM and fb messenger user output | by Abdulrahman Jamjoom and Josh Kupersmith", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events
    data = request.get_json()
    # log(data)

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                log("messaging_event: ")
                log(messaging_event)

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text
                    quick_reply_bool = messaging_event["message"].get("quick_reply")

                    if quick_reply_bool:
                        quick_payload = messaging_event["message"]["quick_reply"]["payload"]
                        
                        if quick_payload == "no": # ask for correct parsing
                            send_message(sender_id, "Shoot, sorry about that. Would be great if you send us the correct tagged tokens in the format we used. Preface your text with the phrase 'NER: '. Ex. NER: (Ahmed, name)...")
                        
                        elif quick_payload == "yes": # Correct parsing -> add it to training data
                            send_message(sender_id, "Perfect, thank you for letting us know!")

                        elif quick_payload == "not sure": # Don't add NER to training data
                            send_message(sender_id, "Alright thanks!")

                    else:
                        msg_start = message_text.split(':')[0]
                        
                        if msg_start == 'NER': # parse user text and add it to training data 
                            # reply to user with error if the text is not in the correct format
                            send_message(sender_id, "Thank you for improving our algorithm!")     
                        else:
                            pos_probs = {}
                            shape_probs = {}
                            word_probs = {}

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

                            # Doesn't account for type = abbreviation
                            def shape(word):
                                if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
                                    return u'number'
                                elif '.' == word:
                                    return u'ending-dot'
                                elif re.match('\W+$', word, re.UNICODE):
                                    return u'punct'
                                elif re.match('\w+$', word, re.UNICODE):
                                    if '-' in word:
                                        return u'contains-hyphen'
                                    elif word.isupper():
                                        return u'uppercase'
                                    elif word.islower():
                                        return u'lowercase'
                                    elif word[0].isupper():
                                        return u'capitalized'
                                    elif word[0].islower():
                                        return u'camelcase'
                                    else:
                                        return u'mixedcase'
                                else:
                                    return u'other'

                            def messenger_ner(sentence_lst):
   
                                # print sentence_dict
                                tagged_sentence = ''
                                for token_data in sentence_dict:
                                    
                                    word, word_pos, word_shape = token_data[0], token_data[1], token_data[2]
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
                                    tagged_sentence += '('+word+','+max_tag+')'

                                return tagged_sentence

                            words = nltk.word_tokenize("Dive into NLTK: Part-of-speech tagging POS Tagger")
                            
                            pos_tags_tuple = nltk.pos_tag(words)

                            pos_tags = []
                            for elmt in pos_tags_tuple:
                                word, tag = elmt
                                pos_tags.append(tag)
                            
                            shapes = []
                            for token in tokens:
                                token_shape = shape(token)
                                shapes.append(token_shape)

                            sentence_data = []
                            for i in xrange(len(words)):
                                sentence_dict.append([words[i], pos[i], shapes[i]])
                                
                            log("TOKENS")
                            log(tokens)
                            log("POS_TAGS")
                            log(pos_tags)
                            log("TOKEN_SHAPES")
                            log(shapes)
                            log("messenger_ner(sentence_data)")
                            log(messenger_ner(sentence_data))
                            
                            send_quickrep_message(sender_id, messenger_ner(sentence_data))

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                    pass

    return "ok", 200


def send_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": os.environ["PAGE_ACCESS_TOKEN"]
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text,
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)

def send_quickrep_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {
        "access_token": os.environ["PAGE_ACCESS_TOKEN"]
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text + '. Did we get the entity tags right?',
            "quick_replies":[
              {
                "content_type":"text",
                "title":"YES!",
                "payload":"yes"
              },
              {
                "content_type":"text",
                "title":"No :(",
                "payload":"no"
              },
              {
                "content_type":"text",
                "title":"Not sure",
                "payload":"not sure"
              }
            ]
        }
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(msg, *args, **kwargs):  # simple wrapper for logging to stdout on heroku
    try:
        if type(msg) is dict:
            msg = json.dumps(msg)
        else:
            msg = unicode(msg).format(*args, **kwargs)
        print u"{}: {}".format(datetime.now(), msg)
    except UnicodeEncodeError:
        pass  # squash logging errors in case of non-ascii text
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)
