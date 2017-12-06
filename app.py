import sys
import json
import csv, codecs, cStringIO
from datetime import datetime
import ast
import string
# NEW IS BELOW
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# NEW IS ABOVE
import os
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
from collections import Counter
# import f_dict

app = Flask(__name__)

# def csv_from_dict(csv_name, dict_name):
#     # Truncate content of file if there is anything there
#     f = open(csv_name, "w+")
#     f.close()
#     # Upload new content to the selected empty csv file
#     with open(csv_name, 'wb') as csv_file:
#         writer = csv.writer(csv_file)
#         for key, value in dict_name.items():
#             writer.writerow([key.encode('utf-8').strip(), value]) # This works for all dicts
#             #writer.writerow([key, value]) # This doesn't work for the words dictionary
# initial_tag_probs = {}
# transition_probs = {}
# f = {}

# csv_from_dict('initial_tag_probs.csv', initial_tag_probs)
# csv_from_dict('transition_probs.csv', transition_probs)
# csv_from_dict('f.csv', f)

# initial_tag_probs = {}
# transition_probs = {}
# f = {}

# def viterbi_prediction(sentence_data):        
#     valid_prediction = []
#     count = 0
#     for i in xrange(len(sentence_data)):
#         data = sentence_data[i]
#         word = data[0]
#         pos = data[1]
#         shape = data[2]
#         max_tag = 'O'
#         if word in list(f['O'][0].keys()):
#             if pos in list(f['O'][1].keys()):
#                 if shape in list(f['O'][2].keys()):
#                     max_prob = -1000000
#                     max_tag = 'O'
#                     for tag in tag_list:
#                         ####### NEED ALL THESE
#                         emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                
#                         # transition model
#                         # prev_tag = row['prev-iob']
#                         pre_tag = 'O' # if the first entry. Is this correct?
#                         if i > 0 :
#                             pre_tag = valid_prediction[i-1]
#                         transition_prob = transition_probs[tag][prev_tag] ###NEED THIS
                    
#                         prob = emission * transition_prob
                    
#                         if prob > max_prob:
#                             max_prob = prob
#                             max_tag = tag
#         else: 
#             max_tag = 'O'
#             max_prob = -1
#             max_tag = 'O'
#             for tag in tag_list:
#                 # p(e|x)
#                 emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                
#                 # transition model
#                 #prev_tag = row['prev-iob']
#                 pre_tag = 'O' # if the first entry. Is this correct?
#                 if i > 0:
#                     pre_tag = valid_prediction[i-1]
#                 transition_prob = transition_probs[tag][prev_tag]
                    
#                 prob = emission * transition_prob
            
#                 if prob > max_prob:
#                     max_prob = prob
#                     max_tag = tag
            
#         valid_prediction += '('+word +',' +max_tag +')'
#     return valid_prediction

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == os.environ["VERIFY_TOKEN"]:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    
    # send_quickrep_message(sender_id, messenger_ner(sentence_data))
    #send_quickrep_message(sender_id,viterbi_prediction(sentence_data))
    # return viterbi_prediction(sentence_data)
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
                            send_message(sender_id, "Shoot, sorry about that. Thank you for letting us know!")
                        
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
                            # Training data variables. All possible options of each indicator variable
                            # pos_list = [u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'RP', u'$', u'NN', u',', u'.', u'TO', u'PRP', u'RB', u';', u':', u'NNS', u'NNP', u'VB', u'WRB', u'RRB', u'CC', u'PDT', u'RBS', u'RBR', u'CD', u'NNPS', u'EX', u'IN', u'WP$', u'MD', u'LRB', u'JJS', u'JJR']      # Word's part of speech options
                            # shape_list = [u'mixedcase', u'lowercase', u'camelcase', u'uppercase', u'capitalized', u'number', u'abbreviation', u'punct', u'other', u'ending-dot', u'contains-hyphen']  # Shape of each word
                            # word_list = [u'limited', u'Autonomy', u'pardon', u'Frontier', u'destroyed', u'outlining', u'four', u'facilities', u'protest', u'Olympics', u'controversial', u'aides', u'Kyrgyz', u'Machimura', u'whose', u'feeding', u'violate', u'eligible', u'weapons-grade', u'poorest', u'Watch', u'activist', u'Western', u'under', u'teaching', u'Secretary', u'worth', u'importation', u'updated', u'risk', u'downstream', u'Iranian', u'regional', u'homes', u'every', u'reforms', u'Old', u'stabbed', u'Jianchao', u'bringing', u'Ehud', u'Abbas', u'school', u'Saturn', u'Economists', u'crops', u'companies', u'1657', u'May', u'pardons', u'shouting', u'enhance', u'Paul', u'shows', u'precursor', u'enjoy', u'Jing', u'force', u'leaders', u'1,600', u'awake', u'consistent', u'estimates', u'direct', u'budget', u'second', u'street', u'Poles', u'Torrijos', u'Collins', u'sailed', u'estimated', u'panda', u'Mugabe', u'even', u'established', u'state-approved', u'pace', u'solemn', u'decisions', u'Minuteman', u'launch', u'spokesman', u'lawmaker', u'establishes', u'toll', u'new', u'increasing', u'officially', u'told', u'Kadima', u'kicked', u'offenses', u'human', u'men', u'non-governmental', u'equals', u'hundreds', u'reported', u'Serbs', u'[', u'103', u'100', u'obtained', u'monarch', u'greenery', u'kids', u'daughter', u'elaborate', u'changed', u'reports', u'Hamas', u'electricity', u'Recalled', u'analysts', u'permit', u'military', u'poverty', u'changes', u'punishment', u'criticism', u'Group', u'secure', u'campaign', u'Carol', u'explained', u'Dohuk', u'Madagascar', u'replace', u'brought', u'Also', u'Health', u'roles', u'jackets', u'stern', u'Italy', u'total', u'indictment', u'unit', u'frustrated', u'plot', u'criminals', u'spoke', u'would', u'army', u'compensation', u'hospital', u'voters', u'June', u'arms', u'charities', u'insult', u'Houses', u'Brigades', u'Stoltenberg', u'calm', u'raids', u'recommend', u'strike', u'marchers', u'type', u'tell', u'Diabetes', u'inquiry', u'supporters', u'holy', u'Ministry', u'successful', u'hurt', u'warn', u'uncertain', u'Nordic', u'90', u'93', u'hold', u'95', u'mountains', u'must', u'passenger', u'1990', u'self-rule', u'overheating', u'1995', u'room', u'thanks', u'rights', u'work', u'fabricating', u'Muhammad', u'pro-rebel', u'Haiti', u'temperatures', u'Colombia', u'Madrid', u'guise', u'sets', u'my', u'initial', u'ABAC', u'Abe', u'give', u'cited', u'in', u'organized', u'Li', u'10,000', u'Warri', u'Abu', u'want', u'New~York', u'Lord', u'drive', u'keep', u'absolute', u'provincial', u'Royal-Dutch', u'end', u'recovery', u'provide', u'verify', u'travel', u'sitting', u'Pinochet', u'damage', u'how', u'Muslims', u'Jack', u'front', u'interview', u'widespread', u'resignation', u'charging', u'offshore', u'massive', u'A', u'rise', u'minority', u'haven', u'replacing', u'after', u'damaged', u'Castro', u'customs', u'wrong', u'lot', u'Tombstone', u'outbursts', u'president', u'20/20', u'law', u'Carlos', u'severity', u'stronghold', u'US', u'All', u'attempt', u'third', u'Ali', u'amid', u'pullout', u'headquarters', u'fictional', u're-started', u'maintain', u'logjams', u'things', u'uninsured', u'operate', u'order', u'freed', u'operations', u'Brazil', u'belong', u'office', u'enter', u'welcomed', u'over', u'vary', u'expects', u'hidden', u'suspend', u'Hilla', u'TRUE', u'bombing', u'His', u'Physician', u'personal', u'succeeds', u'Badawi', u',', u'Hussain', u'writing', u'better', u'Sadr', u'Zimbabwe', u'Judge', u'stalemate', u'weeks', u'Banda', u'Khost', u'easier', u'Pervez', u'then', u'them', u'gunning', u'diseases', u'combination', u'Manpower', u'prevented', u'Administration', u'6,26,000', u'safe', u'Thomas', u'break', u'band', u'Merck', u'Omar', u'they', u'1,11,000', u'schools', u'imprisoning', u'strategic', u'capsized', u'bank', u'Control', u'several', u'Liberal', u'Jiaxuan', u'regularly', u'Cairo', u'arrested', u'rocks', u'Four', u'Archaeological', u'each', u'fatalities', u'went', u'side', u'Millions', u'signing', u'financial', u'Tangerang', u'independent', u'series', u'crimes', u'unlawfully', u'vote', u'arrives', u'combining', u'substantially', u'coastlines', u'allay', u'17.09', u'network', u'driving', u'tried', u'government-funded', u'vessels', u'organizations', u'William', u'encourage', u'al-Qaida-linked', u'arrests', u'medicine', u'surprise', u'newly', u'disagreements', u'Saadoun', u'India', u'Cuban', u'Committees', u'barrier', u'city', u'Karbala', u'University', u'threatened', u'Levy', u'campaigns', u'USSR', u'Including', u'Iran', u'estimate', u'wanted', u'storm', u'immigrated', u'Clinton', u'Monday', u'Roche', u'created', u'September', u'starts', u'refugee', u'correctness', u'days', u'Horn', u'ministries', u'renew', u'levels', u'counter-terrorism', u'Human', u'Mirror', u'Mian', u'arrived', u'Niger', u'employee', u'Bosnia-Herzegovina', u'cleaning', u'inspectors', u'Ziauddin', u'already', u'state-run', u'Palestinian', u'Tensions', u'meet', u'Tobago', u'charter', u'outlying', u'Interviewed', u'channels', u'Judaidah', u'adopted', u'another', u'AIDS', u'thick', u'shrine', u'deaths', u'payload', u'171', u'abundantly', u'Centers', u'service', u'top', u'night', u'girls', u'H5N1', u'needed', u'38-year-old', u'airs', u'rates', u'too', u'Press', u'draft', u'inflate', u'Afghan', u'militancy', u'damaging', u'mutate', u'urban', u'Ahmed', u'murder', u'Salvatrucha', u'centered', u'rescue', u'villages', u'took', u'rejected', u'Qaida', u'Summit', u'western', u'Ciampino', u'kept', u'upsurge', u'Gene', u'Mireya', u'begins', u'Advisory', u'Adnan', u'FALSE', u'target', u'showed', u'defiance', u'negotiator', u'nations', u'project', u'matter', u'Somalians', u'classes', u'painful', u'Sea', u'bloodshed', u'powers', u'minus', u'Trinidad', u'palace', u'ran', u'upset', u'talking', u'atomic', u'seed', u'manner', u'seen', u'regulate', u'seek', u'tells', u'contents', u'relatively', u'forced', u'strength', u'1968', u'Egyptian', u'4,500', u'responsible', u'-', u'isolated', u'1962', u'Mahdi', u'contact', u'Minister', u'forces', u'academy', u'curfew', u'reversed', u'Bank', u'al-Zarqawi', u'committing', u'though', u'distraction', u'Obita', u'automaker', u'Gilani', u'extending', u'paved', u'regular', u'decisively', u'panic', u'Abramoff', u'explosives', u'morality', u'Sgorbati', u'Alston', u'oath', u'medical', u'camp', u'flow', u'accident', u'treaty', u'orderly', u'points', u'Thirteen', u'deeply', u'warming', u'drunkard', u'voting', u'responded', u'came', u'reserve', u'sub-contractor', u'saying', u'renewed', u'pardoned', u'bomb', u'reactor', u'pope', u'hunger', u'Union', u'Abdul', u'meetings', u'Agency', u'nominated', u'ending', u'attempts', u'privately-owned', u'radio', u'Qureia', u'stepping', u're-bidding', u'evening', u'bail', u'European', u'Muslim-majority', u'20-year', u'just', u'Yusuf', u'Republicans', u'Nairobi', u'touched', u'stabbing', u'pursuing', u'mend', u'announce', u'population', u'jailed', u'mascots', u'wearing', u'do', u'exports', u'Havana', u'reconstruction', u'loved', u'Nevertheless', u'commodity', u'pre-empt', u'conduct', u'stop', u'television', u'herd', u'coast', u'criticized', u'vaccinations', u'despite', u'report', u'Organization', u'Palestinians', u'Soviet', u'unfortunately', u'hall', u'runs', u'Wild', u'Leaders', u'countries', u'Washington', u'disapora', u'twice', u'bad', u'Duarte', u'favors', u'insurgents', u'particulate', u'ruins', u'secretary', u'respond', u'blew', u'Caribbean', u'disaster', u'scolded', u'recipients', u'Adviser', u'infectious', u'nun', u'decided', u'result', u'Sydney', u'corrupt', u'John', u'subject', u'outbreak', u'Indonesian', u'said', u'capacity', u'hijack', u'pressured', u'ambitions', u'away', u'Djibouti', u'targets', u'mud', u'future', u'cooperation', u'Ugandan', u'encounters', u'discovery', u'preserve', u'Wolfowitz', u'accord', u'never', u'terms', u'nationwide', u'confusion', u'weak', u'tripled', u'Zimbabwean', u'70th', u'clues', u'news', u'picking', u'debt', u'Bayelsa', u'improve', u'received', u'climate', u'Lima', u'cow', u'met', u'country', u'ill', u'uncertainty', u'against', u'tax', u'Kirkuk', u'32-year-old', u'planned', u'faces', u'Navy', u'requests', u'Syrian', u'asked', u'obscenity', u'assassination', u'1.5', u'Conservationists', u'character', u'Hadley', u'epidemic', u'Kurram', u'hygiene', u'arraigned', u'month-long', u'pregnancy', u'250', u'Likewise', u'257', u'speak', u'conference', u'Chavez', u'Hashem', u'jobless', u'engines', u'basis', u'condemned', u'37-year-old', u'Southern', u'three', u'been', u'.', u'Baathist', u'predetermined', u'much', u'interest', u'basic', u'expected', u'parents', u'entered', u'dozens', u'board', u'Acute', u'life', u'Georgia', u'families', u'prosperous', u'Vice', u'hospitalized', u'dismiss', u'worker', u'personally', u'prosecutors', u'worked', u'violations', u'diverted', u'east', u'Deputy', u'launcher', u'economies', u'air', u'near', u'aid', u'gave', u'flooding', u'Tuesday', u'launched', u'Oo', u'seven', u'played', u'equator', u'is', u'it', u'expenses', u'player', u'Bush', u'experts', u'Goatherd', u'Majlis-e-Amal', u'exile', u'march', u'mouse', u'if', u'Benedict', u'bottles', u'emergency', u'Bystanders', u'perform', u'demonstration', u'make', u'Stop', u'amount', u'ex-dictator', u'Volcker', u'potentially', u'ammunition', u'James', u'President', u'sidelines', u'unharmed', u'meets', u'satellite', u'raid', u'Institute', u'Jakarta', u'published', u'evil', u'Nursery', u'4,40,000', u'marched', u'Latvia', u'Shahdi', u'Home', u'Hushiar', u'Hamdi', u'postponed', u'IAEA', u'cross', u'1976', u'restive', u'failing', u'1973', u'ocean', u'Uganda', u'practices', u'detained', u'sandbags', u'Paraguay', u'claims', u'the', u'camps', u'investments', u'zones', u'left', u'spurred', u'After', u'background', u'quoted', u'sentence', u'photo', u'Lithuania', u'quotes', u'fighters', u'pedestrians', u'farther', u'victim', u'breaching', u'followed', u'Georgians', u'yet', u'previous', u'depicts', u'Russians', u'candidate', u'ease', u'Labado', u'had', u'Hundreds', u'Gao', u'Baghdad', u'squad', u'teenager', u'prison', u'News', u'Philip', u'has', u'reversal', u'humanity', u'photographed', u'33-member', u'sailors', u'elders', u'Juba', u'possible', u'GOATHERD', u'Morning', u'injuring', u'Chrysler', u'Carriles', u'judge', u'highly', u'hostage-takers', u'insurgent', u'59', u'congressman', u'contradict', u'spared', u'seaside', u'50', u'52', u'specific', u'offices', u'officer', u'54', u'archaeological', u'security', u'Species', u'towards', u'Idrac', u'hunt', u'old', u'Bangladesh', u'deal', u'people', u'foreigners', u'mosques', u'Tom', u'topping', u'dead', u'Zambian', u'Commission', u'Nippon', u'meters', u'election', u'winds', u'escape', u'extended', u'enemies', u'denies', u'southeastern', u'Life', u'Saltillo', u'for', u'Cuba', u'Huygens', u'comments', u'polluted', u'arguing', u'asking', u'select', u'denied', u'Rising', u'participation', u'He', u'core', u'knew', u'repository', u'payment', u'pose', u'Daily', u'Swedes', u'MS-13', u'jolted', u'Paktika', u'defensive', u'Project', u'surgeries', u'post', u'shaken', u'First', u'chapter', u'Dragan', u'Moqtada', u'attacks', u'dollars', u'months', u'Peru', u'harassing', u'Without', u'ensure', u'logistical', u'own', u'eight', u'efforts', u'Uji', u'primitive', u'WHO', u'statements', u'shopkeepers', u'presence', u'civil', u'Romano', u'prisoners', u'Abdullah', u'bound', u'well-suited', u'son', u'down', u'device', u'laborers', u'right', u'magazines', u'coastal', u'Geithner', u'1990s', u'Korean', u'reducing', u'rely', u'Inter-American', u'crowd', u'African-American', u'support', u'flying', u'legislation', u'why', u'conservation', u'way', u'extremism', u'resulted', u'Reserve', u'was', u'war', u'launching', u'asset', u'Rangoon', u'head', u'medium', u'form', u'offer', u'January', u'becoming', u'6', u'discourage', u'landing', u'failure', u'hemisphere', u'magnitude', u'Survey', u'elderly', u'227', u'blindness', u'patient', u'surrender', u'atrocities', u'delegation', u'administering', u'portions', u'happen', u'lashed', u'unthinkable', u'inside', u'islands', u'Estimates', u'demonstrations', u'until', u'retire', u'Tony', u'unusually', u'mountain', u'Brighton', u'dues', u'assistance', u'Jose', u'promises', u'medication', u'156', u'Nuclear', u'Obeidi', u'tournament', u'store', u'one-third', u'evidence', u'soothe', u'Maathai', u'younger', u'promised', u'Gaza', u'request', u'ship', u'negotiations', u'constructed', u'1,000', u'rough', u'protested', u'relay', u'no', u'when', u'Possible', u'handed', u'Andreu', u'negotiators', u'flood', u'setting', u'role', u'holding', u'digital', u'test', u'Many', u'3.9', u'shrink', u'Costa', u'cleric', u"'s", u'bakery', u'welcome', u'Melbourne', u'fell', u'Tibet', u'establish', u'Earth', u'rolling', u'shatters', u'demanding', u'died', u'billion', u'militias', u'regime', u'cheated', u'Dulaymi', u'Radical', u'proposed', u'Charles', u'landed', u'together', u'uncovered', u'loyal', u'time', u'push', u'Roger', u'banners', u'dust', u'Moscow', u'profits', u'adult', u'neighbors', u'managed', u'launchers', u'drugs', u'2,75,000', u'global', u'focus', u'deliveries', u'mild', u'Republic', u'/', u'convoy', u'Engineers', u'bomber', u'helicopter', u'Bahr', u'suicide', u'428', u'casualties', u'Dan', u'father', u'counter-attack', u'environment', u'charge', u'Mutahida', u'supplies', u'terror', u'suffered', u'96', u'Tourism', u'penetrating', u'1991', u'fatigues', u'scampered', u'Washington-based', u'administered', u'spate', u'Besigye', u'1992', u'mourn', u'tanks', u'advised', u'10-year', u'word', u'forgiving', u'blast', u'feeble', u'Jersey', u'governments', u'drag', u'vacancy', u'denounced', u'respiratory', u'Statements', u'level', u'did', u'die', u'dig', u'proposals', u'brother', u'standards', u'leave', u'Reuters', u'subway', u'Mahmoud', u'team', u'Tal', u'bars', u'round', u'Kevin', u'multi-layered', u'prevent', u'policemen', u'says', u'temples', u'dealing', u'sign', u'Stephen', u'slowed', u'cost', u'Lage', u'Rice', u'sordid', u'Red', u'Rica', u'enticing', u'Akmatbayev', u'representative', u'Georgian', u'contracted', u'patrols', u'supporter', u'current', u'suspect', u'goes', u'During', u'appeal', u'Latgalians', u'filled', u'supporting', u'processing', u'jury', u'Shaanxi', u'Libya', u'heavily', u'passage', u'Age', u'Costello', u'Controversy', u'water', u'renounce', u'groups', u'English', u'alone', u'along', u'Times', u'teacher', u'change', u'Warner', u'passengers', u'unidentified', u'Associated', u'institute', u'Meanwhile', u'guilty', u'trial', u'439', u'usually', u'Crimea', u'shook', u'12-month', u'Lankan', u'useful', u'extra', u'Justice', u'problematic', u'chaired', u'When', u'safeguard', u'Alberto', u'crisis', u'market', u'Australia', u'troops', u'working', u'prove', u'Vision', u'positive', u'permanent-status', u'visit', u'only', u'shortfalls', u'by', u'Tiger', u'territories', u'live', u'upheld', u'ca.', u'scope', u'Udi', u'Hardline', u'today', u'streamlined', u'riot', u'Evo', u'club', u'altar', u'apparent', u'1,200', u'levees', u'claiming', u'Uruzgan', u'archaeologists', u'cases', u'effort', u'fly', u'tribunal', u'migrants', u'Pope', u'Liu', u'German', u'car', u'Jens', u'cap', u'flu', u'Pacific', u'believes', u'sandstorm', u'bombmaker', u'harmonious', u'values', u'can', u'believed', u'making', u'anti-government', u'newspapers', u'loaning', u'heart', u'citizens', u'figure', u'Vietnamese', u'Federal', u'Media', u'greenest', u'1980s', u'heard', u'stroke', u'dropped', u'council', u'allowed', u'offense', u'counting', u'organizer', u'232', u'parliamentary', u'validated', u'winter', u'silos', u'inundated', u'divided', u'exceptional', u'ambush', u'species', u'1', u'canyons', u'vital', u'fourth', u'survivors', u'genocide', u'Why', u'economy', u'map', u'product', u'massacre', u'information', u'may', u'designated', u'fed', u'southern', u'crashed', u'membership', u'produce', u'designed', u'Martin', u'date', u'such', u'data', u'grow', u'man', u'relinquish', u'Etihad', u'stress', u'natural', u'sectors', u'16,000', u'5.8', u'ones', u'so', u'deposit', u'snowed', u'truce', u'Montiglio', u'talk', u'Islam', u'unnamed', u'Mississippi', u'serving', u'shield', u'insurgency', u'Hezbollah', u'midnight', u'years', u'stability', u'ended', u'tonic', u'argued', u'managers', u'cold', u'still', u'birds', u'Ghana', u'police', u'Elsewhere', u'1834', u'Nancy-Amelia', u'facility', u'Hispanics', u'window', u'suspicion', u'workforces', u'orchestrating', u'World', u'areas', u'decades', u'aliens', u'happened', u'Augusto', u'Sister', u'halt', u'Muhammed', u'civilian', u'grandson', u'investigated', u'nation', u'1954', u'Rome', u'She', u'half', u'not', u'now', u'discuss', u'administer', u'term', u'workload', u'name', u'Simple', u'III', u'Shinzo', u'47', u'Husin', u'rock', u'Agricultural', u'entirely', u'quarter', u'fired', u'challenged', u'torch', u'exercises', u'Geological', u'Barack', u'Enriched', u'entering', u'year', u'territory', u'girl', u'Benin', u'living', u'opened', u'Medusa', u'profit', u'tensions', u'activities', u'Hague-based', u'increase', u'Norwegian', u'12-year-old', u'investigation', u'emerged', u'uprising', u'churches', u'Khabarovsk', u'receiving', u'bordering', u'maneuver', u'Stone', u'earlier', u'Rights', u'semi-autonomous', u'drugmaker', u'million', u'incentives', u'orbits', u'Felipe', u'Bakri', u'rebel', u'plea', u'Transportation', u'Security', u'Foreign', u'inevitable', u'care', u'training', u'A.D.', u'language', u'ministry', u'flare', u'impose', u'pandemic', u'thing', u'Senegal', u'place', u'Topol', u'10-month-old', u'slaying', u'mortars', u'nonprofit', u'Ahmadinejad', u'warning', u'think', u'frequent', u'first', u'searching', u'gunshot', u'tennis', u'Guard', u'There', u'Asia', u'one', u'reopen', u'temporarily', u'Black', u'fast', u'suspended', u'invasion', u'comprises', u'Farid', u'message', u'fight', u'open', u'beachfront', u'millions', u'facilitation', u'little', u'district', u'Osman', u'caught', u'trillion', u'Congo', u'anyone', u'Lieutenant', u'Sandstorms', u'2', u'sweep', u'Abduction', u'friend', u'gives', u'nutrition', u'mining', u'Martyrs', u'thaw', u'wrongdoing', u'mostly', u'that', u'season', u'pelting', u'broadcast', u'released', u'al-Nuimei', u'inmates', u'eliminated', u'drowned', u'Rumsfeld', u'than', u'boyfriend', u'11', u'10', u'13', u'Wanted', u'15', u'14', u'17', u'16', u'19', u'rival', u'27-year', u'Titan', u'diplomatic', u'transferred', u'spokeswoman', u'sabotaged', u'accused', u'were', u'marines', u'counselor', u'Nicaragua', u'Yankee', u'and', u'searchers', u'illness', u'Court', u'remained', u'41st', u'topics', u'confessed', u'turned', u'locations', u'Japanese', u'say', u'Respiratory', u'presiding', u'Rachid', u'anger', u'recover', u'Venezuela', u'any', u'Saengprathum', u'conversion', u'inspections', u'al-Sadr', u'Zambia', u'De', u'aside', u'peacekeeping', u'endangered', u'potential', u'take', u'roadside', u'projects', u'registered', u'200', u'begin', u'sure', u'Another', u'normal', u'track', u'price', u'Cassini', u'indicted', u'paid', u'Separately', u'Kypriano', u'Ivanov', u'backslid', u'Anti-Japanese', u'occurred', u'stormed', u'remarks', u'Tommy', u']', u'Childhood', u'Markos', u'democratic', u'8,500', u'Ms.', u'considered', u'average', u'later', u'Humans', u'sale', u'al-Baghdadi', u'federal', u'slaves', u'Strip', u'senior', u'shop', u'Web', u'urge', u'shot', u'surplus', u'show', u'allegations', u'Research', u'discovered', u'Ethiopia', u'bedside', u'dictator', u'southwestern', u'high-profile', u'rounded', u'ground', u'Province', u'Mikhail', u'TamilNet', u'slow', u'Yury', u'Polish', u'Saddam', u'daily', u'oil-for-food', u'behind', u'crime', u'impunity', u'34-day', u'going', u'black', u'raising', u'developed', u'Services', u'sensitivities', u'congressional', u'dispute', u'intercontinental', u'predict', u'explosion', u'closed', u'get', u'hostage', u'freezing', u'Ahmad', u'mission', u'surging', u'nearly', u'determine', u'awaiting', u'Illinois', u'prime', u'reveal', u'Alliance', u'radar', u'floods', u'Uruguay', u'morning', u'Kony', u'embassy', u'bombs', u'liable', u'worries', u'finishing', u'marred', u'where', u'al-Uloum', u'arrival', u'declared', u'Forces', u'violated', u'committed', u'seat', u'relative', u'infrastructure', u'donate', u'Musharraf', u'Saeb', u'concern', u'suspects', u'detect', u'Iowa', u'That', u'ways', u'review', u'boy', u'sites', u'weapons', u'originated', u'enough', u'bureau', u'environmentally', u'Members', u'between', u'affecting', u'Thai', u'before', u'across', u'Halutz', u'Sri', u'jobs', u'25', u'spring', u'U.S.', u'preach', u'killing', u'blame', u'Burmese', u'Gaulle', u'March', u'spark', u'cities', u'come', u'originates', u'dates', u'missions', u'mob', u'Kani', u'internationally-recognized', u'many', u'region', u'according', u'contract', u'blasts', u'Malaysia', u'columnist', u'Srebrenica', u'residency', u'senator', u'traded', u'Goats', u'comes', u'nearby', u'among', u'Silva', u'150', u'effective', u'period', u'insist', u'auditors', u'60', u'Fox', u'confirm', u'one-stop', u'cancel', u'Leader', u'boat', u'considering', u'late', u'evacuate', u'capable', u'west', u'rebuild', u'airliner', u'airlines', u'But', u'pilgrims', u'NATO-led', u'collided', u'production', u'mutilating', u'Reporters', u'2010', u'abating', u'borders', u'500', u'engine', u'avian', u'transportation', u'recruiting', u'\xc9', u'thousand', u'formed', u'Inspector', u'wounds', u'readings', u'photos', u'minister', u'declaration', u'suspected', u'U.S.-brokered', u'former', u'those', u'pilot', u'case', u'developing', u'these', u'tribal', u'polls', u"n't", u'cast', u'Fujimori', u'ongoing', u'Spain', u'policies', u'residence', u'newspaper', u'situation', u'abducted', u'landslide', u'Open', u'taking', u'Salvadorans', u'U.S.-led', u'prevalence', u'Sudan', u'Azur', u'Tennis', u'800', u'commander', u'strain', u'prosecutor', u'Rahm', u'Somali', u'return', u'activity', u'technology', u'Admiral', u'Israel', u'missiles', u'develop', u'media', u'granted', u'seeking', u'regulations', u'same', u'surveillance', u'fixation', u'Horbach', u'speech', u'angered', u'Brotherhood', u'Through', u'struggling', u'document', u'Lanka', u'events', u'week', u'backgrounds', u'oil', u'centuries', u'Salvador', u'I', u'boosted', u'refusal', u'driver', u'II', u'persons', u'running', u'Bay', u'Seven', u'renewable', u'Qadeer', u'largely', u'charges', u'hydrocarbons', u'floor', u'It', u'al-Rubaei', u'amounts', u'severe', u'costs', u'relief', u'In', u'very', u'easing', u'seventh-century', u'canceled', u'Botswana', u'bodies', u'charged', u'Worshipers', u'United', u'gangs', u'being', u'money', u'Vatican', u'actions', u'Conservative', u'lavish', u'violent', u'kill', u'disperse', u'3,000', u'touch', u'captured', u'weekly', u'Mullen', u'death', u'grounded', u'irrationality', u'smugglers', u'confirmed', u'airfield', u'Margaret', u'Solidaria', u'bloc', u'treatment', u'Although', u'republic', u'Ariel', u'scheduled', u'hover', u'negotiating', u'around', u'Aung', u'ruler', u'Nobel-Prize', u'unnecessary', u'unveil', u'early', u'Tour', u'inflation', u'makes', u'Baltic', u'accepted', u'lady', u'U.N.', u'ruled', u'Yemen', u'Smaller', u'amputation', u'endorses', u'miracle', u'facing', u'tsunami', u'London', u'laughing', u'drainage', u'Gunmen', u'served', u'Operation', u'rich', u'stationary', u'ceremonial', u'Mara', u'Waziristan', u'Army', u'West', u'Punjab', u'spilled', u'Somalia', u'Baiji', u'Dr.', u'images', u'International', u'racial', u'Thaksin', u'accomplice', u'Vienna', u'Danish', u'slaughtering', u'Government', u'News-agency', u'teachings', u'Earlier', u'recorded', u'legal', u'moon', u'critical', u'bridges', u'provides', u'moderate', u'knife', u'severed', u'unsuccessfully', u'Chilean', u'policeman', u'Later', u'assembly', u'scientific', u'business', u'obliged', u'sixth', u'warlords', u'inspect', u'local', u'broken', u'Vietnam', u'70', u'contracts', u'UTC', u'manufacturer', u'on', u'stone', u'central', u'ace', u'island', u'industry', u'violence', u'discussed', u'meaning', u'Trade', u'informing', u'Kids', u'Mozambique', u'airline', u'act', u'Jeep', u'tariffs', u'referred', u'or', u'organizers', u'heavy', u'lands', u'protests', u'hypothermia', u'burning', u'No', u'two-day', u'communication', u'image', u'scores', u'Susilo', u'authorities', u'rising', u'homeless', u'parties', u'elementary', u'Pakistani', u'shrinking', u'intervention', u'her', u'area', u'bribes', u'removing', u'there', u'alleged', u'meals', u'start', u'anti-Japanese', u'sealed', u'Rose', u'punished', u'Bishkek', u'valley', u'patients', u'Obama', u'passports', u'tight', u'mastered', u'SARS', u'unsettled', u'smuggling', u'peaceful', u'delayed', u'strikes', u'cabinet', u'announcement', u'two-thirds', u'Bhumibol', u'trying', u'with', u'buying', u'outskirts', u'fraud', u'applying', u'Rawalpindi', u'House', u'Adil', u'Joseph', u'emissions', u'abuses', u'politically', u'agree', u'Special', u'misery', u'strongly', u'wants', u'naysayers', u'required', u'citizenship', u'Space', u'stealing', u'moved', u'al', u'Thursday', u'general', u'navy', u'as', u'hypertension', u'at', u'Nicanor', u'confidence-building', u'moves', u'operation', u'Center', u'4.5', u'again', u'4.6', u'pamphlets', u'piracy', u'personnel', u'Sudanese', u'insurance', u'Ken', u'field', u'2001', u'5', u'extrajudicial', u'Scott', u'you', u'really', u'remote-controlled', u'immunity', u'2007', u'poor', u'Kurdish', u'briefly', u'fuel-efficient', u'games', u'separate', u'infighting', u'Fernando', u'includes', u'gifts', u'Ghormley', u'Pearson', u'important', u'Serb', u'coverage', u'7.5', u'quarantined', u'captives', u'missed', u'building', u'guerrillas', u'Major', u'Mexican', u'bilateral', u'calls', u'invest', u'odds', u'Biak', u'Musab', u'kilogram', u'enrichment', u'bystander', u'directors', u'Street', u'mass', u'nutrient', u'overseas', u'starting', u'bus', u'Analysts', u'Canada', u'represent', u'all', u'influenza', u'consider', u'caused', u'hands', u'lack', u'dollar', u'month', u'unrest', u'spacecraft', u'debris', u'Swaziland', u'talks', u'follow', u'Moses', u'research', u'religious', u'children', u'strangers', u'Byzantine', u'hunting', u'pledged', u'opportunities', u'to', u'program', u'captivity', u'safety', u'lawmakers', u'paying', u'HMS', u'woman', u'appointment', u'returned', u'returning', u'far', u'detention', u'commuted', u'worst', u'fall', u'Panamanian-flagged', u'ticket', u'condition', u'--', u'ingratitude', u'Jabaliya', u'list', u'joined', u'large', u'Ismail', u'lobbyist', u'independence', u'Presidents', u'small', u'study', u'Severe', u'Topol-M', u'Ashibli', u'neighborhood', u'Shaowu', u'guaranteed', u'streets', u'nearing', u'past', u'Department', u'rate', u'design', u'perspective', u'further', u'functioning', u'investment', u'al-Dulaimi', u'what', u'Korea', u'80-year-old', u'resume', u'Bijeljina', u'Darfur', u'witnesses', u'version', u'Zenawi', u'scientists', u'learned', u'apply', u'public', u'movement', u'multinational', u'turmoil', u'full', u'Prosecutors', u'Mohammed', u'hours', u'killings', u'operating', u'Latvian', u'civilians', u'concluding', u'strong', u'Anbar', u'lucrative', u'Hamid', u'77-year-old', u"Shi'ite", u'naval', u'ahead', u'exist', u'allows', u'1845', u'Alvaro', u'soldier', u'airport', u'healthier', u'Congolese', u'social', u'runoff', u'murdering', u'1940', u'point', u'ex-president', u'family', u'dissidents', u'Top', u'pasture', u'ask', u'self-imposed', u'aimed', u'trained', u'Bedfordshire', u'gunmen', u'pleaded', u'Nigeria', u'cynics', u'armed', u'Ohio-based', u'one-seat', u'Europe', u'Hamas-led', u'ridding', u'Joint', u'petroleum', u'two', u'stepped', u'comparing', u'Daw', u'Rwanda', u'soil', u'Russian', u'Bambang', u'injury', u'markets', u'more', u'Karzai', u'Dick', u'door', u'formality', u'ejected', u'company', u'tested', u'American', u'So', u'basically', u'broke', u'known', u'tsunamis', u'Harcourt', u'producing', u'town', u'Sutham', u'Haniyeh', u'hour', u'offender', u'citizen', u'Vioxx', u'coalition', u'remain', u'pressing', u'nine', u'evolved', u'learn', u'abandon', u'strategies', u'National', u'history', u'Repairs', u'prompt', u'KindHearts', u'CAR', u'accept', u'states', u'pushed', u'Congress', u'protects', u'outrage', u'airplane', u'Venezuelan', u'deplored', u'Kenya', u'!', u'huge', u'needs', u'Parliament', u'court', u'goal', u'rather', u'acts', u'surpassed', u'Embassy', u'donors', u'plant', u'Migrants', u'plans', u'disgraced', u'1970s', u'different', u'questioning', u'Texas', u'retirement', u'Gilchrist', u'plane', u'Emirates', u'Swiss', u'backyard', u'Residents', u'Yekhanurov', u'undecided', u'a', u'telephoned', u'short', u'three-tenths', u'deluge', u'Mexico', u'register', u'overnight', u'deserted', u'coal', u'responsibility', u'Democrats', u'deterioration', u'pay', u'banks', u'Armed', u'playing', u'remnants', u'help', u'finalized', u'reform', u'crackdowns', u'soon', u'trade', u'held', u'scientist', u'through', u'DeLay', u'committee', u'signs', u'suffer', u'overflowed', u'its', u'Zhaoxing', u'Benjamin', u'member', u'26', u'27', u'20', u'21', u'Donald', u'23', u'environmentalist', u'29', u'actually', u'clashes', u'resort', u'absence', u'parts', u'systems', u'Cumple', u'Surgeon', u'Tamil', u'three-day', u'hurting', u'good', u'motivated', u'northwestern', u'food', u'propose', u'brutally', u'walls', u'outlawed', u'compound', u'communities', u'assailed', u'Obesity', u'Mogadishu', u'complain', u'harshest', u'easily', u'frigate', u'cave-ins', u'pregnant', u'coping', u'fended', u'Ram', u'Resistance', u'Authority', u'tear', u'stopped', u'exiled', u'Council', u'12th', u'found', u'leveled', u'characterized', u'investigators', u'barrels', u'breakdown', u'harm', u'Afghans', u'pirates', u'Zone', u'Harare', u'generation', u'house', u'energy', u'hard', u'Militant', u'case-by-case', u'expect', u'Morales', u'disembark', u'carry', u'Federer', u'Criminal', u'event', u'More', u'worshipers', u'deals', u'dominates', u'funding', u'alcohol', u'clandestine', u'since', u'prosecution', u'Tamiflu', u'adhere', u'acting', u'health', u'Israeli-Palestinian', u'7', u'anti-viral', u'issue', u'plotting', u'fled', u'Ukraine', u'risen', u'Afar', u'judge-and-jury', u'sparked', u'hosted', u'According', u'Inacio', u'Papua', u'houses', u'reason', u'base', u'members', u'Mali', u'put', u'Ninh', u'moans', u'beginning', u'curtailed', u'El', u'jirga', u'benefits', u'Neskovic', u'People', u'Haas', u'persuading', u'George', u'Iyad', u'threat', u'Guardian', u'Are', u'pilots', u'fallen', u'lawsuits', u'feed', u'ex-Clinton', u'major', u'Cholily', u'Syndrome', u'Island', u'distasteful', u'KIND-HEARTED', u'radical', u'netted', u'Hugo', u'number', u'Narrates', u'elder', u'13,000', u'Indian', u'fees', u'Taleban', u'Silvio', u'wages', u'ballistic', u'regret', u'checkpoint', u'"', u'heads', u'Tasnim', u'leading', u'threatening', u'least', u'militant', u'belongs', u'commissioning', u'station', u'vowed', u'erupted', u'Delta', u'fought', u'hundred', u'Sunni', u'voted', u'recovered', u'listed', u'audit', u'selling', u'pipelines', u'passed', u'trapped', u'exploded', u'Lashkar-e-Jhangvi', u'part', u'Karadzic', u'Ocean', u'Malik', u'1910', u'treasures', u'interviewer', u'four-day', u'carnage', u'center', u'multi-billion', u'gotten', u'5,00,000', u'imply', u'aims', u'youth', u'Peruvian', u'Officials', u'Atomic', u'toward', u'ousted', u'kilometers', u'Faso', u'sentenced', u'deciding', u'modeled', u'minds', u'strengthen', u'Beattie', u'suspicious', u'Father', u'Kerry', u'airstrike', u'Sutalinov', u'sentences', u'Swazis', u'option', u'sell', u'Sichuan', u'built', u'Islamist-controlled', u'Ukrainian', u'Food', u'jeopardize', u'Virginia', u'port', u'Arab', u'extracted', u'internal', u'Germany', u'build', u'al-Qaim', u'Addis', u'Arak', u'With', u'shops', u'Germans', u'province', u'Kyrgyzstan', u'added', u'Violators', u'cargo', u'reckon', u'quote', u'eggs', u'english', u'reach', u'chart', u'most', u'virus', u'73', u'plan', u'significant', u'Guyana', u'services', u'accepting', u'The', u'extremely', u'approved', u'soared', u'constitution', u'wreckage', u'Honduras', u'devastated', u'Egypt', u'Anne-Marie', u'clear', u'humanitarian', u'cover', u'shortly', u'inspector', u'charitable', u'Mwanawasa', u'reestablished', u'flattened', u'If', u'dragged', u'joining', u'sector', u'celebrated', u'rebels', u'gold', u'gates', u'commissions', u'doctors', u'Islamist', u'converge', u'Majority', u'indicates', u'businesses', u'fine', u'find', u'Fatah', u'Poor', u'cell', u'giant', u'believe', u'Redmond', u'deserts', u'northern', u'justice', u'ruin', u'Muslim', u'excessively', u'19th', u'French', u'Kizza', u'failed', u'flights', u'surveyed', u'pretty', u'8', u'Su-24', u'Prime', u'Bedford', u'his', u'hit', u'Thirty', u'mercy', u'dependent', u'compiled', u'banned', u'flowing', u'financing', u'revered', u'rest', u'Both', u'reporters', u'during', u'him', u'Arizona', u'enemy', u'Olympic', u'resolve', u'warmup', u'progressive', u'sacrificed', u'doubled', u'Former', u'investigate', u'stored', u'Kandahar', u'cartoons', u'withdrawal', u'gathered', u'river', u'Endangered', u'Chairman', u'bulky', u'disappearance', u'set', u'opposed', u'achieved', u'carrying', u'activists', u'startup', u'France', u'tip', u'see', u'defense', u'are', u'sea', u'mediators', u'2.3', u'close', u'Argentina', u'declined', u'feared', u'Ramda', u'threats', u'expert', u'Critics', u'Regular', u'Blair', u'Park', u'rejecting', u'determined', u'conveyed', u'Egyptians', u'Tehran', u'probably', u'burned', u'afflicted', u'conditions', u'corruption', u'risks', u'available', u'King', u'recently', u'missing', u'targeted', u'attention', u'confiscated', u'poorer', u'aircraft', u'ranked', u'turning', u'opposition', u'African', u'Irbil', u'Refugees', u'abruptly', u'both', u'Kooyong', u'last', u'senators', u'Iraqis', u'restaurant', u'secrets', u'treated', u'annual', u'foreign', u'sensitive', u'connection', u'became', u'Mosul', u'rose', u'long-term', u'anti-war', u'mean', u'finds', u'Wam', u'demonizing', u'supply', u'reasons', u'Panama', u'address', u'allegiance', u'community', u'cried', u'village', u'vessel', u'slots', u'War', u'belt', u'lease', u'finally', u'raise', u'incomes', u'create', u'Ibrahim', u'political', u'due', u'earthquake', u'convicted', u'Most', u'secret', u'Downing', u'Baquba', u'embezzlement', u'airstrikes', u'Ritchie', u'assassinated', u'meeting', u'4', u'firm', u'dialogue', u'flight', u'champion', u'sending', u'kilograms', u'Polls', u'gas', u'remark', u'16-year-old', u'mine', u'Condoleezza', u'fund', u'six-party', u'lives', u'Saudi', u'demand', u'Raza', u'incoming', u'towns', u'presented', u'consulate', u'handling', u'politician', u'metro', u'Classic', u'frozen', u'bill', u'elections', u'governor', u'earthquakes', u'healthy', u'while', u'replaced', u'consular', u'behavior', u'withdrawing', u'fleet', u'Nablus', u'advising', u'City', u'shoddy', u'Khan', u'hoping', u'employees', u'century', u'backing', u'Mulongoti', u'Cheney', u'Tim', u'itself', u'Sisco', u'Instead', u'ready', u'center-left', u'leaving', u'all-important', u'Ramallah', u'fate', u'U.S.-based', u'anti-terrorist', u'stimulated', u'East', u'FAO', u'widely', u'grand', u'9', u'survey', u'dozen', u'conflict', u'higher', u'development', u'Faith', u'used', u'co-star', u'affairs', u'comprehensive', u'defeat', u'alive', u'yesterday', u'Pakistanis', u'steps', u'helped', u'flown', u'moving', u'user', u'contractors', u'Christian', u'recent', u'recognized', u'lower', u'task', u'flared', u'Latin', u'flock', u'Soh', u'chickens', u'spent', u'without', u'person', u'Economic', u'flags', u'withdraw', u'Talks', u'organization', u'Jerusalem', u'Quality', u'stem', u'year-on-year', u'observed', u'Wangari', u'Labor', u'Erekat', u'pardoning', u'shape', u'openly', u'plagued', u'Reconstruction', u'world', u'alternative', u'Kenyan', u'injuries', u'ban', u'letting', u'cut', u'$', u'majority', u'workers', u'deputy', u'education', u'source', u'causing', u'deliberately', u'exhibition', u'win', u'ballots', u'impoverished', u'Bekasi', u'noting', u'chairman', u'bin', u'On', u'victims', u'They', u'finalize', u'demands', u'prefer', u'bid', u'Air', u'suffering', u'clemency', u'Aslam', u'Aid', u'bit', u'sustainable', u'royalties', u'formal', u'Pakistan', u'Alexandria', u'imposed', u'shaped', u'defended', u'follows', u'commanders', u'associate', u'collect', u'continue', u'indication', u'popular', u'tribes', u'privately', u'Summers', u'Tokyo', u'Java', u'often', u'Gulf', u'al-Aqsa', u'Qiang', u'some', u'back', u'extremist', u'trends', u'economic', u'nothing', u'grenades', u'hard-line', u'ourselves', u'Algerian', u'delivered', u'oil-rich', u'1917', u'fears', u'affects', u'decision', u'per', u'commented', u'religion', u'Ecuador', u'donations', u'journalist', u'be', u'measures', u'run', u'Bill', u'agreement', u'venues', u'refused', u'step', u'Command', u'Reports', u'become', u'refuses', u'lottery', u'Bob', u'Russia', u'August', u'faith', u'wildlife', u'cautious', u'mingled', u'Ron', u'Voters', u'range', u'involving', u'Yemeni', u'Committee', u'Bosnian', u'offshoot', u'anti-Denmark', u'block', u'pollution', u'prices', u'real', u'enduring', u'terrorists', u'into', u'within', u'usual', u'signature', u'inaction', u'Terrorist', u'evasion', u'Bosco', u'military-led', u'statistics', u'Salazar', u'exposed', u'U.S', u'long', u'crackdown', u'infected', u'grave', u'1,31,000', u'examination', u':', u'Guantanamo', u'record', u'Professionals', u'himself', u'elsewhere', u'pounded', u'Rhymes', u'great', u'collapsed', u'Malaysian', u'immigration', u'hoped', u'boys', u'underlined', u'properly', u'repeatedly', u'filed', u'hopes', u'Mohanna', u'deployed', u'Beijing', u'Office', u'restarted', u'Africa', u'up', u'us', u'planet', u'mobile', u'G-8', u'Those', u'planes', u'reviewed', u'warmer', u'Ohio', u'called', u'disappeared', u'ordered', u'adults', u'stripped', u'Ney', u'influence', u'notarized', u'Aviv', u'coordinator', u'single', u'October', u'curb', u'home', u'rally', u'denying', u'kidney', u'journalists', u'independents', u'Commissioner', u'%', u'Zoba', u'Fidel', u'TV', u'peace', u'uranium', u'Shwe', u'thinks', u'preventing', u'politicians', u'Israelis', u'transport', u'Student', u'Athens', u'pandas', u'Agha', u'68-34', u'draw', u'users', u'reportedly', u'provided', u're-election', u'problems', u'hideouts', u'helping', u'AP', u'militants', u'AU', u'prepares', u'depicting', u'missile', u'sides', u'ago', u'land', u'de-radicalization', u'lead', u'vice', u'officers', u'vehicles', u'orbit', u'walked', u'An', u'2002', u'2003', u'2000', u'summit', u'2006', u'At', u'2004', u'2005', u'Uribe', u'international', u'2008', u'2009', u'requires', u'having', u'once', u're-opened', u'unemployment', u'issued', u'mines', u'highways', u'Sumatra', u'Nobutaka', u'gang', u'go', u'contributions', u'centers', u'issues', u'Dodge', u'Vahidi', u'gases', u'graves', u'Business', u'concerned', u'mosque', u'young', u'attackers', u'citing', u'squeezing', u'stable', u'Not', u'include', u'friendly', u'sent', u'drunk', u'seized', u'mainstream', u'outside', u'continues', u'Constitutional', u'miners', u'wave', u'questioned', u'chooses', u'accuse', u'Talabani', u'continued', u'arrange', u'Charlotte', u'positions', u'explosions', u'eve', u'try', u'archipelago', u'Defense', u'verdict', u'?', u'Murat', u'fortified', u'obesity', u'refer', u'ceasefire', u'enforce', u'warheads', u'sacrificial', u'gripping', u'pledge', u'Lebanon', u'participate', u'procession', u'crop', u'Councilor', u'fold', u'State', u'noise', u'video', u'traveling', u'picked', u'growing', u'cat', u'Norway', u'brokered', u'reconciliation', u'Neolithic', u'plays', u'power', u'giving', u'saw', u'slight', u'expressed', u'danger', u'access', u'Now', u'waiting', u'printing', u'capital', u'incidents', u'firms', u'America', u'bird', u'exercise', u'body', u'proliferation', u'Faridullah', u'voluntarily', u'exchange', u'Freedom', u'transmitting', u'commercial', u'jointly', u'delivering', u'following', u'slogans', u'northeast', u'Green', u'expansion', u'others', u'Tarmiyah', u'consideration', u'Hussein', u'kidnapped', u'al-Dulaymi', u'39', u'Chinese', u'Burkina', u'technical', u'32', u'Energy', u'30', u'37', u'35', u'34', u'leadership', u'resulting', u'oversees', u'submerged', u'Arabia', u'residents', u'Religious', u'maker', u'Staff', u'named', u'debts', u'socializing', u'claim', u'presidents', u'Robert', u'manage', u'private', u'names', u'Brazilian', u'Police', u'sugar', u'use', u'fee', u'from', u'pawn', u'remains', u'illegal', u'management', u'ministers', u'next', u'few', u'circumstances', u'NATO', u'vehicle', u'timing', u'Suspected', u'midday', u'Saboor', u'parliament', u'formally', u'Bergner', u'upcoming', u'started', u'becomes', u'freedom', u'Wednesday', u'cooperate', u'industrial', u'train', u'paroled', u'Aigle', u'sharply', u'demobilization', u'appointee', u'Berlusconi', u'VOA', u'escaped', u'account', u'Bittok', u'animals', u'Estonia', u'sanctions', u'this', u'critic', u'Haitian', u'recession', u'reserves', u'of', u'English-language', u'about', u'1,300', u'withdrew', u'Nicole', u'proof', u'control', u'Israeli', u'patrol', u'Northern', u'painkiller', u'process', u'Non-Proliferation', u'risk-sharing', u'trumped', u'purposes', u'pieces', u'high', u'EU', u'Mr.', u'Maldives', u'pulling', u'something', u'sought', u'Party', u'Australian', u'Several', u'Christmas', u'rape', u'D.C.', u'breakup', u'surrounded', u'democracy', u'six', u'occur', u'regions', u'located', u'suburb', u'seeds', u'Sunnis', u'Authorities', u'grudgingly', u'Yu', u'secular', u'normalize', u'buildings', u'ABC', u'attend', u'Posada', u'Joel', u'samples', u'Disease', u'British', u'3', u'Images', u'Roman', u'abuse', u'demonstrating', u'ethnic', u'ties', u'da', u'status', u'realized', u'bombers', u'HIV', u'counter', u'lines', u'Community', u'One', u'Middle', u'chief', u'allow', u'Greece', u'Wael', u'subsequently', u'volunteers', u'preparing', u'cease-fire', u'Catholic', u'Regina', u'holds', u'coordinated', u'produces', u'move', u'Bahamas', u'produced', u'Ethiopian-born', u'alliance', u'including', u'looks', u'quake', u'Interior', u'fertile', u'Beirut', u'operative', u'industries', u'helicopters', u'Shell', u'treat', u'Emergency', u'chosen', u'Sint', u'ships', u'winning', u'Luiz', u'permanent', u'degrees', u'choose', u'stalled', u'Luis', u'2017', u'holiday', u'warrants', u'extradition', u'2012', u'Task', u'crash', u'greater', u'Miami', u'auto', u'questions', u'material', u'hate', u'seawater', u'Adam', u'Social', u'satellites', u'assets', u'flee', u'day', u'presidential', u'Summer', u'Katz', u'San', u'Sam', u'Iraqi', u'Panamanian', u'warned', u'eased', u'Emmanuel', u'identified', u'finance', u'Channu', u'Celsius', u'rockets', u'Calderon', u'shortage', u'crossing', u'accounts', u'eastern', u'Network', u'doing', u'apologize', u'700', u'Representatives', u'related', u'Witnesses', u'society', u'Kilju', u'measure', u'separating', u'As', u'Kabul', u'87', u'special', u'out', u'secretly', u"'", u'entertainment', u'arriving', u"Shi'ites", u'chaos', u'Zalinge', u'Tang', u'dominated', u'China', u'cause', u'annexed', u'announced', u'shut', u'Shortly', u'Dianne', u'Ashura', u'Indonesia', u'release', u'completely', u'undocumented', u'advisor', u'Senator', u'20,000', u'2,00,000', u'resumed', u'Aceh', u'allies', u'prepare', u'Lula', u'mortar', u'could', u'shells', u'obstacle', u'times', u'counterpart', u'intoxicated', u'Hyde', u'downed', u'V-6', u'scare', u'south', u'unspent', u'al-Qaida', u'jurists', u'blown', u'possessing', u'60-year', u'powerful', u'scene', u'owned', u'Bilfinger', u'improving', u'Orakzai', u'improvements', u'owner', u'reached', u'Local', u'Chiefs', u'ancient', u'North', u'publication', u'practitioners', u'bears', u'flag', u'system', u'relations', u'their', u'attack', u'Katutsi', u'scrapped', u'ankle', u'Iraq', u'final', u'Association', u'low', u'pervasive', u'Binh', u'punish', u'completed', u'This', u'lists', u'environmental', u'Mercosur', u'Democratic', u'state-monitored', u'Berger', u'bodyguard', u'relaxing', u'liquid', u'Meles', u'riches', u'reacted', u'involved', u'probe', u'appealing', u'Military', u'briefing', u'visited', u'landmine', u'providing', u'bet', u'collecting', u'Information', u'1.6', u'1.3', u'Isfahan', u'1.8', u'Other', u'defending', u'have', u';', u'need', u'Shinawatra', u'border', u'linked', u'Balad', u'Development', u'documents', u'swam', u'acquitted', u'Abdullahi', u'agency', u'mid', u'parks', u'Xinhua', u'concerns', u'which', u'funds', u'jail', u'techniques', u'Chakul', u'Sharon', u'worldwide', u'regard', u'led', u'six-year', u'clash', u'Sevastopol', u'wind-blown', u'who', u'harmful', u'eventide', u'visa', u'Games', u'preparations', u'preliminary', u'emancipation', u'wounding', u'class', u'Azahari', u'Burma', u'so-called', u'deny', u'Yoweri', u'won', u'neighboring', u'Some', u'disease', u'large-scale', u'face', u'High', u'proceedings', u'alHurra', u'error', u'wounded', u'guard', u'typical', u'Saturday', u'Provincial', u'Santos', u'currently', u'Paris', u'agreed', u'supported', u'3-Jun', u'Daley', u'bring', u'planning', u'greenhouse', u'soldiers', u'fear', u'longer', u'economist', u'debate', u'Independence', u'staff', u'answer', u'Ruz', u'based', u'earned', u'(', u'cache', u'controls', u'should', u'bombings', u'candidates', u'York', u'fuel', u'8,000', u'communist', u'thousands', u'secretaries', u'hope', u'meant', u'Hassan', u'plaintiff', u'Yudhoyono', u'Tanzania', u'unbeatable', u'overall', u'bear', u'likely', u'120', u'joint', u'Hague', u'checkpoints', u'drawn', u'Popular', u'Tel', u'River', u'procedures', u'Italian', u'Organizers', u'symbols', u'520', u'unauthorized', u'course', u'New', u'approach', u'quarantine', u'overflowing', u'calling', u'Svalbard', u'extremists', u'130', u'Experts', u'she', u'contain', u'murders', u'rein', u'fixed', u'widow', u'spotted', u'Afghanistan', u'Because', u'Dharmeratnam', u'consulates', u'prayers', u'mistreated', u'national', u'1994', u'we', u'asylum-seeking', u'Lawrence', u'officials', u'desperate', u'converting', u'operated', u'Central', u'incurable', u'White', u'closer', u'Good', u'televised', u'wealth', u'Issac', u'attempting', u'nuclear', u'Museveni', u'cruel', u'worship', u'state', u'Representative', u'action', u'progress', u'July', u'Republican', u'sabotage', u'mainly', u'12-member', u'tore', u'surge', u'opening', u'rescuers', u'gunships', u'Kashmir', u'agencies', u'laid-off', u'interfere', u'efficiency', u'job', u'MSWATI', u'Allawi', u'key', u'group', u'defied', u'distribution', u'restrictions', u'invited', u'Ethiopian', u'incident', u'drug', u'Osama', u'Jihad', u'urging', u'Leonella', u'safely', u'otherwise', u'comment', u'Timothy', u'unclear', u'diplomats', u'ambushed', u'committee-chairman', u'Guinea', u'ca', u'570', u'disarmament', u'caverns', u'ambushes', u'batch', u'carbon', u'allocated', u'labor', u'unearthed', u'Gilgit', u'provinces', u'Boston', u'Pentastar', u'addition', u'decent', u'6-Apr', u'thanked', u'Bolivian', u'growth', u'agreements', u'Colorado', u'Aziz', u'Thailand', u'proposal', u'waste', u'diversity', u'turban', u'foes', u'Ababa', u'Islamabad', u'sufficient', u'protect', u'Nations', u'bulk', u'policy', u'finished', u'life-bearing', u'Pardons', u'ice', u'suspension', u'gross', u'an', u'gagged', u'present', u'And', u'Meantime', u'criminal', u'Force', u'abandoned', u'CDC', u'intimidate', u'plain', u'Radovan', u'Number', u'Euphrates', u'General', u'will', u'Clinton-era', u'wild', u'Si', u'almost', u'Shanghai', u'site', u'surface', u'Moscoso', u'arrangements', u'demanded', u'turns', u'claimed', u'You', u'Khushab', u'Mazar-e-Sharif', u'ransom', u'terrorist', u'productivity', u'Garden', u'Nigerian', u')', u'began', u'administration', u'extradited', u'Bangkok', u'compassionately', u'Halliburton', u'northwest', u'2,012', u'largest', u'party', u'terrorism', u'Sperling', u'difficult', u'injured', u'tighten', u'kidnapping', u'cubic', u'Prophet', u'Medical', u'upon', u'effect', u'beast', u'Airport', u'floodwaters', u'vault', u'Manuel', u'frequently', u'Victor', u'expand', u'diabetes', u'poultry', u'26-year', u'kidnappings', u'off', u'raping', u'Adulyadej', u'Aden', u'Senate', u'obstacles', u'well', u'fighting', u'Itar-Tass', u'States', u'idiot', u'testify', u'Colombo', u'deadly', u'position', u'Contreras', u'audio', u'arising', u'draconian', u'latest', u'restore', u'rocket', u'less', u'percent', u'proximity', u'wealthy', u'domestic', u'4.8', u'Yass', u'sources', u'Sunday', u'interim', u'fishing', u'seats', u'Pro-Russian', u'low-level', u'taken', u'warship', u'rapid', u'7,000', u'Islamic', u'Far', u'Company', u'Friday', u'lake', u'capabilities', u'killed', u'Brigadier', u'arrest', u'add', u'APEC', u'Taliban', u'Despite', u'challenges', u'Families', u'prosecuting', u'match', u'confront', u'tests', u'northeastern', u'increased', u'Kampala', u'government', u'Helmand', u'1,00,000', u'protesters', u'Sawyer', u'historic', u'five', u'know', u'deployment', u'press', u'immediately', u'prominent', u'loss', u'displaced', u'like', u'lost', u'Ibero-American', u'admitted', u'Hagino', u'war-torn', u'downplaying', u'leaves', u'clinging', u'works', u'refugees', u'Saud', u'replacement', u'because', u'habitat', u'polar', u'doled', u'Saakashvili', u'authority', u'motive', u'Bombings', u'export', u'Cooperation', u'Helicopter', u'119', u'Borders', u'Speaking', u'Mike', u'lamppost', u'custody', u'broad', u'avoid', u'detainees', u'miner', u'However', u'goats', u'demonstrators', u'does', u'mutation', u'transmitted', u'Jalal', u'leader', u'demonstrated', u'kidnappers', u'blowing', u'728', u'Kind-hearted', u'reaching', u'Prodi', u'Kelantan', u'pressure', u'host', u'disastrous', u'although', u'instability', u'targeting', u'insults', u'loans', u'trials', u'refuge', u'stage', u'LRA', u'rationing', u'rare', u'carried', u'getting', u'December', u'certainty', u'sectarian', u'urged', u'nightclub', u'revamp', u'statement', u'Americans', u'Descriptions', u'supposed', u'Japan', u'emperor', u'pistols', u'489', u'previously', u'Juan', u'Bolivia', u'Two', u'Khartoum', u'cocoa', u'chronic', u'anniversary', u'weather', u'Scientists', u'roads', u'quickly', u'Thousands', u'significantly', u'postal', u'regarded', u'ridge', u'additional', u'Bali', u'rivals', u'blanketed', u'reduction', u'awarded', u'Amy', u'spread', u'madrassas', u'describes', u'biggest', u'California', u'November', u'rumors', u'function', u'fiscal', u'north', u'triggered', u'Science', u'Sivaram', u'Coalition', u'but', u'Madden', u'protectionist', u'construction', u'courts', u'larger', u'Following', u'remote', u'Tynychbek', u'highest', u'25-year-old', u'he', u'count', u'also', u'quarterly', u'made', u'Mustafa', u'Corps', u'places', u'whether', u'dangerous', u'official', u'signed', u'placed', u'Britain', u'stories', u'ruling', u'pumped', u'U.S.-Mexico', u'problem', u'plutonium', u'CIA', u'attained', u'begun', u'links', u'Laden', u'recognize', u'Treasury', u'rubble', u'generating', u'Olmert', u'resumption', u'sank', u'suicide-bomber', u'Last', u'shooting', u'Colombian', u'worse', u'diplomat', u"'ll", u'30-day', u'dressed', u'attacked', u'offensive', u'43', u'40', u'Younis', u'Delivery', u'fire', u'other', u'Five', u'details', u'cashed', u'scandal', u'stimulant', u'illegally', u'star', u'Chan', u'colonized', u'Khayam', u'Spanish', u'outlets', u'stay', u'April', u'chance', u'Act', u'Insurgents', u'Congressman', u'friends', u'South', u'exposure', u'using', u'Hemingway', u'space', u'33', u'rule', u'portion', u'shouted', u'compete', u'Pyongyang', u'understand', u'parole', u'8th'] # All the words in the small data set [Change to large later on]

                            tag_list = [u'I-art', u'B-gpe', u'B-art', u'I-per', u'I-tim', u'B-org', u'O', u'B-geo', u'B-tim', u'I-geo', u'B-per', u'I-eve', u'B-eve', u'I-gpe', u'I-org', u'I-nat', u'B-nat'] 

                            # pos_probs = {}
                            # shape_probs = {}
                            # word_probs = {}

                            initial_tag_probs = {}
                            transition_probs = {}
                            f = f_dict.f

                            def dict_from_csv(csv_name):
                                print "on csv_file_name: ", csv_name
                                with open(csv_name, 'rb') as csv_file:
                                    init_dict = dict(csv.reader(csv_file))
                                    new_dict = {}
                                    for key, value in init_dict.iteritems():
                                        new_dict[unicode(key, 'utf-8')] = ast.literal_eval(value)
                                return new_dict

                            # Populate the pos, word, and shape dicts from their CSV files
                            initial_tag_probs = dict_from_csv('initial_tag_probs.csv')
                            transition_probs = dict_from_csv('transition_probs.csv')
                            # f = dict_from_csv('f.csv')

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

                            def features_from_tag(test_data):
                                response_1 = data_small['word']
                                response_2 = data_small['pos']
                                response_3 = data_small['shape']
                                
                                predictor = data_small['tag']
                                pred_final = pd.get_dummies(predictor)
                                
                                classify1 = RandomForestClassifier()
                                classify2 = RandomForestClassifier()
                                classify3 = RandomForestClassifier()

                                
                                classify1.fit(pred_final, response_1)
                                classify2.fit(pred_final, response_2)
                                classify3.fit(pred_final, response_3)
                                
                                # get likelihoods for each tag
                                
                                # GET ONE PREDICTION FOR EAST POSSIBLE TAG
                                target = pd.get_dummies(pd.DataFrame(tag_list))
                                
                                p1 = classify1.predict_proba(target)
                                p2 = classify2.predict_proba(target)
                                p3 = classify3.predict_proba(target)
                                
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
                                

                            f, final_word_list, final_pos_list, final_shape_list = features_from_tag(data_valid)

                            def viterbi_prediction(sentence_data):        
                                valid_prediction = ''
                                valid_pred_list = []
                                count = 0
                                for i in xrange(len(sentence_data)):
                                    data = sentence_data[i]
                                    word = data[0]
                                    pos = data[1]
                                    shape = data[2]
                                    max_tag = 'O'
                                    if word in list(f['O'][0].keys()):
                                        if pos in list(f['O'][1].keys()):
                                            if shape in list(f['O'][2].keys()):
                                                max_prob = -1000000
                                                max_tag = 'O'
                                                for tag in tag_list:
                                                    ####### NEED ALL THESE
                                                    emission = 1.0*f[tag][0][word]*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                                            
                                                    # transition model
                                                    # prev_tag = row['prev-iob']
                                                    prev_tag = 'O' # if the first entry. Is this correct?
                                                    if i > 0 :
                                                        prev_tag = valid_pred_list[i-1]
                                                    print "trans prob: ", transition_probs[tag]
                                                    transition_prob = transition_probs[tag][prev_tag] ###NEED THIS
                                                
                                                    prob = emission * transition_prob
                                                
                                                    if prob > max_prob:
                                                        max_prob = prob
                                                        max_tag = tag
                                    else: 
                                        max_tag = 'O'
                                        max_prob = -1
                                        max_tag = 'O'
                                        for tag in tag_list:
                                            # p(e|x)
                                            emission = 1.0*f[tag][1][pos]*f[tag][2][shape] * initial_tag_probs[tag]
                                            
                                            # transition model
                                            #prev_tag = row['prev-iob']
                                            prev_tag = 'O' # if the first entry. Is this correct?
                                            if i > 0:
                                                prev_tag = valid_pred_list[i-1]
                                            transition_prob = transition_probs[tag][prev_tag]
                                                
                                            prob = emission * transition_prob
                                        
                                            if prob > max_prob:
                                                max_prob = prob
                                                max_tag = tag
                                        
                                    valid_pred_list.append(max_tag)
                                    valid_prediction += '('+word +',' +max_tag +')'
                                return valid_prediction
                            # def messenger_ner(sentence_data):

                            #     # print sentence_dict
                            #     tagged_sentence = ''
                            #     for token_data in sentence_data:
                                    
                            #         word, word_pos, word_shape = token_data[0], token_data[1], token_data[2]
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
                            #         tagged_sentence += '('+word+','+max_tag+')'

                            #     return tagged_sentence

                            words = nltk.word_tokenize(message_text)
                            
                            pos_tags_tuple = nltk.pos_tag(words)

                            pos_tags = []
                            for elmt in pos_tags_tuple:
                                word, tag = elmt
                                pos_tags.append(tag)
                            
                            shapes = []
                            for token in words:
                                token_shape = shape(token)
                                shapes.append(token_shape)

                            sentence_data = []
                            for i in xrange(len(words)):
                                sentence_data.append([words[i], pos_tags[i], shapes[i]])

                            # send_quickrep_message(sender_id, messenger_ner(sentence_data))
                            send_quickrep_message(sender_id,viterbi_prediction(sentence_data))
                            

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
