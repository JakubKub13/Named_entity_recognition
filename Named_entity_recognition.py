# !pip install transformers
#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_train.pkl
#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_test.pkl

from transformers import pipeline
import pickle # we need this to load our dataset

name_entity_recognition_model = pipeline("ner", aggregation_strategy='simple', device=0)

with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)