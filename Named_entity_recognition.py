# !pip install transformers
#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_train.pkl
#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_test.pkl

from transformers import pipeline
import pickle # we need this to load our dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer

name_entity_recognition_model = pipeline("ner", aggregation_strategy='simple', device=0)

with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)

print(corpus_test)

inputs = []
targets = []

for sentence_tag_pairs in corpus_test:
  tokens = []
  target = []
  for token, tag in sentence_tag_pairs:
    tokens.append(token)
    target.append(tag)
  inputs.append(tokens)
  targets.append(target)

print(inputs[9])
detokenizer = TreebankWordDetokenizer()

print(detokenizer.detokenize(inputs[9]))

print(targets[9])

name_entity_recognition_model(detokenizer.detokenize(inputs[9]))

def compute_prediction(tokens, input_, ner_result):
  # map huggin face ner result to list of tags for later performance assesment
  # tokens is the original tokenized sentence
  # input_ is the detokenized string

  predicted_tags = []
  state = 'O' # keeps track of state, so if O --> B, if B --> I, if I --> I
  current_index = 0
  for token in tokens:
    # find the token in the input_ (should be at or near the start)
    index = input_.find(token)
    assert(index >= 0)
    current_index += index # where we are currently pointing to

    # print(token, current_index) # debug

    #check if this index belongs to an entity and assign label
    tag = 'O'
    for entity in ner_result:
      if current_index >= entity['start'] and current_index < entity['end']:
        # then this token belongs to an entity
        if state == 'O':
          state = 'B'
        else:
          state = 'I'
        tag = f"{state}-{entity['entity_group']}"
        break
    if tag == 'O':
      # reset the state
      state = 'O'
    predicted_tags.append(tag)

    # remove the token from input_
    input_ = input_[index + len(token):]

    # update current index
    current_index += len(token)

  # sanity check
  print("len(predicted_tags)", len(predicted_tags))
  print("len(tokens", len(tokens))
  assert(len(predicted_tags) == len(tokens))
  return predicted_tags