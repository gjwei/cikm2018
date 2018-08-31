#!/usr/bin/env python
#-*- coding:utf-8 -*- 
#Author: gjwei
import os
import ipdb

os.environ['CORENLP_HOME'] = "/home/gjw/workplace/corenlp/stanford-corenlp-full-2018-02-27/"

import corenlp

text = """?cómo puedo recibir un reembolso mediante tarjeta?"""

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.
with corenlp.CoreNLPClient(annotators="tokenize ssplit lemma ner".split()) as client:
  ann = client.annotate(text)
  print(ann.sentence[0])

# You can access annotations using ann.
sentence = ann.sentence[0]
# ipdb.set_trace()

result = [word.originalText for word in sentence.token]
lemmas = [word.lemma for word in sentence.token]
print(result)
print(lemmas)




# Use tokensregex patterns to find who wrote a sentence.
# sentences contains a list with matches for each sentence.
# length tells you whether or not there are any matches in this




