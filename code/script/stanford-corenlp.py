#!/usr/bin/env python
#-*- coding:utf-8 -*- 
#Author: gjwei

import logging

# Simple usage
from stanfordcorenlp import StanfordCoreNLP



def get_pos(poss):
    result = []
    for pos in poss:
        result.append(pos[1])
    return ' '.join(result)



nlp = StanfordCoreNLP(r'/home/gjw/workplace/corenlp/stanford-corenlp-full-2018-02-27/', lang='es',
                      timeout=500000
                    )

sentence = 'Hola, me gustaría saber si un vendedor necesita que le envíe de vuelta'
print('Tokenize:', nlp.word_tokenize(sentence))
print('Part of Speech:',get_pos(nlp.pos_tag(sentence)))
# print('Named Entities:', nlp.ner(sentence))
print('Constituency Parsing:', nlp.parse(sentence))
print('Dependency Parsing:', nlp.dependency_parse(sentence))



# props={'annotators': 'tokenize,ssplit,lemma,ner','pipelineLanguage':'es','outputFormat':'json'}
# print(nlp.annotate(sentence, props))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))

nlp.close() # Do not forget to close! The backend server will consume a lot memery.