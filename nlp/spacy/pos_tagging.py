import spacy

nlp = spacy.load('en_core_web_sm')
sent = '''I have been to U.K and U.S.A.'''
tokens = nlp(sent)

print([(token.text, token.pos_) for token in tokens])

# Please refer to https:/ / spacy. io/ api/ annotation#sectionnamed-
# entities for a full list of named entity tags.
