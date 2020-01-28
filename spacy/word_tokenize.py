import spacy

nlp = spacy.load('en_core_web_sm')
sent = '''I have been to U.K and U.S.A.'''
tokens = nlp(sent)

print([token.text for token in tokens])

