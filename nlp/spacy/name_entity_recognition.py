import spacy

nlp = spacy.load('en_core_web_sm')
sent = 'The book written by Hayden Liu in 2018 sold at $30 in America'
tokens = nlp(sent)
print([(token.text, token.label_) for token in tokens.ents])

# Please refer to https:/ / spacy. io/ api/ annotation#sectionnamed-
# entities for a full list of named entity tags.
