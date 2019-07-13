from nltk.stem import WordNetLemmatizer

word_lemmatizer = WordNetLemmatizer()

print(word_lemmatizer.lemmatize('machines'))

print(word_lemmatizer.lemmatize('learning'))

# Word net lemmatizer only lemmatizes on nouns by default
