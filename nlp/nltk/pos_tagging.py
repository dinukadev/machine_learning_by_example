from nltk import pos_tag, word_tokenize
import nltk

sent = '''I am reading a book.
          It is Python Machine Learning By Example,
          2nd Edition.'''

print(pos_tag(word_tokenize(sent)))

nltk.help.upenn_tagset('PRP')
