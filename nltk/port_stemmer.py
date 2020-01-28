from nltk.stem import PorterStemmer

port_stemmer = PorterStemmer()

print(port_stemmer.stem('machines'))

print(port_stemmer.stem('learning'))
