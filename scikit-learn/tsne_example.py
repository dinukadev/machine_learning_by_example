import matplotlib.pyplot as plt
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

categories = ['talk.religion.misc', 'comp.graphics', 'sci.space']
group = fetch_20newsgroups(categories=categories)
all_names = set(names.words())
count_vector = CountVectorizer(stop_words='english', max_features=500)
lemmatizer = WordNetLemmatizer()


def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True


data_cleaned = []
for doc in group.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split()
                           if is_letter_only(word) and word not in all_names)
    data_cleaned.append(doc_cleaned)

tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)

data_cleaned_count = count_vector.fit_transform(data_cleaned)

data_tsne = tsne_model.fit_transform(data_cleaned_count.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=group.target)
plt.show()
