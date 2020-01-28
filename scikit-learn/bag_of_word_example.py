from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
group = fetch_20newsgroups()

count_vector = CountVectorizer(max_features=500, stop_words='english')


def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True


data_cleaned = []

for doc in group.data:
    docs_cleaned = ' '.join(word for word in doc.split()
                            if is_letter_only(word))
    data_cleaned.append(docs_cleaned)


data_count = count_vector.fit_transform(data_cleaned)
# print(data_count)
# print(data_count[0])
print(count_vector.get_feature_names())
