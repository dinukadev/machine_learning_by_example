from sklearn.datasets import fetch_20newsgroups
import numpy as np

group = fetch_20newsgroups()
# this downloads the news groups data sets

print(group.keys())

print(group['target_names'])

print(np.unique(group.target))

print(group.data[0])

print(group.target_names[group.target[0]])
