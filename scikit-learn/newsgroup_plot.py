import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

group = fetch_20newsgroups()

sns.distplot(group.target)

plt.show()

