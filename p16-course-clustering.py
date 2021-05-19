import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering


def course_to_level(num: int) -> int:
    if num < 200:
        return 1
    elif num < 300:
        return 2
    elif num < 400:
        return 3
    elif num < 500:
        return 4
    elif num < 600:
        return 5
    elif num >= 1000:
        return 1
    else:
        return 5


df = pd.read_json("data/midd_cs_courses.jsonl", lines=True)

print(df[["number", "title"]])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.description).toarray()
numbers = df.number.array.to_numpy()
levels = [course_to_level(n) for n in df.number.to_list()]

## IDEAS: edges
# Create edges between courses in the same cluster.
# Or are pre-requisites. (number mentioned in text?)
#    'plot([x1,x2], [y1,y2])' a line...
edges = set()
for i, text in enumerate(df.description):
    for j, course_num in enumerate(numbers):
        if "CSCI 0{}".format(course_num) in text or "CSCI {}".format(course_num) in text:
            edges.add((i, j))


## TODO: improve this visualization of CS courses.
#

## IDEAS: compare PCA to TSNE
# PCA doesn't have a perplexity parameter.
# What does TSNE do better on this dataset?

perplexity = 15
viz = TSNE(perplexity=perplexity, random_state=42)
#viz = PCA()
cluster = KMeans(n_clusters=5)
cluster_ids = cluster.fit_predict(X)
#print(cluster_ids)

V = viz.fit_transform(X)

# Right now, let's assign colors to our class-nodes based on their number.
color_values = levels  # TODO swap this.

plt.title("T-SNE(Courses), perplexity={}".format(perplexity))
plt.scatter(V[:, 0], V[:, 1], alpha=1, s=13, c=color_values, cmap="turbo")

# Annotate the scattered points with their course number.
for i in range(len(numbers)):
    course_num = str(numbers[i])
    x = V[i, 0]
    y = V[i, 1]
    plt.annotate(course_num, (x, y))


# Create a kmeans clustering of the courses.
# Then apply colors based on the kmeans-prediction to the below t-sne graph.

# for (i, cluster_i) in enumerate(cluster_ids):
#     for (j, cluster_j) in enumerate(cluster_ids):
#         if cluster_i != cluster_j:
#             continue
#         if cluster_i < cluster_j:
#             continue
#         # draw edges
#         plt.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], alpha=0.2, c="gray")

for (i, j) in edges:
        # draw edges
    plt.plot([V[i,0], V[j,0]], [V[i,1], V[j,1]], alpha=0.2, c="gray")

plt.savefig("graphs/p16-tsne-courses-p{}.png".format(perplexity))
plt.show()
