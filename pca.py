import numpy as np
import csv
from sklearn.decomposition import PCA

file1 = "data/nse_input.csv"
file2 = "data/nse_target.csv"
raw_data = open(file1, "rt")
raw_data2 = open(file2, "rt")
next(raw_data)
next(raw_data2)
reader = csv.reader(raw_data, delimiter=',')
reader2 = csv.reader(raw_data2, delimiter=',')
x = list(reader)
y = list(reader2)
input_data = np.array(x).astype('float')
target_data = np.array(y).astype('float')
pca = PCA(n_components=6)
pca.fit(input_data)
print(pca.explained_variance_ratio_)
