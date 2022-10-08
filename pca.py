import numpy as np
import csv

file1 = "nse_input.csv"
file2 = "nse_target.csv"
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
print(input_data[:5], target_data[:5])