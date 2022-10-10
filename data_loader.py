import numpy as np
import csv
from sklearn.decomposition import PCA

class Data_loader():
  """
  _input_file: data of input
  _output_file: data of output
  """
  def __init__(self, input_file: str, output_file: str):
    self._input_file = input_file
    self._output_file = output_file

  def give_array(self):
    """
    return the array representation of data in input_file and output_file
    """
    raw_data = open(self._input_file, "rt")
    raw_data2 = open(self._output_file, "rt")
    next(raw_data)
    next(raw_data2)
    reader = csv.reader(raw_data, delimiter=',')
    reader2 = csv.reader(raw_data2, delimiter=',')
    x = list(reader)
    y = list(reader2)
    input_data = np.array(x).astype('float')
    target_data = np.array(y).astype('float')
    return [input_data, target_data]


  def do_pca(self):
    """
    Do pca on the given input and output data.
    """
    raw_data = open(self._input_file, "rt")
    raw_data2 = open(self._output_file, "rt")
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
    
    return pca.explained_variance_ratio_
