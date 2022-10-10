import numpy as np

class Modifier():
  """
  """
  
  def __init__(self, input_array, pca_result):
    """
    """
    self._input_array = input_array
    self._pca_result = pca_result
  
  def modify(self):
    """
    """
    indices_remove = []
    for i in range(len(self._pca_result)):
      if self._pca_result[i] < 0.1:
        indices_remove.append(i)
    count = 0
    for index in indices_remove:
      self._input_array = np.delete(self._input_array, index-count)
      count += 1

    return self._input_array
