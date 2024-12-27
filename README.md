Points to be improved:

1. Fix the warning that happens during the calculation of the Kurtosis:

C:\Users\Usuario\Desktop\QuickAccess\University\Master\data_streaming_processing\anomaly_detection_project\streamRHFPython\streamRHF.py:27: SmallSampleWarning: After omitting NaNs, one or more axis-slices of one or more sample arguments is too small; corresponding elements of returned arrays will be NaN. See documentation for sample size requirements.
  kurtosis_values = kurtosis(data, fisher=False, nan_policy='omit')

In the subtrees, the number of instances is low (2,3,4) and it seems to be causing problems. 

2. Function compute_scores()

Currently it is using a for loop until it finds the instance in the tree. To optimize it, we could maybe use a hash table to save the position of each added instance in the trees, then the computation of the score could be done in constant time O(1).

