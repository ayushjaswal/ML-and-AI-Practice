# use odd number for k -> advised
# if big data set -> high number of k
# if small data set -> low number of k
# weights -> a "uniform" parameter gives equal importance to all datapoint
# weights -> a "distant" parameter gives greater importance to closer datapoint

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
