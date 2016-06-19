import numpy as np
import pandas as pd
import renders as rs

from IPython.display import display
from sklearn.cross_validation import train_test_split


data = pd.read_csv("customers.csv")


means = data.groupby(['Channel']).mean()

