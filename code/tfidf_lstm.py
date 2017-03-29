import pandas as pd
import numpy as np

label_col = [i + 1 for i in range(10)]
df_label = pd.read_csv('./data/DATA_TFIDF_HADM_TOP10.csv', usecols=label_col)
label = df_label.values

df_features = pd.read_csv('./data/DATA_TFIDF_HADM_TOP10.csv', usecols=[11])
x = df_features.values


print('done')