import pandas as pd
import pickle
from feature import *

pofile = open('positive_dataset.txt')
arrayOLines = pofile.readlines()
positive_result = []
for line in arrayOLines:
    line = line.strip().split('   ')
    positive_result.append(line)
pofile.close()
positive_result = pd.DataFrame(positive_result).dropna(axis=0)

nefile = open('negative_dataset.txt')
lines = nefile.readlines()
negative_result = []
for line in lines:
    line = line.strip().split('   ')
    negative_result.append(line)
nefile.close()
negative_result = pd.DataFrame(negative_result).dropna(axis=0)

frames = [positive_result, negative_result]
result = pd.concat(frames)
result.columns = ['p1', 'p2', 'label']

ans = []
for index, row in result.iterrows():
    p1, p2, label = row['p1'], row['p2'], row['label']
    p1_feature = feature_map(p1_path, p1, criteria)
    p2_feature = feature_map(p2_path, p2, criteria)
    alist = [p1.values(), p2.values(), int(label)]
    ans.append((alist))

with open(feature_file, 'wb') as f:
    pickle.dump(ans, f)
    f.close()