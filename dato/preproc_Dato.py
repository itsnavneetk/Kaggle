

from __future__ import print_function

from collections import Counter
import glob
import multiprocessing
import os
import re
import sys
import time

import pandas as pd

print('--- Read training labels')
train_labels = pd.read_csv('data/train_v2.csv')
train_keys = dict([a[1] for a in train_labels.iterrows()])
test_files = set(pd.read_csv('data/sampleSubmission_v2.csv').file.values)

def create_data(filepath):
    values = {}
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as infile:
        text = infile.read()
    values['file'] = filename
    if filename in train_keys:
        values['sponsored'] = train_keys[filename]
    values['lines'] = text.count('\n')
    values['spaces'] = text.count(' ')
    values['tabs'] = text.count('\t')
    values['braces'] = text.count('{')
    values['brackets'] = text.count('[')
    values['words'] = len(re.split('\s+', text))
    values['length'] = len(text)
    
    values['http'] = text.count('http')
    values['meta'] = text.count('meta')
    values['link'] = text.count('link')
    values['img'] = text.count('img')
    values['script'] = text.count('script')
    values['href'] = text.count('href')
    values['style'] = text.count('style')
    values['<div'] = text.count('<div')
    values['<a'] = text.count('<a')
    values['<p'] = text.count('<p')
    values['<span'] = text.count('<span')
    values['<ul'] = text.count('<ul')
    values['<li'] = text.count('<li')
    
    return values

filepaths = glob.glob('data/*/*.txt')

num_tasks = len(filepaths)

print(num_tasks)

print("working...")

results = map(create_data, filepaths)

print("completed mapping")

df_full = pd.DataFrame(list(results))

train = df_full[df_full.sponsored.notnull()].fillna(0)
test = df_full[df_full.sponsored.isnull() & df_full.file.isin(test_files)].fillna(0)

print("writing to csv")

train.to_csv('train_feats.csv', index=False)
test.to_csv('test_feats.csv', index=False)