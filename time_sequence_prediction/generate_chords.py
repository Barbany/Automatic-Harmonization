import numpy as np
import torch
import pandas as pd
import json

seq_len = 30

df = pd.read_csv('../data/clean_data.tsv', sep='\t')
df = df['chord'].values

with open('../data/mappings.json', "r") as infile:
    mapping = json.load(infile)

chords = np.vectorize(mapping['chord'].get)(df)

num_chords = len(chords)

chords = chords[:(num_chords - num_chords % seq_len)].reshape(-1, seq_len)
print(chords.shape) # (936, 30)

torch.save(chords, open('traindata.pt', 'wb'))