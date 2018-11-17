import torch
from torch.utils.data import Dataset

from preproc import clean_data

import numpy as np
import pandas as pd
import json

class FolderDataset(Dataset):

	def __init__(self, path, split_by_phrase, partition, partitions, verbose=False):
		super().__init__()

		# Define set of data
		self.data = []

		clean_data_file = path + 'clean_data.tsv'

		# Check if a cleaned version of data has already been created
		if not os.path.isfile(clean_data_file):
			clean_data(path, clean_data_file)

		# Define npy dataset file name for every partition
		npy_files = []
		for partition in partitions:
			npy_names.append(path + partition + '_clean_data.npy')
		
		# Define JSON file with phrase delimiters
		json_file = path + 'phrases.json'

		# Check if dataset has to be created
		if len(npy_files) != len([f for f in npy_files if os.path.isfile()]):
			if verbose:
				print('Extracting chords from: ', clean_data_file)

			# Load pandas dataframe with all annotations
			df = pd.read_csv(clean_data_file, sep='\t')

			# Check if the JSON file with phrase divisions has to be created
			if not os.path.isfile(json_file) and split_by_phrase:
				end_idxs = df.loc[df['phraseend']].index.values
				num_phrases = len(end_idxs)
				num_chords = df.shape[0]

				phrases = []

				# JSON does not recognize NumPy data types
				# Convert the number to a Python int before serializing the object
				longest_sequence = 0
				for pos, end_idx in enumerate(end_idxs):
					if pos == 0:
						item = {"from": 1, "to": int(end_idx + 1)}
						if end_idx > longest_sequence:
							longest_sequence = end_idx
					else:
						item = {"from": int(end_idxs[pos - 1] + 1), "to": int(end_idx + 1)}
						if end_idx - end_idxs[pos - 1] > longest_sequence:
							longest_sequence = end_idx

						if pos == num_phrases - 1:
							phrases.append(item)
							item = {"from": int(end_idx) + 1, "to": num_chords}
							if num_chords - end_idx > longest_sequence:
								longest_sequence = end_idx
					phrases.append(item)

				# Append longest sequence value to pad zeros to others for uniform length
				phrases.append({"longest_sequence": longest_sequence})

				# Save all phrases to a JSON file
				with open(json_file, 'w') as outfile:
					json.dump(phrases, outfile)

			if split_by_phrase:
				with open(json_file, "r") as infile:
					phrases = json.load(infile)

				longest_sequence = phrases[-1]
				# Don't take longest sequence value into account
				num_phrases = len(phrases) - 1
				radom_phrases = np.random.permutation(num_phrases)

				from_ = 0
				for partition in partitions:
					to = np.round(num_phrases * partitions[partition] / 100)
					partition_data = []
					for phrase in phrases[radnom_phrases[from_ : to]]:
						partition_data.append(df[phrase["from"] : phrase["to"]].values)
					
					# Save numpy files
					np.save(path + partition + '_clean_data.npy', partition_data)
					print('Dataset created for ' + partition + ' partition', '-' * 60, '\n')

					from_ = to
				
				if verbose:
					print('There are still ' num_phrases - from_ ' unassigned for rounding issues')

		# Load previously created dataset
		self.data = np.load(path + partition + '_clean_data.npy')

		# Compute length for current partition
		self.length = np.prod(self.data.shape[0:2])

		print('Data shape:', self.data.shape)
		print('Dataset loaded for ' + partition + ' partition', '-' * 60, '\n')

	def __getitem__(self, index):
		# Compute which sample within n_batch has to be returned given an index
		# n_batch, sample_in_batch = divmod(index, self.batch_size)

		data = torch.from_numpy(self.data[index][:-1])
		target = torch.from_numpy(self.data[index][1:])

		# Reset all hidden states to avoid predicting with non-related samples
		# if(new phrase) reset = True

		if verbose:
			print('Data size: ', data.size())

		return data, reset, target

	def __len__(self):
		return self.length
