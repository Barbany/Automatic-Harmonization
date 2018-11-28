import torch
from torch.utils.data import Dataset

from preproc.clean_data import create_clean_file

import numpy as np
import pandas as pd
import json
import os


class FolderDataset(Dataset):

    def __init__(self, path, split_by_phrase, partition, partitions, verbose=False):
        super().__init__()

        # Define set of data
        self.data = []
        self.verbose = verbose

        clean_data_file = path + 'clean_data.tsv'

        # Check if a cleaned version of data has already been created
        if not os.path.isfile(clean_data_file):
            df = create_clean_file(path, clean_data_file)
        else:
            # Load pandas dataframe with all annotations
            df = pd.read_csv(clean_data_file, sep='\t')

        # Define npy dataset file name for every partition
        npy_files = []
        for part in partitions:
            npy_files.append(path + part  + '_' + split_by_phrase*'phrase-split' + 
            	(not split_by_phrase)*'sequential' + '_clean_data.npy')

        # Define JSON file with phrase delimiters
        json_file = path + 'phrases.json'

        # Check if dataset has to be created
        if len(npy_files) != len([f for f in npy_files if os.path.isfile(f)]):
            if self.verbose:
                print('Extracting chords from: ', clean_data_file)

            num_chords = df.shape[0]

            # Check if the JSON file with phrase divisions has to be created
            if not os.path.isfile(json_file) and split_by_phrase:
                end_idxs = df.loc[df['phraseend']].index.values
                num_phrases = len(end_idxs)

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
                            longest_sequence = end_idx - end_idxs[pos - 1]

                        if pos == num_phrases - 1:
                            phrases.append(item)
                            item = {"from": int(end_idx) + 1, "to": num_chords}
                            if num_chords - end_idx > longest_sequence:
                                longest_sequence = num_chords - end_idx
                    phrases.append(item)

                # Append longest sequence value to pad zeros to others for uniform length
                phrases.append({"longest_sequence": int(longest_sequence)})

                # Save all phrases to a JSON file
                with open(json_file, 'w') as outfile:
                    json.dump(phrases, outfile)

            # Split by phrases once the JSON file has been created
            if split_by_phrase:
            	# Open phrase ranges
                with open(json_file, "r") as infile:
                    phrases = json.load(infile)

                longest_sequence = phrases[-1]["longest_sequence"]

                # Don't take longest sequence value into account (additional row in JSON file)
                num_phrases = len(phrases) - 1

                # Get random indices
                random_phrases = np.random.permutation(num_phrases)

                from_ = 0
                for part_idx, part in enumerate(partitions):
                    to = from_ + int(np.round(num_phrases * partitions[part] / 100))
                    if verbose:
                    	print('Select random phrases from ', from_, ' to ', to)
                    # Prellocate numpy array (phrases x sequence index x label and features)
                    # We now fill it with special character 'x' to avoid doing padding every time
                    part_data = np.array(['x' for _ in range((to - from_ + 1) * longest_sequence * len(df.columns))])
                    part_data = part_data.reshape(-1, longest_sequence, len(df.columns))
                    if verbose:
                    	print('Data for', part, 'partition has shape', part_data.shape)

                    for phrase_num, phrase_idx in enumerate(random_phrases[from_: to]):
                    	phrase = phrases[phrase_idx]
                    	aux = df[phrase["from"]: phrase["to"]].values
                    	part_data[phrase_num, range(len(aux))] = aux
                    	# if verbose:
                    		# print('Processing phrase (with randomized index)', phrase_num)

                    # Save numpy files
                    np.save(npy_files[part_idx], part_data)
                    print('Dataset created for ' + part + ' partition', '-' * 60, '\n')

                    from_ = to

                if self.verbose:
                    print('There are still ' + str(num_phrases - from_) + ' phrases unassigned for rounding issues')

            else:
                # TODO: Take into account change of movement/quartet?
                from_ = 0
                for part_idx, part in enumerate(partitions):
                    to = int(np.round(num_chords * partitions[part] / 100))
                    part_data = df[from_:to]
                    np.save(npy_files[part_idx], part_data)
                    print('Dataset created for ' + part + ' partition', '-' * 60, '\n')

                    from_ = to

                if self.verbose:
                    print('There are still ' + str(num_phrases - from_) + ' phrases unassigned for rounding issues')

        # Load previously created dataset
        self.data = np.load(path + partition  + '_' + split_by_phrase*'phrase-split' + 
            	(not split_by_phrase)*'sequential' + '_clean_data.npy')

        # Compute length for current partition
        self.length = self.data.shape[0]

        print('Data shape:', self.data.shape)
        print('Dataset loaded for ' + partition + ' partition', '-' * 60, '\n')

    def __getitem__(self, index):
        # Compute which sample within n_batch has to be returned given an index
        # n_batch, sample_in_batch = divmod(index, self.batch_size)

        data = torch.from_numpy(self.data[index][:-1])
        target = torch.from_numpy(self.data[index][1:])

        # Reset all hidden states to avoid predicting with non-related samples
        # if(new phrase) reset = True
        reset = False

        if self.verbose:
            print('Data size: ', data.size())

        return data, reset, target

    def __len__(self):
        return self.length
