import torch
from torch.utils.data import Dataset

from preproc.clean_data import create_clean_file
from utils.params import default_params

import numpy as np
import pandas as pd
import json
import os


class FolderDataset(Dataset):
    def __init__(self, path, split_by_phrase, len_seq_phrase, partition, partitions, batch_size, verbose=False):
        super().__init__()

        # Define TSV for clean data and JSON files for mappings (numeral -> float) and for phrase delimiters
        clean_data_file = path + 'clean_data.tsv'
        mapping_file = path + 'mappings.json'
        json_file = path + 'phrases.json'

        # Define set of data, verbosity level and vocabulary size (of chords and all the database)
        self.data = []
        self.verbose = verbose

        # Define npy dataset file name for every partition
        npy_files = []
        for part in partitions:
            npy_files.append(path + part + '_' + split_by_phrase * 'phrase-split' +
                             (not split_by_phrase) * 'sequential' + '_clean_data.npy')

        # Check if a cleaned version of data has already been created
        if not os.path.isfile(clean_data_file):
            df = create_clean_file(path, clean_data_file, mapping_file, verbose)

        else:
            # Load pandas dataframe with all annotations
            df = pd.read_csv(clean_data_file, sep='\t')

        # Check if dataset has to be created
        if len(npy_files) != len([f for f in npy_files if os.path.isfile(f)]):

            with open(mapping_file, "r") as infile:
                mapping = json.load(infile)

            if self.verbose:
                print('Extracting chords from: ', clean_data_file)

            for col in df.columns:
                if col in mapping:
                    df[col] = np.vectorize(mapping[col].get)(df[col])

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
                    if self.verbose:
                        print('Select random phrases from ', from_, ' to ', to)
                    # Prellocate numpy array (phrases x sequence index x label and features)
                    # We now fill it with sentinel (infinity value) to avoid doing padding every time
                    part_data = np.ones(((to - from_ + 1), longest_sequence, len(df.columns))) * float(self.vocabulary_size)

                    if self.verbose:
                        print('Data for', part, 'partition has shape', part_data.shape)

                    for phrase_num, phrase_idx in enumerate(random_phrases[from_: to]):
                        phrase = phrases[phrase_idx]
                        aux = df[phrase["from"]: phrase["to"]].values
                        part_data[phrase_num, range(len(aux))] = aux

                    # Save numpy files
                    np.save(npy_files[part_idx], part_data)
                    print('Dataset created for ' + part + ' partition', '-' * 60, '\n')

                    from_ = to

                if self.verbose:
                    print('There are still ' + str(num_phrases - from_) + ' phrases unassigned for rounding issues')

            else:
                # list with the indexes where there is a change of movement
                # print("There are {} chords".format(num_chords))
                changes_mov = [0] + list(idx for idx, (i, j) in enumerate(zip(df['mov'], df['mov'][1:]), 1) if i != j)
                # print("Changes of movement at {}".format(changes_mov))

                # TODO: Take into account change of movement/quartet?
                from_ = 0
                for part_idx, part in enumerate(partitions):
                    to = from_ + int(np.round(num_chords * partitions[part] / 100))
                    if self.verbose:
                        print("Taking {0}% of the chords for {1}, that are {2}".format(partitions[part], part, to))
                    if to not in changes_mov and part != 'test':
                        closest_end_mov = self.__find_nearest(changes_mov, to)
                        to = closest_end_mov
                    if part == 'test':
                        to = num_chords
                    # We don't need the movement to train the model
                    df_model = df.drop('mov', 1)
                    padding_size = len_seq_phrase - (to - from_) % len_seq_phrase
                    padding = np.ones((padding_size, len(df_model.columns)), dtype=float) * float(self.vocabulary_size)

                    part_data = np.array(df_model[from_:to], dtype=float)
                    part_data = np.concatenate((part_data, padding), axis=0)
                    part_data = part_data.reshape(-1, len_seq_phrase, len(df_model.columns))

                    np.save(npy_files[part_idx], part_data)
                    if self.verbose:
                        print('Type of data is', part_data.dtype)
                        print('Dataset created for ' + part + ' partition\n', '-' * 60, '\n')

                    from_ = to

        # Load previously created dataset
        self.data = np.load(path + partition + '_' + split_by_phrase * 'phrase-split' +
                            (not split_by_phrase) * 'sequential' + '_clean_data.npy')

        # Compute length for current partition
        self.length = self.data.shape[0]

        print('Data shape:', self.data.shape)
        print('Dataset loaded for ' + partition + ' partition\n' + '-' * 60, '\n')


    def __getitem__(self, index):
        # Compute which sample within n_batch has to be returned given an index
        # n_batch, sample_in_batch = divmod(index, self.batch_size)
        chords = self.data[index][:, 0]
        features = self.data[index][:, 1:]

        # TODO: Put a certain batch size? For now it's one
        #  - Do expand_dim to chords because this is not considered when creating dataset
        # TODO: All sequence or chords[:-1] for first parameter??
        chords_torch = torch.from_numpy(np.expand_dims(chords, axis=0))
        target = torch.from_numpy(chords[1:])
        features_torch = torch.from_numpy(features)

        return chords_torch, target, features_torch

    def __len__(self):
        return self.length

    @staticmethod
    def __find_nearest(array, value):
        # find the element in the array with a nearest value to the param value
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
