import torch
from torch.utils.data import Dataset

import numpy as np
import csv

class FolderDataset(Dataset):

	def __init__(self, path, batch_size, cond_dim, partition):
		super().__init__()

		# Define class variables from initialization parameters
		self.batch_size = batch_size

		# Define sets of data, conditioners and speaker IDs
		self.data = []

		# Define npy training dataset file name
		npy_name = 'npy_datasets/' + partition + '/clean_data.npy'

		# Check if dataset has to be created
		create_dataset = not os.path.isfile(npy_name)

		if create_dataset:
			print('Create ' + partition + ' dataset', '-' * 60, '\n')
			print('Extracting chords from: ', path)

			# Get file names from partition's list list
			chord_nums = open(path + 'chors_' + partition + '.list', 'r').read().splitlines()

			# Load each of the files from the list. Note that extension has to be added
			with open(path + 'all_annotations.tsv') as fd:
			    reader = csv.reader(fd, delimiter='\t')
			    d = [row for idx, row in enumerate(reader) if idx in chord_nums]

				# Append/Concatenate current audio file, speech conditioners and speaker ID
				self.data = np.append(self.data, d)
			# TODO: Apply pre-processing techniques

			total_samples = self.data.shape[0]
			dim_cond = self.cond.shape[1]
			print('Total samples: ', total_samples)

			lon_seq = self.seq_len+self.overlap_len
			self.num_samples = self.batch_size*(total_samples//(self.batch_size*lon_seq*self.cond_len))

			print('Number of samples (1 audio file): ', self.num_samples)
			self.total_samples = self.num_samples * (self.seq_len+self.overlap_len) * self.cond_len
			total_conditioning = self.total_samples//self.cond_len
			self.data = self.data[:self.total_samples]
			self.cond = self.cond[:total_conditioning]
			self.data = self.data[:self.total_samples].reshape(self.batch_size, -1)

			self.length = self.total_samples // self.seq_len

			self.cond = self.cond[:total_conditioning].reshape(self.batch_size, -1, dim_cond)

			self.global_spk = self.global_spk[:total_conditioning].reshape(self.batch_size, -1)

			self.audio = self.audio[:total_conditioning].reshape(self.batch_size, -1)

			# Save maximum and minimum of de-normalized conditioners for conditions of train partition
			if partition == 'train' and not os.path.isfile(npy_name_min_max_cond):
				# Compute maximum and minimum of de-normalized conditioners of train partition
				if norm_ind:
					print('Computing maximum and minimum values for each speaker of training dataset.')
					num_spk = len(spk)
					self.max_cond = np.empty(shape=(num_spk, cond_dim))
					self.min_cond = np.empty(shape=(num_spk, cond_dim))
					for i in range(num_spk):
						print('Computing speaker', i, 'of', num_spk, 'with ID:', spk[i])
						self.max_cond[i] = np.amax(self.cond[self.global_spk == i], axis=0)
						self.min_cond[i] = np.amin(self.cond[self.global_spk == i], axis=0)
				else:
					print('Computing maximum and minimum values for every speaker of training dataset.')
					self.max_cond = np.amax(np.amax(self.cond, axis=1), axis=0)
					self.min_cond = np.amin(np.amin(self.cond, axis=1), axis=0)
				np.save(npy_name_min_max_cond, np.array([self.min_cond, self.max_cond]))

			# Load maximum and minimum of de-normalized conditioners
			else:
				self.min_cond = np.load(npy_name_min_max_cond)[0]
				self.max_cond = np.load(npy_name_min_max_cond)[1]

			# Normalize conditioners with absolute maximum and minimum for each speaker of training partition
			if norm_ind:
				# Normalize conditioners with absolute maximum and minimum for each speaker of training partition
				print('Normalizing conditioners for each speaker of training dataset.')
				for i in range(len(spk)):
					self.cond[self.global_spk == i] = (self.cond[self.global_spk == i] - self.min_cond[i]) / \
													  (self.max_cond[i] - self.min_cond[i])
			else:
				# Normalize conditioners with absolute maximum and minimum for each speaker of training partition
				print('Normalizing conditioners for all speakers of training dataset.')
				self.cond = (self.cond - self.min_cond) / (self.max_cond - self.min_cond)

			# Save partition's dataset
			np.save(npy_name_data, self.data)
			np.save(npy_name_cond, self.cond)
			np.save(npy_name_spk, self.global_spk)
			np.save(npy_name_audio_id, self.audio)

			print('Dataset created for ' + partition + ' partition', '-' * 60, '\n')

		else:
			# Load previously created dataset
			self.data = np.load(npy_name_data)
			self.global_spk = np.load(npy_name_spk)

			if look_ahead:
				if os.path.isfile(npy_name_cond.replace('.npy', '_ahead.npy')):
					self.cond = np.load(npy_name_cond.replace('.npy', '_ahead.npy'))
				else:
					self.cond = np.load(npy_name_cond)
					delayed = np.copy(self.cond)
					delayed[:, :-1, :] = delayed[:, 1:, :]
					self.cond = np.concatenate((self.cond, delayed), axis=2)
					np.save(npy_name_cond.replace('.npy', '_ahead.npy'), self.cond)
			else:
				self.cond = np.load(npy_name_cond)

			# Load maximum and minimum of de-normalized conditioners
			self.min_cond = np.load(npy_name_min_max_cond)[0]
			self.max_cond = np.load(npy_name_min_max_cond)[1]

			# Compute length for current partition
			self.length = np.prod(self.data.shape) // self.seq_len

			print('Data shape:', self.data.shape)
			print('Conditioners shape:', self.cond.shape)
			print('Global speaker shape:', self.global_spk.shape)

			print('Dataset loaded for ' + partition + ' partition', '-' * 60, '\n')

	def __getitem__(self, index):
		verbose = False

		# Compute which sample within n_batch has to be returned given an index
		n_batch, sample_in_batch = divmod(index, self.batch_size)

		# Compute start and end for both input data and target sequences
		start_data = n_batch * self.seq_len
		start_target = start_data + self.overlap_len
		end_target = start_target + self.seq_len

		if not self.ulaw:
			data = torch.from_numpy(self.data[sample_in_batch][start_data:end_target-1]).long()
			target = torch.from_numpy(self.data[sample_in_batch][start_target:end_target]).long()
		else:
			data = self.quantize(torch.from_numpy(self.data[sample_in_batch][start_data:end_target-1]), self.q_levels)
			target = self.quantize(torch.from_numpy(self.data[sample_in_batch][start_target:end_target]), self.q_levels)

		# Count number of acoustic parameters computations in a sequence (1 computation every 80 audio samples)
		cond_in_seq = self.seq_len//self.cond_len

		if n_batch == 0:        # Reset all hidden states to avoid predicting with non-related samples
			reset = True
			from_cond = n_batch * cond_in_seq + 1
		else:
			reset = False
			from_cond = n_batch * cond_in_seq + 1

		to_cond = from_cond + cond_in_seq

		if verbose:
			print('batch', n_batch)
			print('sample in batch', sample_in_batch)
			print('from cond', from_cond)
			print('to cond', to_cond)

		cond = torch.from_numpy(self.cond[sample_in_batch][from_cond:to_cond])

		# Get the speaker ID for each conditioner in the sequence
		global_spk = self.global_spk[sample_in_batch][from_cond:to_cond]

		# Assume most repeated speaker as it doesn't matter on transitions from one audio to another
		global_spk = np.argmax(np.bincount(global_spk.astype(int)))

		spk = torch.from_numpy(np.array([global_spk]))

		if verbose:
			print('data size: ', data.size(), 'with sequence length: ', self.seq_len, 'and overlap: ', self.overlap_len)
			print('conditioner size: ', cond.size())
			print('speaker size: ', spk.size())

		return data, reset, target, cond, spk

	def __len__(self):
		return self.length
