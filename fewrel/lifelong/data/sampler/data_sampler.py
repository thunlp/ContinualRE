import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import wordninja
import random

class sequence_data_sampler(object):

	def __init__(self, data_sampler, seed = None):
		self.data_sampler = data_sampler
		self.batch = 0
		self.len = data_sampler.num_clusters
		if data_sampler.seed != None:
			random.seed(data_sampler.seed)
		self.shuffle_index = list(range(self.len))
		random.shuffle(self.shuffle_index)
		self.shuffle_index = np.argsort(self.shuffle_index)
		self.seen_relations = []
		self.history_test_data = []

	def __iter__(self):
		return self

	def __next__(self): 
		if self.batch == self.len:
			raise StopIteration()
		index = self.shuffle_index[self.batch]
		self.batch += 1
		
		training_data = self.data_sampler.splited_training_data[index]
		valid_data = self.data_sampler.splited_valid_data[index]
		test_data = self.data_sampler.splited_test_data[index]

		current_relations = []
		for data in training_data:
			if data[0] not in self.seen_relations:
				self.seen_relations.append(data[0])
			if data[0] not in current_relations:
				current_relations.append(data[0])

		cur_training_data = self.remove_unseen_relation(training_data, self.seen_relations)
		cur_valid_data = self.remove_unseen_relation(valid_data, self.seen_relations)
		self.history_test_data.append(test_data)

		cur_test_data = []
		for j in range(self.batch):
			cur_test_data.append(self.remove_unseen_relation(self.history_test_data[j], self.seen_relations))
		return cur_training_data, cur_valid_data, cur_test_data, self.data_sampler.test_data, self.seen_relations, current_relations

	def __len__(self):
		return self.len
	
	def remove_unseen_relation(self, dataset, seen_relations):
		cleaned_data = []
		for data in dataset:
			neg_cands = [cand for cand in data[1] if cand in seen_relations and cand != data[0]]
			if len(neg_cands) > 0:
				cleaned_data.append([data[0], neg_cands, data[2], data[3]])
			else:
				if self.data_sampler.config['task_name'] == 'FewRel':
					cleaned_data.append([data[0], data[1][-2:], data[2], data[3]])
		return cleaned_data

class data_sampler(object):

	def __init__(self, 
				 config = None, 
				 relation_embedding_model = None, 
				 tokenizer = None, 
				 max_length = 128, 
				 blank_padding = False):
		
		self.config = config
		# self.relation_embedding_model = relation_embedding_model
		# self.relation_embedding_model.to(config['device'])
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.blank_padding = blank_padding

		# self.training_data = self._gen_data(config['training_file'])
		# self.test_data = self._gen_data(config['test_file'])
		# self.valid_data = self._gen_data(config['valid_file'])
		# np.save('training_file', self.training_data)
		# np.save('test_data', self.test_data)
		# np.save('valid_data', self.valid_data)

		self.training_data = np.load('data/fewrel/training_file.npy', allow_pickle=True)
		self.test_data = np.load('data/fewrel/test_data.npy', allow_pickle=True)
		self.valid_data = np.load('data/fewrel/valid_data.npy', allow_pickle=True)
		
		self.relation_names, self.id2rel = self._read_relations(config['relation_file'])
		self.id2rel_pattern = {}
		for i in self.id2rel:
			tokens, length = self._transfrom_sentence(self.id2rel[i])
			self.id2rel_pattern[i] = (i, [i], tokens, length)

		self.num_clusters = config['num_clusters']
		# self.cluster_labels, self.rel_features = self._cluster_data(self.num_clusters)
		self.cluster_labels = {}
		self.rel_features = {}
		rel_index = np.load("data/fewrel/rel_index.npy")
		# copyright super Xu Han 2019
		rel_cluster_label = np.load("data/fewrel/rel_cluster_label.npy")
		rel_feature = np.load("data/fewrel/rel_feature.npy")
		for index, i in enumerate(rel_index):
			self.cluster_labels[i] = rel_cluster_label[index]
			self.rel_features[i] = rel_feature[index]

		self.splited_training_data = self._split_data(self.training_data, self.cluster_labels, self.num_clusters)
		self.splited_valid_data = self._split_data(self.valid_data, self.cluster_labels, self.num_clusters)
		self.splited_test_data = self._split_data(self.test_data, self.cluster_labels, self.num_clusters)
		self.seed = None

	# sampling files

	def __iter__(self):
		return sequence_data_sampler(self, self.seed)

	def set_seed(self, seed):
		self.seed = seed	

	# reading training, valid, test files
	def _remove_return_sym(self, str):
		return str.split('\n')[0]
	
	def _read_samples(self, file):
		sample_data = []
		with open(file) as file_in:
			for line in file_in:
				items = line.split('\t')
				if(len(items[0]) > 0):
					relation_ix = int(items[0])
					if items[1] != 'noNegativeAnswer':
						candidate_ixs = [int(ix) for ix in items[1].split()]
						question = self._remove_return_sym(items[2])
						sample_data.append([relation_ix, candidate_ixs, question])
		return sample_data

	def _transfrom_sentence(self, data):
		tokens = self.tokenizer.tokenize(data)
		length = min(len(tokens), self.max_length)
		if self.blank_padding:
			tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]'])
		else:
			tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.tokenizer.vocab['[UNK]'])
		if (len(tokens) > self.max_length):
			tokens = tokens[:self.max_length]
		return tokens, length

	def _transform_questions(self, data):
		for sample in data:
			tokens = self.tokenizer.tokenize(sample[2])
			length = min(len(tokens), self.max_length)
			if self.blank_padding:
				tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]'])
			else:
				tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.tokenizer.vocab['[UNK]'])
			if (len(tokens) > self.max_length):
				tokens = tokens[:self.max_length]
			sample[2] = tokens
			sample.append(length)
		return data

	def _gen_data(self, file):
		data = self._read_samples(file)
		data = self._transform_questions(data)
		return np.asarray(data)

	# spliting files
	
	def _split_data(self, data_set, cluster_labels, num_clusters):
		splited_data = [[] for i in range(num_clusters)]
		for data in data_set:
			splited_data[cluster_labels[data[0]]].append(data)
		return splited_data

	def _read_relations(self, file):
		relation_list = [self._split_relation_into_words(self._remove_return_sym('fill fill fill'))]
		id2rel = {0:'fill fill fill'}
		with open(file) as file_in:
			for line in file_in:
				relation_list.append(self._split_relation_into_words(self._remove_return_sym(line)))
				id2rel[len(id2rel)] = self._remove_return_sym(line)
		return relation_list, id2rel
	
	def _split_relation_into_words(self, relation):
		word_list = []
		for word_seq in relation.split("/")[-3:]:
			for word in word_seq.split("_"):
				word_list += wordninja.split(word)
		return " ".join(word_list)
	
	def _get_relation_names_in_dataset(self, relations_index_in_dataset):
		relation_names = [self.relation_names[num] for num in relations_index_in_dataset]
		return relation_names

	def _get_relations_index_in_dataset(self, data):
		relation_pool = []
		for i in data:
			relation_number = i[0]
			if relation_number not in relation_pool:
				relation_pool.append(relation_number)
		return relation_pool
	
	def _gen_relation_embedding(self):
		training_relation_index = self._get_relations_index_in_dataset(self.training_data)
		valid_relation_index = self._get_relations_index_in_dataset(self.valid_data)
		test_relation_index = self._get_relations_index_in_dataset(self.test_data)
		
		relation_index = list(training_relation_index)
		for index in test_relation_index + valid_relation_index:
			if index not in relation_index:
				relation_index.append(index)
		relation_index = np.array(relation_index)
		relation_names = self._get_relation_names_in_dataset(relation_index)
		
		tokens = []
		lengths = []
		for relation in relation_names:
			token, length = self.relation_embedding_model.tokenize(relation)
			tokens.append(token)
			lengths.append(length)
		tokens = torch.cat(tokens, 0).to(self.config['device'])
		lengths = torch.cat(lengths, 0).to(self.config['device'])
		relation_embeddings, _ = self.relation_embedding_model.predict(tokens, lengths)
		return relation_index, relation_embeddings, relation_tokens, relation_lengths

	def _cluster_data(self, num_clusters = 20, relation_names = None, train_set = None, valid_set = None, test_set = None, relation_embedding_model = None):
		relation_index, relation_embeddings = self._gen_relation_embedding()
		kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(relation_embeddings)
		#print(kmeans.inertia_)
		labels = kmeans.labels_
		rel_embed = {}
		cluster_index = {}
		for i in range(len(relation_index)):
			cluster_index[relation_index[i]] = labels[i]
			rel_embed[relation_index[i]] = relation_embeddings[i]
		rel_index = np.asarray(list(relation_index))
		return cluster_index, rel_embed
