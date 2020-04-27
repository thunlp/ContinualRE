## The code is based on the implementation from:
## https://github.com/hongwang600/RelationDectection
##
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import time

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence
from evaluate import evaluate_model
from data_partition import cluster_data
from config import CONFIG as conf
from train import train

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
device = conf['device']
num_clusters = conf['num_clusters']
lr = conf['learning_rate']
model_path = conf['model_path']
epoch = conf['epoch']
random_seed = conf['random_seed']
task_memory_size = conf['task_memory_size']
loss_margin = conf['loss_margin']
sequence_times = conf['sequence_times']

def split_data(data_set, cluster_labels, num_clusters, shuffle_index):
    splited_data = [[] for i in range(num_clusters)]
    for data in data_set:
        cluster_number = cluster_labels[data[0]]
        index_number = shuffle_index[cluster_number]
        splited_data[index_number].append(data)
    return splited_data

# remove unseen relations from the dataset
def remove_unseen_relation(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            #data[1] = neg_cands
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            #cleaned_data.append([data[0], data[1][-2:], data[2]])
            pass
    return cleaned_data

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')

def sample_memory_data(sample_pool, sample_size):
    if len(sample_pool) > 0:
        sample_indexs = random.sample(range(len(sample_pool)),
                                      min(sample_size, len(sample_pool)))
        return [sample_pool[index] for index in sample_indexs]
    else:
        return []

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary,embedding, cluster_labels, num_clusters,
                 shuffle_index):
    splited_training_data = split_data(training_data, cluster_labels,
                                       num_clusters, shuffle_index)
    splited_valid_data = split_data(valid_data, cluster_labels,
                                    num_clusters, shuffle_index)
    splited_test_data = split_data(testing_data, cluster_labels,
                                   num_clusters, shuffle_index)
    #print(splited_training_data)
    '''
    for data in splited_training_data[0]:
        print(data)
        print(cluster_labels[data[0]])
    '''
    #print(cluster_labels)
    seen_relations = []
    current_model = None
    memory_data = []
    all_seen_samples = []
    sequence_results = []
    rel_sample_count = {}
    #np.set_printoptions(precision=3)
    result_whole_test = []
    all_seen_rels = []
    for i in range(num_clusters):
        seen_relations += [data[0] for data in splited_training_data[i] if
                          data[0] not in seen_relations]
        current_train_data = remove_unseen_relation(splited_training_data[i],
                                                    seen_relations)
        current_valid_data = remove_unseen_relation(splited_valid_data[i],
                                                    seen_relations)
        current_test_data = []
        for j in range(i+1):
            current_test_data.append(
                remove_unseen_relation(splited_test_data[j], seen_relations))
        #memory_data = sample_memory_data(all_seen_samples, task_memory_size)
        #print(memory_data)
        for this_sample in current_train_data:
            if this_sample[0] not in all_seen_rels:
                all_seen_rels.append(this_sample[0])
        current_model = train(current_train_data, current_valid_data,
                              vocabulary, embedding_dim, hidden_dim,
                              device, batch_size, lr, model_path,
                              embedding, all_relations, current_model, epoch,
                              all_seen_samples, task_memory_size, loss_margin,
                              all_seen_rels)
        results = [evaluate_model(current_model, test_data, batch_size,
                                  all_relations, device)
                   for test_data in current_test_data]
        print_list(results)
        sequence_results.append(np.array(results))
        for this_data in current_train_data:
            pos_index = this_data[0]
            if pos_index not in rel_sample_count:
                rel_sample_count[pos_index] = 1
                all_seen_samples.append(this_data)
            elif rel_sample_count[pos_index] < 10:
                rel_sample_count[pos_index] += 1
                all_seen_samples.append(this_data)
        result_whole_test.append(evaluate_model(current_model,
                                                testing_data, batch_size,
                                                all_relations, device))
    print('test set size:', [len(test_set) for test_set in current_test_data])
    return sequence_results, result_whole_test

def print_avg_results(all_results):
    avg_result = []
    for i in range(len(all_results[0])):
        avg_result.append(np.average([result[i] for result in all_results], 0))
    for line_result in avg_result:
        print_list(line_result)
    return avg_result

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels = cluster_data(num_clusters)
    random.seed(random_seed)
    start_time = time.time()
    all_results = []
    result_all_test_data = []
    seeds = [0,10,20]
    for seed in seeds:
        for i in range(sequence_times):
            shuffle_index = list(range(num_clusters))
            random_seed = seed + 100*i
            random.seed(random_seed)
            random.shuffle(shuffle_index)
            sequence_results, result_whole_test = run_sequence(
                training_data, testing_data, valid_data, all_relations,
                vocabulary, embedding, cluster_labels, num_clusters, shuffle_index)
            all_results.append(sequence_results)
            result_all_test_data.append(result_whole_test)
    avg_result_all_test = np.average(result_all_test_data, 0)
    for result_whole_test in result_all_test_data:
        print_list(result_whole_test)
    print_list(avg_result_all_test)
    print_avg_results(all_results)
    end_time = time.time()
    #elapsed_time = end_time - start_time
    elapsed_time = (end_time - start_time) / sequence_times
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
