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
from utils import process_testing_samples, process_samples, ranking_sequence,\
    get_grad_params, copy_param_data
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
            #cleaned_data.append(data)
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            cleaned_data.append([data[0], data[1][-2:], data[2]])
            pass
    return cleaned_data

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')

def gen_fisher(model, train_data, all_relations):
    num_correct = 0
    #testing_data = testing_data[0:100]
    softmax_func = nn.LogSoftmax()
    loss_func = nn.NLLLoss()
    fisher_batch_size = 1
    batch_epoch = (len(train_data)-1)//fisher_batch_size+1
    fisher = None
    for i in range(batch_epoch):
        model.zero_grad()
        losses = []
        samples = train_data[i*fisher_batch_size:(i+1)*fisher_batch_size]
        questions, relations, relation_set_lengths = process_samples(
            samples, all_relations, device)
        #print('got data')
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs =\
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        #print(pad_questions)
        pad_questions = pad_questions.to(device)
        pad_relations = pad_relations.to(device)
        #print(pad_questions)

        model.init_hidden(device, sum(relation_set_lengths))
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths)
        all_scores = all_scores.to('cpu')
        start_index = 0
        for length in relation_set_lengths:
            scores = all_scores[start_index:start_index+length]
            start_index += length
            losses.append(loss_func(softmax_func(scores).view(1, -1),
                                    torch.tensor([0])))
        loss_batch = sum(losses)
        #print(loss_batch)
        loss_batch.backward()
        grad_params = get_grad_params(model)
        #for param in grad_params:
         #   print(param.grad)
        if fisher is None:
            fisher = [param.grad**2/batch_epoch
                         for param in grad_params]
        else:
            fisher = [fisher[i]+param.grad**2/batch_epoch
                         for i,param in enumerate(grad_params)]

    return fisher

def get_mean_fisher(model, train_data, all_relations):
    grad_params = get_grad_params(model)
    grad_mean = copy_param_data(grad_params)
    grad_fisher = gen_fisher(model, train_data, all_relations)
    return grad_mean, grad_fisher

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary, embedding, cluster_labels, num_clusters,
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
    grads_means = []
    grads_fishers = []
    sequence_results = []
    #np.set_printoptions(precision=3)
    result_whole_test = []
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
        current_model = train(current_train_data, current_valid_data,
                              vocabulary, embedding_dim, hidden_dim,
                              device, batch_size, lr, model_path,
                              embedding, all_relations, current_model, epoch,
                              grads_means, grads_fishers, loss_margin)
        grad_mean, grad_fisher = get_mean_fisher(current_model,
                                                 current_train_data,
                                                 all_relations)
        #print(grad_mean)
        grads_means.append(grad_mean)
        grads_fishers.append(grad_fisher)
        results = [evaluate_model(current_model, test_data, batch_size,
                                  all_relations, device)
                   for test_data in current_test_data]
        print_list(results)
        sequence_results.append(np.array(results))
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
    random_seed = int(sys.argv[1])
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels = cluster_data(num_clusters)
    start_time = time.time()
    all_results = []
    result_all_test_data = []
    for i in range(sequence_times):
        shuffle_index = list(range(num_clusters))
        random_seed = int(sys.argv[1]) + 100*i
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
