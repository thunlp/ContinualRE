import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import time
import json
import os
from tqdm import tqdm, trange
from sklearn.cluster import KMeans

import lifelong 
from lifelong.model.encoder import lstm_encoder
from lifelong.model.module import proto_softmax_layer, simple_lstm_layer
from lifelong.data.sampler import data_sampler
from lifelong.utils import set_seed
from lifelong.utils import outputer
from data_loader import get_data_loader

def evaluate_model(config, model, test_set, num_class):
    model.eval()
    data_loader = get_data_loader(config, test_set, False, False)
    num_correct = 0
    total = 0.0
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        logits, rep = model(sentences, lengths)
        distances = model.get_mem_feature(rep)
        logits = logits
        short_logits = distances
        for index, logit in enumerate(logits):
            score = short_logits[index]#logits[index] + short_logits[index] + long_logits[index]
            total += 1.0
            golden_score = score[labels[index]]
            max_neg_score = -2147483647.0
            for i in neg_labels[index]: #range(num_class): 
                if (i != labels[index]) and (score[i] > max_neg_score):
                    max_neg_score = score[i]
            if golden_score > max_neg_score:
                num_correct += 1
    return num_correct / total

def get_memory(config, model, proto_set):
    memset = []
    resset = []
    rangeset= [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader(config, memset, False, False)
    features = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    protos = []
    print ("proto_instaces:%d"%len(features))
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
    protos = torch.cat(protos, 0)
    return protos

# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(mem_set, proto_set, config, model, sample_set, num_sel_data):
    data_loader = get_data_loader(config, sample_set, False, False)
    features = []
    labels = []
    for step, (labels, neg_labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(num_sel_data, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = sample_set[sel_index]
        mem_set.append(instance)
        proto_set[instance[0]].append(instance)
    return mem_set

# Use K-Means to select what samples to save
def select_data_twice(mem_set, proto_set, config, model, sample_set, num_sel_data, at_least = 3):
    data_loader = get_data_loader(config, sample_set, False, False)
    features = []
    for step, (_, _, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(num_sel_data, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    rel_info = {}
    rel_alloc = {}
    for index, instance in emumerate(sample_set):
        if not instance[0] in rel_info:
            rel_info[instance[0]] = []
            rel_alloc[instance[0]] = 0
        rel_info[instance[0]].append(index)
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = sample_set[sel_index]
        rel_alloc[instance[0]] += 1
    rel_alloc = [(i, rel_alloc[i]) for i in rel_alloc]
    at_least = min(at_least, num_sel_data // len(rel_alloc))
    while True:
        rel_alloc = sorted(rel_alloc, key=lambda num : num[1], reverse = True)
        if rel_alloc[-1][1] >= at_least:
            break
        index = 0
        while rel_alloc[-1][1] < at_least:
            if rel_alloc[index][1] <= at_least:
                index = 0
            rel_alloc[-1][1] += 1
            rel_alloc[index][1] -= 1
            index+=1
    print (rel_alloc)
    for i in rel_alloc:
        label = i[0]
        num = i[1]
        tmp_feature = features[rel_info[label]]
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(tmp_feature)

    mem_set.append(instance)
    proto_set[instance[0]].append(instance)
    return mem_set

def train_simple_model(config, model, train_set, epochs):
    data_loader = get_data_loader(config, train_set)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, neg_labels, sentences, lengths) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, _ = model(sentences, lengths)
            labels = labels.to(config['device'])
            loss = criterion(logits, labels)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
        print (np.array(losses).mean())
    return model

def train_model(config, model, mem_set, epochs, current_proto):
    data_loader = get_data_loader(config, mem_set, batch_size = 5)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        # current_proto = get_memory(config, model, proto_memory)
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (labels, neg_labels, sentences, lengths) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, rep = model(sentences, lengths)
            logits_proto = model.mem_forward(rep)
            labels = labels.to(config['device'])
            loss = (criterion(logits_proto, labels))
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
    return model

if __name__ == '__main__':

    f = open("config/config_fewrel.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    
    root_path = '.'
    word2id = json.load(open(os.path.join(root_path, 'glove/word2id.txt')))
    word2vec = np.load(os.path.join(root_path, 'glove/word2vec.npy'))
    encoder = lstm_encoder(
        token2id = word2id, 
        word2vec = word2vec, 
        word_size = len(word2vec[0]), 
        max_length = 128, 
        pos_size = None, 
        hidden_size = config['hidden_size'], 
        dropout = 0, 
        bidirectional = True, 
        num_layers = 1, 
        config = config)
    sampler = data_sampler(config,None, encoder.tokenizer)
    model = proto_softmax_layer(
        encoder, 
        num_class = len(sampler.id2rel), 
        id2rel = sampler.id2rel, 
        drop = 0, 
        config = config)
    model = model.to(config["device"])
    word2vec_back = word2vec.copy()
    
    printer = outputer()
    for i in range(5):
        
        set_seed(config, config['random_seed'] + 100 * i)
        sampler.set_seed(config['random_seed'] + 100 * i)

        sequence_results = []
        result_whole_test = []
        mem_data = []
        proto_memory = []
        num_class = len(sampler.id2rel)
        for i in range(len(sampler.id2rel)):
            proto_memory.append([sampler.id2rel_pattern[i]])

        for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
            mem_data_back = mem_data.copy()
            proto_memory_back = []
            for i in range(len(sampler.id2rel)):
                proto_memory_back.append((proto_memory[i]).copy())

            model = train_simple_model(config, model, mem_data + training_data, 1)
            select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'])

            for i in range(2):
                current_proto = get_memory(config, model, proto_memory)
                # model.set_memorized_prototypes(current_proto)
                model = train_simple_model(config, model, mem_data + training_data, 1)
                model = train_model(config, model, mem_data, 1, current_proto)
            
            # mem_data = mem_data_back
            # proto_memory = proto_memory_back
            # select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'])

            current_proto = get_memory(config, model, proto_memory)
            model.set_memorized_prototypes(current_proto)
            results = [evaluate_model(config, model, item, num_class) for item in test_data]
            print ((np.array(results)).mean())
            printer.print_list(results)
            sequence_results.append(np.array(results))
            result_whole_test.append(evaluate_model(config, model, test_all_data, num_class))
        # store the result
        printer.append(sequence_results, result_whole_test)
        # initialize the models 
        model = model.to('cpu')
        del model
        torch.cuda.empty_cache()
        encoder = lstm_encoder(
            token2id = word2id, 
            word2vec = word2vec_back.copy(), 
            word_size = len(word2vec[0]), 
            max_length = 128, 
            pos_size = None, 
            hidden_size = config['hidden_size'], 
            dropout = 0, 
            bidirectional = True, 
            num_layers = 1, 
            config = config)
        model = proto_softmax_layer(
            sentence_encoder = encoder, 
            num_class = len(sampler.id2rel), 
            id2rel = sampler.id2rel, 
            drop = 0, 
            config = config)
        model.to(config["device"])
    # output the final avg result
    printer.output()

