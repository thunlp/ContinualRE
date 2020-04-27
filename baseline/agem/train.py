## The code is based on the implementation from:
## https://github.com/hongwang600/RelationDectection
##

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    copy_grad_data, get_grad_params
from evaluate import evaluate_model
from config import CONFIG as conf

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
num_cands = conf['num_cands']
device = conf['device']
lr = conf['learning_rate']
loss_margin = conf['loss_margin']

def sample_memory_data(sample_pool, sample_size):
    if len(sample_pool) > 0:
        sample_indexs = random.sample(range(len(sample_pool)),
                                      min(sample_size, len(sample_pool)))
        return [sample_pool[index] for index in sample_indexs]
    else:
        return []

def feed_samples(model, samples, loss_function, all_relations, device):
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

    model.zero_grad()
    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       reverse_question_indexs, reverse_relation_indexs,
                       question_lengths, relation_lengths)
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    start_index = 0
    for length in relation_set_lengths:
        pos_scores.append(all_scores[start_index].expand(length-1))
        neg_scores.append(all_scores[start_index+1:start_index+length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths)-
                                    len(relation_set_lengths)))
    loss.backward()

# copied from facebook open scource. (https://github.com/facebookresearch/
# GradientEpisodicMemory/blob/master/model/gem.py)
def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().view(-1).double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    x = gradient_np - (np.dot(gradient_np, memories_np)/
                       np.dot(memories_np, memories_np)) * memories_np
    gradient.copy_(torch.Tensor(x).view(-1))

# copied from facebook open scource. (https://github.com/facebookresearch/
# GradientEpisodicMemory/blob/master/model/gem.py)
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def get_grads_memory_data(model, memory_data, loss_function,
                          all_relations, device):
    if len(memory_data) == 0:
        return []
    memory_data_grads = []
    memory_data_set = [memory_data]
    for data in memory_data_set:
        #print(data)
        feed_samples(model, data, loss_function, all_relations, device)
        memory_data_grads.append(copy_grad_data(model))
        #print(memory_data_grads[-1][:10])
    if len(memory_data_grads) > 1:
        return torch.stack(memory_data_grads)
    elif len(memory_data_grads) == 1:
        return memory_data_grads[0].view(1,-1)
    else:
        return []

def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100, all_seen_samples=[],
          task_memory_size=100, loss_margin=0.5, all_seen_rels=[]):
    if model is None:
        torch.manual_seed(100)
        model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                                np.array(embedding), 1, device)
    loss_function = nn.MarginRankingLoss(loss_margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for epoch_i in range(epoch):
        #print('epoch', epoch_i)
        #training_data = training_data[0:100]
        for i in range((len(training_data)-1)//batch_size+1):
            memory_data = sample_memory_data(all_seen_samples, task_memory_size)
            for this_sample in memory_data:
                rel_cands = [rel for rel in all_seen_rels if rel!=this_sample[0]]
                this_sample[1] = random.sample(rel_cands,
                                               min(len(rel_cands),num_cands))
            memory_data_grads = get_grads_memory_data(model, memory_data,
                                                      loss_function,
                                                      all_relations,
                                                      device)
            #print(memory_data_grads)
            samples = training_data[i*batch_size:(i+1)*batch_size]
            feed_samples(model, samples, loss_function, all_relations, device)
            sample_grad = copy_grad_data(model)
            if len(memory_data_grads) > 0:
                if torch.matmul(memory_data_grads,
                                torch.t(sample_grad.view(1,-1))) < 0:
                    project2cone2(sample_grad, memory_data_grads)
                    grad_params = get_grad_params(model)
                    grad_dims = [param.data.numel() for param in grad_params]
                    overwrite_grad(grad_params, sample_grad, grad_dims)
            optimizer.step()
        '''
        acc=evaluate_model(model, valid_data, batch_size, all_relations, device)
        if acc > best_acc:
            torch.save(model, model_path)
    best_model = torch.load(model_path)
    return best_model
    '''
    return model

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100)
    #print(training_data[0:10])
    #print(testing_data[0:10])
    #print(valid_data[0:10])
    #print(all_relations[0:10])
