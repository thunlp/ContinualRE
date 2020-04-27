## The code is based on the implementation from:
## https://github.com/hongwang600/RelationDectection
##

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    get_grad_params, copy_param_data
from evaluate import evaluate_model
from config import CONFIG as conf

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
device = conf['device']
lr = conf['learning_rate']
p_lambda = conf['lambda']
loss_margin = conf['loss_margin']

def param_loss(model, means, fishers, p_lambda):
    grad_params = get_grad_params(model)
    loss = torch.tensor(0.0).to(device)
    for i, param in enumerate(grad_params):
        #print(fishers[i])
        #print(param.data)
        #print(means[i])
        #print(p_lambda*fishers[i]*(param.data-means[i])**2)
        loss += (p_lambda*fishers[i]*(param-means[i])**2).sum()
    #print('loss', loss)
    return loss

def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100, grad_means=[], grad_fishers=[], loss_margin=2.0):
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
            samples = training_data[i*batch_size:(i+1)*batch_size]
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
            loss = loss.sum()
            #loss.to(device)
            #print(loss)
            for i in range(len(grad_means)):
                grad_mean = grad_means[i]
                grad_fisher = grad_fishers[i]
                #print(param_loss(model, grad_mean, grad_fisher, p_lambda))
                loss += param_loss(model, grad_mean, grad_fisher,
                                   p_lambda).to('cpu')
            loss.backward()
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
