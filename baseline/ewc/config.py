import torch
CONFIG= {
    'learning_rate': 0.001,
    'embedding_dim': 300,
    'hidden_dim': 200,
    'batch_size': 50,
    'num_clusters': 20,
    'epoch': 3,
    'lambda': 1,
    'random_seed': 100,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'model_path': 'model.pt',
    'device': torch.device('cuda:4' if torch.cuda.is_available() else 'cpu'),
    'relation_file': './data/relation.2M.list',
    'training_file': './data/train.replace_ne.withpool',
    'test_file': './data/test.replace_ne.withpool',
    'valid_file': './data/valid.replace_ne.withpool',
    'glove_file': './data/glove.6B.300d.txt'
}
