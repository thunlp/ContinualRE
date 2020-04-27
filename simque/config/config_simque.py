import json

CONFIG= {
    'learning_rate': 0.001,
    'embedding_dim': 300,
    'hidden_size': 300,
    'batch_size': 50,
    'gradient_accumulation_steps':1,
    'num_clusters': 20,
    'epoch': 2,
    'random_seed': 100,
    'task_memory_size': 10,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'num_cands': 10,
    'num_steps': 1,
    'num_constrain': 10,
    'data_per_constrain': 10,
    'lr_alignment_model': 0.0001,
    'use_gpu': True,
    'relation_file': './data/simpleqa/relation_names.txt',
    'training_file': './data/simpleqa/training_files.txt',
    'test_file': './data/simpleqa/test_files.txt',
    'valid_file': './data/simpleqa/valid_files.txt',
    # 'relation_file': './data/simpleqa/relation.2M.list',
    # 'training_file': './data/simpleqa/train.replace_ne.withpool',
    # 'test_file': './data/simpleqa/test.replace_ne.withpool',
    # 'valid_file': './data/simpleqa/valid.replace_ne.withpool',
    'task_name': 'SimQue',
    'num_workers':4,
    'max_grad_norm':1
}

f = open("config_simque_10.json", "w")
f.write(json.dumps(CONFIG))
f.close()