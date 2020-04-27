# ContinualRE
--



Codes and datasets for our paper "Continual Relation Learning via Episodic Memory Activation and Reconsolidation"

If you use the code, please cite the following paper:

```
 @inproceedings{han2018neural,
   title={Continual Relation Learning via Episodic Memory Activation and Reconsolidation},
   author={Han, Xu and Dai, Yi and Gao, Tianyu and Lin, Yankai and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
   booktitle={Proceedings of ACL},
   year={2020}
 }
```

Requirements
==========

The model is implemented using PyTorch. The versions of packages used are shown below.


*	numpy==1.18.0

*	scikit-learn==0.22.1

*	scipy==1.4.1

*	torch==1.3.0

*	tqdm==4.41.1



Baselines
==========

The main experimental settings come from the project [Lifelong Relation Detection](https://github.com/hongwang600/ Lifelong_Relation_Detection).

We adapt some typical lifelong learning methods for continual relation learning, including EMR, AGEM and EWC. The code of these models can be found in the folder "./baseline/".



Datasets
==========

We provide all the datasets and word embeddings used in our experiments.

+ [[Download (datasets)]](https://cloud.tsinghua.edu.cn/f/75578dfc8d974cd98c58/?dl=1)
+ [[Download (word embeddings)]](https://cloud.tsinghua.edu.cn/f/75578dfc8d974cd98c58/?dl=1)


Run the experiments
==========

####(0) To run the experiments, unpack the datasets and word embeddings first

```
unzip data.zip -d data/
unzip glove.zip -d glove/
```

####(1) For FewRel

```
cp -r data/ fewrel/
cp -r glove/ fewrel/
cd fewrel
python run_multi_proto.py
```

####(2) For SimpleQuestions

```
cp -r data/ simque/
cp -r glove/ simque/
cd simque
python run_multi_proto.py
```

####(3) For TACRED

```
cp -r data/ tacred/
cp -r glove/ tacred/
cd tacred
python run_multi_proto.py
```

####(4) For some special settings

All the config files can be found in "./fewrel/config/", "./tacred/config/", and  "./simque/config/". By changing the config file name in the code "run\_multi\_proto.py", we can run experiments with different settings. In "./fewrel/config/", "./tacred/config/", and  "./simque/config/", we also provide code to generate customized settings.

