from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO
from rdkit import Chem
from rdkit.Chem import Descriptors
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import numpy as np
import random
import os
import time
from model import *
from mol_featurizer import *
import timeit
import pickle
import pandas as pd

device = torch.device('cuda:0')

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def to_dataset(data):
    model1 = Word2Vec.load("word2vec_30.model")
    dir_input = ('E:/model_final/')
    os.makedirs(dir_input, exist_ok=True)
    data1 =pd.read_csv(dir_input + data +'.csv',index_col=0,header=None, encoding='utf-8')
    data_list = data1.index.tolist()
    N = len(data_list)
    compounds, adjacencies,proteins,interactions = [], [], [], []
    for no, i in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequence, interaction = i.strip().split(" ")
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol.GetNumAtoms()
        except:
            continue
        atom_feature, adj = mol_features(smiles)
        protein_embedding = get_protein_embedding(model1, seq_to_kmers(sequence))
        label = np.array(interaction,dtype=np.float32)

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        protein = torch.FloatTensor(protein_embedding)
        label = torch.LongTensor(label)

        compounds.append(atom_feature)
        adjacencies.append(adj)
        proteins.append(protein)
        interactions.append(label)
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    
    return dataset



"""Load preprocessed data."""

with open("E:/model_final/dataset_dev2.0.txt","rb") as f:
    data = pickle.load(f)
#with open("E:/model_final/dataset_test.txt","rb") as f1:
    #data1 = pickle.load(f1)
dataset_dev = data
#dataset_test = data1
dataset_train="dataset_train2.0"
dataset=to_dataset(dataset_train)
dataset_train = dataset
print(len(dataset_train))
print(len(dataset_dev))
#print(len(dataset_test))

""" create model ,trainer and tester """
protein_dim = 100
atom_dim = 34
hid_dim = 64
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1
batch = 64
lr = 1e-4
weight_decay = 1e-4
iteration = 200
kernel_size = 9

"""Output files."""

dir_output_1=("E:/model_final/output/result")
dir_output_2=("E:/model_final/output/model")
os.makedirs(dir_output_1, exist_ok=True)
os.makedirs(dir_output_2, exist_ok=True)
file_AUCs = 'E:/model_final/output/result/AUCs_optimization2.0'+ '.txt'
#file_model = 'E:/model_final/output/model/' + 'AUCs_optimization'
AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev')
with open(file_AUCs, 'w') as f:
    f.write(AUCs + '\n')
paramss=('lr\tbatch\tdropout\tweight_decay\tn_layers\tbest_epoch\tbest_AUC')
params_file = 'E:/model_final/output/result/BayesianOptimization2.0'+ '.txt'
with open(params_file,'w') as f:
    f.write(paramss + '\n')


def f(lr,batch,dropout,weight_decay,n_layers, direction = True):
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    """Start training."""
    print('Training...')
    print(AUCs)
    batch=int(round(batch))
    n_layers=int(round(n_layers))
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load("E:/model_final/AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,lookaheadradam,ealy_stop.pt"),strict=False)
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)
    

    start = timeit.default_timer()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
    best_param ={}
    best_param["train_epoch"] = 0
    best_param["dev_epoch"] = 0
    best_param["train_loss"] = 100000
    best_param["dev_AUC"] = 0.5
    #param=[lr,batch,dropout,weight_decay,n_layers,kernel_size]

    for epoch in range(1, iteration + 1):
        loss_train = trainer.train(dataset_train, device)
        #train_loss= validation.valid(dataset_train)
        #dev_loss = validation.valid(dataset_dev)
        #AUC_train, PRC_train = tester.test(dataset_train)
        AUC_dev, PRC_dev = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start

        AUC = [epoch, time, loss_train, AUC_dev, PRC_dev]
        scheduler.step()
        tester.save_AUCs(AUC, file_AUCs)
        print('\t'.join(map(str, AUC)))
        if best_param["train_loss"] > loss_train:
            best_param["train_epoch"] = epoch
            best_param["train_loss"] = loss_train
        if AUC_dev > best_param["dev_AUC"]:
            best_param["dev_epoch"] = epoch
            best_param["dev_AUC"] = AUC_dev
            file_model='E:/model_final/output/model/' + str(lr)+str(",")+str(batch)+str(",")+str(dropout)+str(",")+str(weight_decay)+str(",")+str(n_layers)
            tester.save_model(model, file_model)
        if (epoch - best_param["train_epoch"] >4) or (epoch - best_param["dev_epoch"] >5):
            #file_model='E:/model_final/output/model/' + str(lr)+str(",")+str(batch)+str(",")+str(dropout)+str(",")+str(weight_decay)+str(",")+str(n_layers)
            #tester.save_model(model, file_model)
            #break
            result=[best_param["dev_epoch"],best_param["dev_AUC"]]
            param=[lr,batch,dropout,weight_decay,n_layers,best_param["dev_epoch"],best_param["dev_AUC"]]

            with open(params_file, 'a') as f:
                f.write('\t'.join(map(str, param))+'\n')
            break

    # GPGO maximize performance by default, set performance to its negative value for minimization
    if direction:
        return best_param["dev_AUC"]
    else:
        return -best_param["dev_AUC"]


from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode='UCB')
params = {'lr':     ('cont', [1e-4, 1e-2]),
          'batch':  ('int', [64, 256]),
          'dropout':('cont', [0, 0.3]),
          #'kernel_size':('int', [5,7,9,11,13]),
          #'kernel_size':(range(5,14,2)),
          'weight_decay':('cont', [1e-6, 1e-2]),
          'n_layers':('int', [1, 6])
         }
np.random.seed(20)
gpgo = GPGO(gp, acq, f, params)
gpgo.run(max_iter=30,init_evals=2)
#gpgo.run(max_iter=30)
gpgo.getResult()
