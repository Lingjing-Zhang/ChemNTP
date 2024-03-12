import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mol_featurizer import *
num_atom_feat = 34
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import os
device = torch.device('cpu')

psmi=['Oc1c(Br)ccc(Oc2ccc(Br)cc2Br)c1Br','COc1c(Br)ccc(Oc2ccc(Br)cc2Br)c1Br','Oc1cc(Oc2ccc(Br)cc2Br)c(Br)cc1Br','Oc1cc(Br)cc(Br)c1Oc1ccc(Br)cc1Br','COc1cc(Br)cc(Br)c1Oc1ccc(Br)cc1Br','Oc1cc(Oc2cc(Br)c(Br)cc2Br)c(Br)cc1Br','Oc1cc(Br)cc(Br)c1Oc1cc(Br)c(Br)cc1Br','COc1cc(Br)cc(Br)c1Oc1cc(Br)c(Br)cc1Br']
sequence='MRPPQCLLHTPSLASPLLLLLLWLLGGGVGAEGREDAELLVTVRGGRLRGIRLKTPGGPVSAFLGIPFAEPPMGPRRFLPPEPKQPWSGVVDATTFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPTSPTPVLVWIYGGGFYSGASSLDVYDGRFLVQAERTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGMGEARRRATQLAHLVGCPPGGTGGNDTELVACLRTRPAQVLVNHEWHVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAEFLAGVRVGVPQVSDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYVFEHRASTLSWPLWMGVPHGYEIEFIFGIPLDPSRNYTAEEKIFAQRLMRYWANFARTGDPNEPRDPKAPQWPPYTAGAQQYVSLDLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL'
smiles=psmi[7]
print(smiles)
model1 = Word2Vec.load("word2vec_30.model")
compounds, adjacencies,proteins,interactions = [], [], [], []

atom_feature, adj = mol_features(smiles)
protein_embedding = get_protein_embedding(model1, seq_to_kmers(sequence))
label = np.array([1.0],dtype=np.float32)

atom_feature = torch.FloatTensor(atom_feature)
adj = torch.FloatTensor(adj)
protein = torch.FloatTensor(protein_embedding)
label = torch.LongTensor(label)

compounds.append(atom_feature)
adjacencies.append(adj)
proteins.append(protein)
interactions.append(label)
dataset = list(zip(compounds, adjacencies, proteins, interactions))
from model_c2_protein import *

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
iteration = 60
kernel_size = 9

encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Predictor(encoder, decoder, device)
model.load_state_dict(torch.load("0.0001,64,0.1,0.0001,3",map_location = lambda storage,loc:storage))
model.to(device)

with torch.no_grad():
    model.eval()
    
    for data in dataset:
        adjs, atoms, proteins, labels = [], [], [], []
        atom, adj, protein, label = data
        adjs.append(adj)
        atoms.append(atom)
        proteins.append(protein)
        labels.append(label)
        data = pack(atoms,adjs,proteins, labels, device)
        correct_labels,predicted_labels,predicted_scores,attention = model(data, train=False)
    
    #predicted_labels,predicted_scores,attention = model(dataset, train=False)
#print(predicted_labels)
#print(predicted_scores,attention)
#print(attention.shape)
print(predicted_scores)
attention = attention.squeeze(0)

# attention = attention.numpy()
# # list = [11,16,1,21,4,17,9,22,12]
# sum = np.zeros(attention.shape[1])
# for i in list:
#     k = attention[i,:]
#     sum += k

sum = torch.sum(attention,dim=0)
sum = sum.numpy()
#print(sum)

sum = np.sum(sum,0)
print(sum.shape)
#print(sum)
num=sum.shape[0]
print(num)

arsimilarity=np.argsort(sum[:])
#print(arsimilarity)

#pval
#sum=sum.tolist()
pval=[]
for i in sum:
    s=0
    for j in sum:
        if j>=i:
            s+=1
    p=s/int(num)
    pval.append(p)


#bde7
x1=[73,82,85,86,119,120,123,124,125,132,201,296,336,337,340,446,447,448,450]
y1=[]
for i in x1:
    a=pval[i]
    y1.append(a)

print(y1)
#[0.032679738562091505, 0.3611111111111111, 0.3562091503267974, 0.7124183006535948, 0.47058823529411764, 0.25163398692810457, 0.8088235294117647, 0.5702614379084967, 0.369281045751634, 0.803921568627451, 0.5833333333333334, 0.6486928104575164, 0.315359477124183, 0.8578431372549019, 0.4068627450980392, 0.6535947712418301, 0.2696078431372549, 0.9934640522875817, 0.1715686274509804]
x1=[73,82,85,86,119,120,123,124,125,132,201,296,336,337,340,446,447,448,450]
y1=[]
for i in x1:
    a=sum[i]
    y1.append(a)
print(y1)
#[0.66542447, 0.20579138, 0.2075496, 0.12299925, 0.1739986, 0.2598726, 0.10329618, 0.14979734, 0.20247996, 0.1041951, 0.14729235, 0.1381174, 0.22982827, 0.08983725, 0.19212608, 0.13687098, 0.24795064, 0.036423083, 0.30895382]
