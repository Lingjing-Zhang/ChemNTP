import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from mol_featurizer import *
num_atom_feat = 34

def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        #a_len = a_len.cuda()
        a = torch.eye(a_len)
        #a = a.cuda()
        #adj = adj.cuda()
        a = a
        adj = adj
        adj = adj + a
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import os

"""CPU or GPU"""
device = torch.device('cpu')

psmi=['Oc1c(Br)ccc(Oc2ccc(Br)cc2Br)c1Br','COc1c(Br)ccc(Oc2ccc(Br)cc2Br)c1Br','Oc1cc(Oc2ccc(Br)cc2Br)c(Br)cc1Br','Oc1cc(Br)cc(Br)c1Oc1ccc(Br)cc1Br','COc1cc(Br)cc(Br)c1Oc1ccc(Br)cc1Br','Oc1cc(Oc2cc(Br)c(Br)cc2Br)c(Br)cc1Br','Oc1cc(Br)cc(Br)c1Oc1cc(Br)c(Br)cc1Br','COc1cc(Br)cc(Br)c1Oc1cc(Br)c(Br)cc1Br']
sequence='MRPPQCLLHTPSLASPLLLLLLWLLGGGVGAEGREDAELLVTVRGGRLRGIRLKTPGGPVSAFLGIPFAEPPMGPRRFLPPEPKQPWSGVVDATTFQSVCYQYVDTLYPGFEGTEMWNPNRELSEDCLYLNVWTPYPRPTSPTPVLVWIYGGGFYSGASSLDVYDGRFLVQAERTVLVSMNYRVGAFGFLALPGSREAPGNVGLLDQRLALQWVQENVAAFGGDPTSVTLFGESAGAASVGMHLLSPPSRGLFHRAVLQSGAPNGPWATVGMGEARRRATQLAHLVGCPPGGTGGNDTELVACLRTRPAQVLVNHEWHVLPQESVFRFSFVPVVDGDFLSDTPEALINAGDFHGLQVLVGVVKDEGSYFLVYGAPGFSKDNESLISRAEFLAGVRVGVPQVSDLAAEAVVLHYTDWLHPEDPARLREALSDVVGDHNVVCPVAQLAGRLAAQGARVYAYVFEHRASTLSWPLWMGVPHGYEIEFIFGIPLDPSRNYTAEEKIFAQRLMRYWANFARTGDPNEPRDPKAPQWPPYTAGAQQYVSLDLRPLEVRRGLRAQACAFWNRFLPKLLSATDTLDEAERQWKAEFHRWSSYMVHWKNQFDHYSKQDRCSDL'
smiles=psmi[7]
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

from model_compound import *
#from model import *
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
#model.load_state_dict(torch.load("E:/model_final/output/model/best_AUCs_optimization",map_location = lambda storage,loc:storage))
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
        correct_labels, predicted_labels, predicted_scores,norm,trg,sum = model(data, train=False)
print(predicted_scores)
#print(sum)
#print(trg)

#calculate similarity
sum = sum.reshape(-1).numpy()
trg = trg.numpy()
similarity = np.zeros(trg.shape[0])
for i in range(trg.shape[0]):
    candidate = trg[i,:]
    similarity[i] = np.dot(candidate,sum)/(np.linalg.norm(candidate)*(np.linalg.norm(sum)))
print(similarity)
arsimilarity=np.argsort(similarity[:])
print(arsimilarity)



import matplotlib.pyplot as plt


from rdkit.Chem import PyMol
from rdkit import Chem
import sys
#from IPython.display import SVG
from rdkit import rdBase
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import DrawMorganBit, DrawMorganBits,DrawMorganEnv, IPythonConsole
mol_1=smiles
mol = Chem.MolFromSmiles(mol_1)
print(mol)

def add_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for i in range( atoms ):
        mol.GetAtomWithIdx(i).SetProp(
            'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
    return mol,atoms

from rdkit.Chem import Draw

mols = []
mol,atoms = add_atom_index(mol)
print(atoms)
print(mol)
img = Draw.MolToImage(mol,size=(1200, 1200))
#img.save("mol.jpg")
