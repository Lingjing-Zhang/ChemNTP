
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency,dtype=np.float32)


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def to_txt(data):
    model = Word2Vec.load("word2vec_30.model")
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
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
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
    with open(dir_input + data +'.txt', "wb") as f:
        pickle.dump(dataset, f)
    return f




if __name__ == "__main__":

    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    import os
    import torch
    import pickle
    """
    DATASET = "data_train"
    
    data =pd.read_csv("E:/final_data/input_data_final.csv",index_col=0,header=None, encoding='utf-8')
    data_list = data.index.tolist()

    N = len(data_list)

    compounds, adjacencies,proteins,interactions = [], [], [], []
    model = Word2Vec.load("word2vec_30.model")
    dir_input = ('E:/model_final/data_train_')
    os.makedirs(dir_input, exist_ok=True)
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequence, interaction = data.strip().split(" ")
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol.GetNumAtoms()
        except:
            continue
        atom_feature, adj = mol_features(smiles)
        compounds.append(atom_feature)
        adjacencies.append(adj)
        interactions.append(np.array([float(interaction)]))
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
        proteins.append(protein_embedding)
        if float(no) % 10000 == 0:
            xx = no/10000
            np.save(dir_input + '/compounds_%s'%xx, compounds,fix_imports=True)
            compounds = []
            np.save(dir_input + '/adjacencies_%s'%xx, adjacencies,fix_imports=True)
            adjacencies = []
            np.save(dir_input + '/interactions_%s'%xx, interactions,fix_imports=True)
            interactions = []
            np.save(dir_input + '/proteins_%s'%xx, proteins,fix_imports=True)
            proteins = []
        else:
            pass
    #proteins=np.array(proteins,dtype="object")
    np.save(dir_input + '/compounds_%s'%(xx+1), compounds,fix_imports=True)
    np.save(dir_input + '/adjacencies_%s'%(xx+1), adjacencies,fix_imports=True)
    np.save(dir_input + '/interactions_%s'%(xx+1), interactions,fix_imports=True)
    np.save(dir_input + '/proteins_%s'%(xx+1), proteins,fix_imports=True)
    print('The preprocess of ' + DATASET + ' dataset has finished!')
    """
    #dataset_test="dataset_test2.0"
    #dataset_dev="dataset_dev2.0"
    #dataset_train="dataset_train"
    #data1=to_txt(dataset_test)
    #data2=to_txt(dataset_dev)
    #data3=to_txt(dataset_train)