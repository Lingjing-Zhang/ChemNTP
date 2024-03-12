
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import torch
import os
from mol_featurizer import *
import random
num_atom_feat = 34
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
from model import *
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
device = torch.device('cpu')
encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model = Predictor(encoder, decoder, device)
model.load_state_dict(torch.load("D:/Chemnpt/pbde/BDE/result/0.0001,64,0.1,0.0001,3",map_location = lambda storage,loc:storage))
model.to(device)
def predict_result(smiles,sequence):

    smiles = str(smiles)
    sequence = str(sequence)
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
            correct_labels,predicted_labels,predicted_scores = model(data, train=False)
        #predicted_labels,predicted_scores,attention = model(dataset, train=False)
    return predicted_labels,predicted_scores


smiles="COc1cc(C(=O)OCCCN(C)CCN(C)CCCOC(=O)c2cc(OC)c(OC)c(OC)c2)cc(OC)c1OC"
sequence="MEQTVLVPPGPDSFNFFTRESLAAIERRIAEEKAKNPKPDKKDDDENGPKPNSDLEAGKNLPFIYGDIPPEMVSEPLEDLDPYYINKKTFIVLNKGKAIFRFSATSALYILTPFNPLRKIAIKILVHSLFSMLIMCTILTNCVFMTMSNPPDWTKNVEYTFTGIYTFESLIKIIARGFCLEDFTFLRDPWNWLDFTVITFAYVTEFVDLGNVSALRTFRVLRALKTISVIPGLKTIVGALIQSVKKLSDVMILTVFCLSVFALIGLQLFMGNLRNKCIQWPPTNASLEEHSIEKNITVNYNGTLINETVFEFDWKSYIQDSRYHYFLEGFLDALLCGNSSDAGQCPEGYMCVKAGRNPNYGYTSFDTFSWAFLSLFRLMTQDFWENLYQLTLRAAGKTYMIFFVLVIFLGSFYLINLILAVVAMAYEEQNQATLEEAEQKEAEFQQMIEQLKKQQEAAQQAATATASEHSREPSAAGRLSDSSSEASKLSSKSAKERRNRRKKRKQKEQSGGEEKDEDEFQKSESEDSIRRKGFRFSIEGNRLTYEKRYSSPHQSLLSIRGSLFSPRRNSRTSLFSFRGRAKDVGSENDFADDEHSTFEDNESRRDSLFVPRRHGERRNSNLSQTSRSSRMLAVFPANGKMHSTVDCNGVVSLVGGPSVPTSPVGQLLPEVIIDKPATDDNGTTTETEMRKRRSSSFHVSMDFLEDPSQRQRAMSIASILTNTVEELEESRQKCPPCWYKFSNIFLIWDCSPYWLKVKHVVNLVVMDPFVDLAITICIVLNTLFMAMEHYPMTDHFNNVLTVGNLVFTGIFTAEMFLKIIAMDPYYYFQEGWNIFDGFIVTLSLVELGLANVEGLSVLRSFRLLRVFKLAKSWPTLNMLIKIIGNSVGALGNLTLVLAIIVFIFAVVGMQLFGKSYKDCVCKIASDCQLPRWHMNDFFHSFLIVFRVLCGEWIETMWDCMEVAGQAMCLTVFMMVMVIGNLVVLNLFLALLLSSFSADNLAATDDDNEMNNLQIAVDRMHKGVAYVKRKIYEFIQQSFIRKQKILDEIKPLDDLNNKKDSCMSNHTAEIGKDLDYLKDVNGTTSGIGTGSSVEKYIIDESDYMSFINNPSLTVTVPIAVGESDFENLNTEDFSSESDLEESKEKLNESSSSSEGSTVDIGAPVEEQPVVEPEETLEPEACFTEGCVQRFKCCQINVEEGRGKQWWNLRRTCFRIVEHNWFETFIVFMILLSSGALAFEDIYIDQRKTIKTMLEYADKVFTYIFILEMLLKWVAYGYQTYFTNAWCWLDFLIVDVSLVSLTANALGYSELGAIKSLRTLRALRPLRALSRFEGMRVVVNALLGAIPSIMNVLLVCLIFWLIFSIMGVNLFAGKFYHCINTTTGDRFDIEDVNNHTDCLKLIERNETARWKNVKVNFDNVGFGYLSLLQVATFKGWMDIMYAAVDSRNVELQPKYEESLYMYLYFVIFIIFGSFFTLNLFIGVIIDNFNQQKKKFGGQDIFMTEEQKKYYNAMKKLGSKKPQKPIPRPGNKFQGMVFDFVTRQVFDISIMILICLNMVTMMVETDDQSEYVTTILSRINLVFIVLFTGECVLKLISLRHYYFTIGWNIFDFVVVILSIVGMFLAELIEKYFVSPTLFRVIRLARIGRILRLIKGAKGIRTLLFALMMSLPALFNIGLLLFLVMFIYAIFGMSNFAYVKREVGIDDMFNFETFGNSMICLFQITTSAGWDGLLAPILNSKPPDCDPNKVNPGSSVKGDCGNPSVGIFFFVSYIIISFLVVVNMYIAVILENFSVATEESAEPLSEDDFEMFYEVWEKFDPDATQFMEFEKLSQFAAALEPPLNLPQPNKLQLIAMDLPMVSGDRIHCLDILFAFTKRVLGESGEMDALRIQMEERFMASNPSKVSYQPITTTLKRKQEEVSAVIIQRAYRRHLLKRTVKQASFTYNKNKIKGGANLLIKEDMIIDRINENSITEKTDLTMSTAACPPSYDRVTKPIVEKHEQEGKDEKAKGK"
predicted_labels,predicted_scores=predict_result(smiles,sequence)
print(predicted_labels,predicted_scores)