import pandas as pd
import torch
import dgl
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from utils import tokens_struct
from pubchemfp import GetPubChemFPs
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.data import DataLoader, Batch
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint, \
    CalculateECFP2Fingerprint
import pickle
from keras.preprocessing import text
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
def collate(data_list):
    batch = Batch.from_data_list(data_list)

    smiles_str = [data.smiles for data in data_list]

    labels =  torch.tensor([data.y for data in data_list]).float()
    seq_len = [i.token.size for i in data_list]
    padded_smiles_batch = pad_sequence([torch.tensor(i.token) for i in data_list], batch_first=True)
    fps_t = torch.FloatTensor([data.fp for data in data_list])
    batch.labels=labels
    batch.seq_len=seq_len
    batch.padded_smiles_batch=padded_smiles_batch
    batch.fps_t=fps_t
    batch.smiles_str=smiles_str
    
    return batch


def dataprocess(path,mulu):
    data = pd.read_csv(path)
    smiles = data['SMILES'].to_list()
    labels = np.where(data['pIC50'] >= 5, 1, 0)
    # token = tokens_struct()
    # node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    # edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    data_list=[]
    token = tokens_struct()
    for i,smiles in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(smiles)


        #coords = generate_3d_coordinates(mol)


        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 167
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
        fp_pubcfp = GetPubChemFPs(mol)  # 881
        fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        fp.extend(fp_ecfp2)
        c_size, features, edge_index = smile_to_graph(smiles)
        smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smiles, 100, 100)
        token1 = np.array(smi_embedding_matrix)
        # token1 = token.encode(smiles)
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        data1 = Data(x=torch.Tensor(features),
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            fp=fp,
                            token=token1,
                            y=torch.FloatTensor([labels[i]]),
                            smiles=smiles,
                            )
        data1.__setitem__('c_size', torch.LongTensor([c_size]))


        data_list.append(data1)
    with open(mulu, 'wb') as f:
        pickle.dump(data_list, f)
        # data, slices = self.collate(data_list)
    return data_list



def generate_3d_coordinates(mol):

    mol = Chem.AddHs(mol)

    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        raise ValueError("3D coordinate generation failed.")

    AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    coords = torch.tensor(coords, dtype=torch.float)
    
    return coords


def smile_w2v_pad(smile, maxlen_,victor_size):

    tokenizer = text.Tokenizer(num_words=100, lower=False,filters="　")
    tokenizer.fit_on_texts(smile)
    smile_ = pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_)
    word_index = tokenizer.word_index
    smileVec_model = {}
    with open("/root/autodl-tmp/MultiCBlo-master/Atom.vec", encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    count=0
    embedding_matrix = np.zeros((100, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    return smile_, word_index, embedding_matrix

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    if edge_index==[]:
        edge_index.append([0,1])

    # # Sequence encoder
    # smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smile, 100, 100)
    # smi_em = np.array(smi_embedding_matrix)

    return c_size, features, edge_index
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr','Pt','Hg','Pb','Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,'other']) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
if __name__ == "__main__":
    smile_to_graph('[Cl-].[Li+]')