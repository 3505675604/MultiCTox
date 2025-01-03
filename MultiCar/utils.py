import torch
import random
import numpy as np
from rdkit import Chem
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
atom_type_max = 100
atom_f_dim = 133
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], }


def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def onek_encoding_unk(key, length):
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1
    return encoding


def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
              onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
              onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
              onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
              onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
              onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
              [1 if atom.GetIsAromatic() else 0] + \
              [atom.GetMass() * 0.01]
    return feature


class tokens_struct():
    def __init__(self):
        self.tokens = [' ', '<unk>', 'C', 'O', '(', ')', 'c', '=', '1', '2', 'N', '3', 'n', 'P', '4', '[', ']', 'S',
                       'H', '5', 'l', '-', '*', 'o', '+', '6', '#', 'M', 'F', 'g', '7', 'B', 'r', 's', 'I', 'e', 'i',
                       '8', 'Z']
        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def unk(self):
        return self.tokens_vocab['<unk>']

    @property
    def pad(self):
        return self.tokens_vocab[' ']

    def get_default_tokens(self):
        """Default SMILES tokens"""
        return self.tokens

    def get_tokens_length(self):
        """Default SMILES tokens length"""
        return self.tokens_length

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            if char not in self.tokens:
                smiles_matrix[i] = self.unk
            else:
                smiles_matrix[i] = self.tokens_vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            chars.append(self.reversed_tokens_vocab[i])
        smiles = "".join(chars)
        return smiles

    def pad_sequence(self, sentence, sen_len=140, pad_index=0):
        # 将每个sentence变成一样的长度
        if len(sentence) > sen_len:
            sentence_tensor = torch.FloatTensor(sentence[:sen_len])
        else:
            sentence_tensor = torch.ones(sen_len) * pad_index
            sentence_tensor[:len(sentence)] = torch.FloatTensor(sentence)
        assert len(sentence_tensor) == sen_len
        return sentence_tensor


def accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    counts = []
    for i in range(y_true.shape[1]):
        p = sum((y_true[:, i] ^ y_pred[:, i]))
        counts.append(p)
    # weight = torch.nn.Softmax()(torch.Tensor(counts))
    weight = torch.Tensor(counts)
    q = torch.ones((y_true.shape[0], 1))
    class_weight = q.matmul(weight.unsqueeze(0))
    return count / y_true.shape[0], class_weight

def evaluate(label, output):
    zs = torch.sigmoid(output).to('cpu').data.numpy()
    
    ts = label.to('cpu').data.numpy().astype(int)
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))
    preds = np.array(preds).astype(int)
    acc = accuracy_score(ts, preds)
    f1score = f1_score(ts, preds)
    try:
        auc=roc_auc_score(ts, preds) ## y_true=ground_truth
    except ValueError:
        auc=0
        pass
    sensitivity = recall_score(ts, preds)#sn
    precision = precision_score(ts, preds)
    conf_matrix = confusion_matrix(ts, preds)

    # 从混淆矩阵中提取TN和FP
    if conf_matrix.shape == (2, 2):
        TN, FP = conf_matrix[0, 0], conf_matrix[0, 1]
    else:   
        TN, FP = 1,0


    # 计算特异性
    specificity = TN / (TN + FP)#sp
    mcc = matthews_corrcoef(ts, preds)

    return acc, f1score, sensitivity,specificity, (specificity+sensitivity)/2,precision#auc=mcc


def evaluate_class_wise(label, output):
    zs = torch.sigmoid(output).to('cpu').data.numpy()
    ts = label.to('cpu').data.numpy().astype(int)
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))
    preds = np.array(preds).astype(int)

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []

    for c in range(11):
        y_true = ts[:, c]
        y_pred = preds[:, c]

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)

        print('Class ' + str(c + 1) + ' Binary Classification Statistics:')

        print('Accuracy(Binary) %.4f, '
              'Precision(Binary) %.4f, '
              'Recall(Binary) %.4f, '
              'F1-Score(Binary) %.4f\n'
              % (acc,
                 precision,
                 recall,
                 f1score))

        acc_list.append(acc)
        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1score)
    return [acc_list, pre_list, rec_list, f1_list]


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    from matplotlib.patches import PathPatch
    # iterating through Axes instances
    for ax in g[1]:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
