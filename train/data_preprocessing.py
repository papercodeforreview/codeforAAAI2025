import json
from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)

sim_matrix = np.zeros([1,1])

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def generate_drug_data(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).transpose(-1, 0)

    new_edge_index = edge_list.transpose(-1, 0)

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats)

    return data


def generate_drug_data_GMPNN(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T

    return features, new_edge_index, edge_feats, line_graph_edge_index





def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    s1, s2 = add_space(s1), add_space(s2)  # 在字中间加上空格
    cv = CountVectorizer(tokenizer=lambda s: s.split())  # 转化为TF矩阵
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()  # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def load_drug_mol_data(args):
    global sim_matrix
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}
    relation_dict = {}
    hash_dict = {}
    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1],
                                                    data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2
        if id1 not in relation_dict:
            relation_dict[id1] = []
        relation_dict[id1].extend([id2])

    sim_matrix = np.ones([len(drug_smile_dict), len(drug_smile_dict)])
    if os.path.exists(f'{args.dirname}/{args.dataset}/matrix.npy'):
        sim_matrix = np.load(f'{args.dirname}/{args.dataset}/matrix.npy')
        with open(f'{args.dirname}/{args.dataset}/ID_dict.json', 'r') as f:
            ID_list_dict = json.load(f)
    else:
        ID_list_dict = {}
        item = 0
        with tqdm(total=len(drug_smile_dict)) as pbar:
            for k, v in drug_smile_dict.items():
                pbar.update(1)
                ID_list_dict[k] = [item]
                item_in = item
                kk = 0
                for k_in, v_in in drug_smile_dict.items():
                    if kk < item_in + 1:
                       kk += 1
                       continue
                    kk = 1000000
                    item_in += 1
                    sim_value = tf_similarity(v, v_in)
                    if (k not in relation_dict) or (k_in not in relation_dict[k]):
                        sim_matrix[item, item_in] = sim_value
                    if (k_in not in relation_dict) or (k not in relation_dict[k_in]):
                        sim_matrix[item_in, item] = sim_value
                item += 1
        with open(f'{args.dirname}/{args.dataset}/ID_dict.json', 'w') as f:
            json.dump(ID_list_dict, f)
        np.save(f'{args.dirname}/{args.dataset}/matrix.npy', sim_matrix)
    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    save_data(drug_data, 'drug_data.pkl', args)
    return drug_data, drug_smile_dict, ID_list_dict


def generate_pair_triplets(args, ds_dict=None, ID_dic = None):
    pos_triplets = []
    drug_ids = []
    num_ID_list = []
    for _ in ID_dic.keys():
        num_ID_list.append(_)
    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)

    if ds_dict == None:
        for id1, id2, s1, s2 in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1], data[args.c_s2]):
            ds_dict[id1] = s1
            ds_dict[id2] = s2

    for id1, id2, relation, s1, s2 in zip(data[args.c_id1], data[args.c_id2], data[args.c_y], data[args.c_s1],
                                          data[args.c_s2]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset is 1-based index, need to substract by 1
        if args.dataset in ('drugbank',):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)
    n_neg_tail = 0
    n_neg_head = 0
    temp_neg_h = []
    temp_neg_t = []
    neg_samples = []
    all_samples_dict = {
        'DB1': [],
        'DB2': [],
        'smile1': [],
        'smile2': [],
        'label': []
    }
    label_list = [1]
    for i in range(args.neg_ent):
        label_list.extend([0])

    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]
        if args.dataset != 'D':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, ID_dic, num_ID_list, args)
            n_neg_tail += len(neg_tails)
            n_neg_head += len(neg_heads)
            all_samples_dict['label'].extend(label_list)
            data_statistics['ALL_NOT_NEG_H_WITH_T'][h] = np.concatenate(
                [data_statistics['ALL_NOT_NEG_H_WITH_T'][h],
                 neg_tails], axis=0)
            data_statistics['ALL_NOT_NEG_H_WITH_T'][h] = np.concatenate(
                [data_statistics['ALL_NOT_NEG_T_WITH_H'][t],
                 neg_heads], axis=0)
            if len(neg_heads) > 0:
                temp_neg_h.extend(["".join(str(neg_h + ',') for neg_h in neg_heads)[:-1]])
            else:
                temp_neg_h.extend([0])
            if len(neg_tails) > 0:
                temp_neg_t.extend(["".join(str(neg_t + ',') for neg_t in neg_tails)[:-1]])
            else:
                temp_neg_t.extend([0])
            all_samples_dict['DB1'].append(h)
            all_samples_dict['DB2'].append(t)
            all_samples_dict['smile1'].append(ds_dict[h])
            all_samples_dict['smile2'].append(ds_dict[t])
            nh = neg_heads.size
            negs = np.concatenate([neg_heads, neg_tails], axis=0)
            for i in range(args.neg_ent):
                if nh:
                    all_samples_dict['DB1'].append(negs[i])
                    all_samples_dict['DB2'].append(t)
                    all_samples_dict['smile1'].append(ds_dict[negs[i]])
                    all_samples_dict['smile2'].append(ds_dict[t])
                    nh -= 1
                else:
                    all_samples_dict['DB1'].append(h)
                    all_samples_dict['DB2'].append(negs[i])
                    all_samples_dict['smile1'].append(ds_dict[h])
                    all_samples_dict['smile2'].append(ds_dict[negs[i]])

        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate(
                    [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                    axis=0)
            )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)

        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

    print(n_neg_head)
    print(n_neg_tail)
    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0],
                       'Drug2_ID': pos_triplets[:, 1],
                       'Y': pos_triplets[:, 2],
                       'Neg_samples_h': temp_neg_h,
                       'Neg_samples_t': temp_neg_t})

    df2 = pd.DataFrame({'DB_1': all_samples_dict['DB1'],
                        'DB_2': all_samples_dict['DB2'],
                        'smile_1': all_samples_dict['smile1'],
                        'smile_2': all_samples_dict['smile2'],
                        'label': all_samples_dict['label']})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    filename2 = f'{args.dirname}/{args.dataset}/Our_all.csv'
    df.to_csv(filename, index=False)
    df2.to_csv(filename2, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    # HT、TH关系记录
    statistics["ALL_NOT_NEG_H_WITH_T"] = defaultdict(dict)
    statistics["ALL_NOT_NEG_T_WITH_H"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1
        # 负例拒绝数组，用于标记不能形成样例对应负例的样本
        statistics["ALL_NOT_NEG_H_WITH_T"][h][t] = 1
        statistics["ALL_NOT_NEG_T_WITH_H"][t][h] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])
    for h in statistics["ALL_NOT_NEG_H_WITH_T"].keys():
        statistics["ALL_NOT_NEG_H_WITH_T"][h] = np.array(list(statistics["ALL_NOT_NEG_H_WITH_T"][h].keys()))
    for t in statistics["ALL_NOT_NEG_T_WITH_H"]:
        statistics["ALL_NOT_NEG_T_WITH_H"][t] = np.array(list(statistics["ALL_NOT_NEG_T_WITH_H"][t].keys()))

    print('getting data statistics done!')

    return statistics


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def all_data_gen():
    return 0


def split_data(args):
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    test_size_ratio = args.test_ratio
    n_folds = args.n_folds
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size_ratio, random_state=seed)
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['ZhangDDI', 'drugbank', 'twosides'],
                        help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, required=True,
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'drugbig': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'drugbig1': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'drugbig2': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'drugbig_test': ('DB_1', 'DB_2', 'smile_1', 'smile_2', 'label'),
        'drugbig_valid': ('DB_1', 'DB_2', 'smile_1', 'smile_2', 'label'),
        'drugbig_train': ('DB_1', 'DB_2', 'smile_1', 'smile_2', 'label'),
        'ZhangDDI': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'DeepDDI': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'ChChMiner': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/drugbank.csv', ','),
        'drugbig': ('data/drugbig.csv', ','),
        'drugbig1': ('data/drugbig1.csv', ','),
        'drugbig2': ('data/drugbig2.csv', ','),
        'drugbig_test': ('data/drugbig_test.csv', ','),
        'drugbig_valid': ('data/drugbig_valid.csv', ','),
        'drugbig_train': ('data/drugbig_train.csv', ','),
        'ZhangDDI': ('data/ZhangDDI.csv', ','),
        'DeepDDI': ('data/DeepDDI.csv', ','),
        'ChChMiner': ('data/ChChMiner.csv', ','),
        'twosides': ('data/twosides_ge_500_1.zip', ',')
    }
    args = parser.parse_args()
    args.gen_mode = 'dis'
    args.dataset = args.dataset.lower()
    args.dataset = 'drugbig'
    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data/preprocessed'

    args.random_num_gen = np.random.RandomState(args.seed)
    ds_dict = None
    if args.operation in ('all', 'drug_data'):
        _, ds_dict, dic = load_drug_mol_data(args)

    if args.operation in ('all', 'generate_triplets'):
        generate_pair_triplets(args, ds_dict, dic)

    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)
