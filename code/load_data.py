import dgl
import torch as th
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import calc_pairwise_cosine_similarity

def load_dataset(args):
    """Load the heterogeneous network of drug-disease association dataset.
    """

    dr_dr = pd.read_csv(f'../data/{args.dataset}/drug_sim.csv', header=None).values
    dr_sim = dr_dr
    for i in range(len(dr_dr)):
        sorted_idx = np.argpartition(dr_sim[i], 15)
        dr_dr[i, sorted_idx[-15:]] = 1
    dr_dr = pd.DataFrame(np.array(np.where(dr_dr == 1)).T, columns=['Drug1', 'Drug2'])
    di_di = pd.read_csv(f'../data/{args.dataset}/dis_sim.csv', header=None).values
    di_sim = di_di
    for i in range(len(di_di)):
        sorted_idx = np.argpartition(di_sim[i], 15)
        di_di[i, sorted_idx[-15:]] = 1
    di_di = pd.DataFrame(np.array(np.where(di_di == 1)).T, columns=['Disease1', 'Disease2'])
    dr_di = pd.read_csv(f'../data/{args.dataset}/drug_dis.csv', header=None)
    dr_di = pd.DataFrame(np.array(np.where(dr_di == 1)).T, columns=['Drug', 'Disease'])
    graph_data = {
        ('drug', 'dr_dr', 'drug'): (th.tensor(dr_dr['Drug1'].values),
                                        th.tensor(dr_dr['Drug2'].values)),
        ('disease', 'di_di', 'disease'): (th.tensor(di_di['Disease1'].values),
                                                    th.tensor(di_di['Disease2'].values)),
        ('drug', 'dr_di', 'disease'): (th.tensor(dr_di['Drug'].values),
                                              th.tensor(dr_di['Disease'].values)),
        ('disease', 'di_dr', 'drug'): (th.tensor(dr_di['Disease'].values),
                                              th.tensor(dr_di['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    dr_feat = np.hstack((dr_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    di_feat = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), di_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(dr_feat).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(di_feat).to(th.float32)

    if args.BERT_emb:
        with open(f'../feat/{args.dataset}/BERT_drug_emb.pkl', 'rb') as f:
            dr_feat = th.tensor(list(pickle.load(f).values())).to(th.float32)
        with open(f'../feat/{args.dataset}/BERT_disease_emb.pkl', 'rb') as f:
            di_feat = th.tensor(list(pickle.load(f).values())).to(th.float32)
    if args.LLM_emb:
        with open(f'../feat/{args.dataset}/LLM_drug_emb.pkl', 'rb') as f:
            dr_feat = th.tensor(list(pickle.load(f).values())).to(th.float32)
        with open(f'../feat/{args.dataset}/LLM_disease_emb.pkl', 'rb') as f:
            di_feat = th.tensor(list(pickle.load(f).values())).to(th.float32)
    dr_sim = calc_pairwise_cosine_similarity(dr_feat)
    disease_sim = calc_pairwise_cosine_similarity(di_feat)
    dr_dr, di_di = dr_sim, disease_sim
    for i in range(len(dr_sim)):
        sorted_idx = np.argpartition(dr_sim[i], 15)
        dr_dr[i, sorted_idx[-15:]] = 1
    for i in range(len(disease_sim)):
        sorted_idx = np.argpartition(disease_sim[i], 15)
        di_di[i, sorted_idx[-15:]] = 1
    dr_dr = pd.DataFrame(np.array(np.where(dr_dr == 1)).T, columns=['Drug1', 'Drug2'])
    di_di = pd.DataFrame(np.array(np.where(di_di == 1)).T, columns=['Disease1', 'Disease2'])
    graph_data = {
        ('drug', 'dr_dr', 'drug'): (th.tensor(dr_dr['Drug1'].values),
                                        th.tensor(dr_dr['Drug2'].values)),
        ('disease', 'di_di', 'disease'): (th.tensor(di_di['Disease1'].values),
                                                    th.tensor(di_di['Disease2'].values)),
        ('drug', 'dr_di', 'disease'): (th.tensor(dr_di['Drug'].values),
                                              th.tensor(dr_di['Disease'].values)),
        ('disease', 'di_dr', 'drug'): (th.tensor(dr_di['Disease'].values),
                                              th.tensor(dr_di['Drug'].values)),
    }
    g_llm = dgl.heterograph(graph_data)
    dr_feat = np.hstack((dr_sim, np.zeros((g_llm.num_nodes('drug'), g_llm.num_nodes('disease')))))
    di_feat = np.hstack((np.zeros((g_llm.num_nodes('disease'), g_llm.num_nodes('drug'))), disease_sim))
    g_llm.nodes['drug'].data['h'] = th.from_numpy(dr_feat).to(th.float32)
    g_llm.nodes['disease'].data['h'] = th.from_numpy(di_feat).to(th.float32)
    return g, g_llm
        

def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id),
                          th.tensor(test_dis_id),
                          etype=('drug', 'dr_di', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'dr_di', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id),
                          th.tensor(test_drug_id),
                          etype=('disease', 'di_dr', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'di_dr', 'drug'))
    return g


def generate_feat(args, g, g_llm=None):
    """Generate the node features for the heterogeneous network.
    """
    # convert protein-disease pairs to disease-protein association matrix
    # BERT feature dimension: 768; LLM feature dimension: 1536
    if isinstance(g, list):
        g_llm = g[1]
        g = g[0]
    if args.dr_fingerprint:
        drug = pd.read_csv(f'../data/{args.dataset}/drug.csv')
        mol = [Chem.MolFromSmiles(smile) for smile in drug['SMILES'].values]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mol]
        g.nodes['drug'].data['fingerprint'] = th.tensor(fps).to(th.float32).to(args.device)
        dr_feat = th.cat([g.nodes['drug'].data['h'], g.nodes['drug'].data['fingerprint']], dim=1)
    else:
        dr_feat = g.nodes['drug'].data['h']
    if args.dis_prot_assoc and args.dataset == 'Bdataset':
        # protein-disease association matrix
        disaese = pd.read_csv(f'../data/{args.dataset}/disease.csv')
        dis_prot = pd.read_csv(f'../data/{args.dataset}/protein_disease.csv')
        num_prot = dis_prot['Protein'].max()
        num_dis = dis_prot['Disease'].max()
        dis_prot_matrix = np.zeros((num_dis+1, num_prot+1))
        dis_prot_matrix[dis_prot['Disease'], dis_prot['Protein']] = 1
        g.nodes['disease'].data['dis_prot_assoc'] = th.from_numpy(dis_prot_matrix).to(th.float32).to(args.device)
        dis_feat = th.cat([g.nodes['disease'].data['h'], g.nodes['disease'].data['dis_prot_assoc']], dim=1)
    else:
        dis_feat = g.nodes['disease'].data['h']
    
    if args.BERT_emb or args.LLM_emb:
        drug_LLM_emb = g_llm.nodes['drug'].data['h']
        disease_LLM_emb = g_llm.nodes['disease'].data['h']

    if args.concatenate_type == 'as_node':
        return {'drug': th.cat([dr_feat, drug_LLM_emb], dim=1),
                'disease': th.cat([dis_feat, disease_LLM_emb], dim=1)}
    elif args.concatenate_type == 'none':
        return {'drug': dr_feat, 'disease': dis_feat}
    else:
        # save the features
        np.save(f'../feat/{args.dataset}/drug_sim_feat.npy', dr_feat.cpu().numpy())
        np.save(f'../feat/{args.dataset}/disease_sim_feat.npy', dis_feat.cpu().numpy())
        np.save(f'../feat/{args.dataset}/drug_LLM_emb.npy', drug_LLM_emb.cpu().numpy())
        np.save(f'../feat/{args.dataset}/disease_LLM_emb.npy', disease_LLM_emb.cpu().numpy())
        return {'drug': dr_feat, 'disease': dis_feat,
                'drug_LLM': drug_LLM_emb, 'disease_LLM': disease_LLM_emb}