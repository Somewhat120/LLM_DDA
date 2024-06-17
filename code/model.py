import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim=None, dropout=0.4):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights_D = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights_D.weight)
            self.weights_R = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights_R.weight)

    def forward(self, feature):
        feature['drug'] = self.dropout(feature['drug'])
        feature['disease'] = self.dropout(feature['disease'])
        R = feature['drug']
        D = feature['disease']
        R = self.weights_R(R)
        D = self.weights_D(D)
        outputs = R @ D.T
        return outputs


class Node_Encoder(nn.Module):
    """The base HeteroGCN layer."""

    def __init__(self, in_feats, out_feats, dropout, rel_names):
        super().__init__()
        HeteroGraphdict = {}
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, out_feats)
            nn.init.xavier_normal_(graphconv.weight)
            HeteroGraphdict[rel] = graphconv
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = dglnn.HeteroGraphConv(HeteroGraphdict, aggregate='sum')
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.prelu = nn.PReLU()

    def forward(self, graph, inputs, bn=False, dp=False):
        h = self.embedding(graph, inputs)
        if bn and dp:
            h = {k: self.prelu(self.dropout(self.bn_layer(v))) for k, v in h.items()}
        elif dp:
            h = {k: self.prelu(self.dropout(v)) for k, v in h.items()}
        elif bn:
            h = {k: self.prelu(self.bn_layer(v)) for k, v in h.items()}
        else:
            h = {k: self.prelu(v) for k, v in h.items()}
        return h


class SemanticAttention(nn.Module):
    """The base attention mechanism used in
    topological subnet embedding block and layer attention block.
    """

    def __init__(self, in_feats, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, is_print=False):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        if is_print:
            print(beta)
        return (beta * z).sum(1)


class Model(nn.Module):
    """The overall MODDA architecture."""

    def __init__(self, args, etypes, ntypes, in_feats):
        super(Model, self).__init__()
        self.ntypes = ntypes
        self.model_type = args.concatenate_type
        hidden_feats = args.hidden_feats
        dropout = args.dropout

        self.drug_linear = nn.Linear(in_feats[0], hidden_feats)
        nn.init.xavier_normal_(self.drug_linear.weight)
        self.disease_linear = nn.Linear(in_feats[1], hidden_feats)
        nn.init.xavier_normal_(self.disease_linear.weight)
        self.encoder_1 = Node_Encoder(hidden_feats, hidden_feats, dropout, etypes)
        self.encoder_2 = Node_Encoder(hidden_feats, hidden_feats, dropout, etypes)

        if self.model_type == 'graph_ae':
            self.drug_linear_LLM = nn.Linear(in_feats[2], hidden_feats)
            nn.init.xavier_normal_(self.drug_linear_LLM.weight)
            self.disease_linear_LLM = nn.Linear(in_feats[3], hidden_feats)
            nn.init.xavier_normal_(self.disease_linear_LLM.weight)
            self.encoder_LLM_1 = nn.Linear(hidden_feats, hidden_feats)
            nn.init.xavier_normal_(self.encoder_LLM_1.weight)
            self.encoder_LLM_2 = nn.Linear(hidden_feats, hidden_feats)
            nn.init.xavier_normal_(self.encoder_LLM_2.weight)

        if self.model_type in ['graph_graph', 'cross_graph']: 
            self.drug_linear_LLM = nn.Linear(in_feats[2], hidden_feats)
            nn.init.xavier_normal_(self.drug_linear_LLM.weight)
            self.disease_linear_LLM = nn.Linear(in_feats[3], hidden_feats)
            nn.init.xavier_normal_(self.disease_linear_LLM.weight)
            self.encoder_LLM_1 = Node_Encoder(hidden_feats, hidden_feats, dropout, etypes)
            self.encoder_LLM_2 = Node_Encoder(hidden_feats, hidden_feats, dropout, etypes)

        self.layer_attention_layer_drug = SemanticAttention(hidden_feats)
        self.layer_attention_layer_dis = SemanticAttention(hidden_feats)

        self.layer_attention_layer_dis_LLM = SemanticAttention(hidden_feats)
        self.layer_attention_layer_drug_LLM = SemanticAttention(hidden_feats)

        self.predict = InnerProductDecoder(hidden_feats)

    def forward(self, g, x):
        if isinstance(g, list):
            g_llm = g[1]
            g = g[0]
        dr_emb, di_emb = [], []
        dr_LLM_emb, di_LLM_emb = [], []
        if self.model_type in ['none', 'as_node']:
            h = {'drug': x['drug'], 'disease': x['disease']}
        else:
            h = {'drug': x['drug'], 'disease': x['disease']}
            h_llm = {'drug': x['drug_LLM'], 'disease': x['disease_LLM']}
        for ntype in self.ntypes:
            h[ntype] = x[ntype]
        h['drug'] = self.drug_linear(h['drug'])
        h['disease'] = self.disease_linear(h['disease'])
        dr_emb.append(h['drug'])
        di_emb.append(h['disease'])

        h = self.encoder_1(g, h, bn=True, dp=True)
        dr_emb.append(h['drug'])
        di_emb.append(h['disease'])
        h = self.encoder_2(g, h, bn=True, dp=True)
        dr_emb.append(h['drug'])
        di_emb.append(h['disease'])

        if self.model_type in ['graph_graph', 'cross_graph']:
            h_llm['drug'] = self.drug_linear_LLM(h_llm['drug'])
            h_llm['disease'] = self.disease_linear_LLM(h_llm['disease'])
            dr_LLM_emb.append(h['drug'])
            di_LLM_emb.append(h['disease'])
            h_llm = self.encoder_LLM_1(g_llm if self.model_type == 'graph_graph' else g_llm,
                                       h_llm, bn=True, dp=True)
            dr_LLM_emb.append(h_llm['drug'])
            di_LLM_emb.append(h_llm['disease'])
            h_llm = self.encoder_LLM_2(g_llm if self.model_type == 'graph_graph' else g_llm,
                                       h_llm, bn=True, dp=True)
            dr_LLM_emb.append(h_llm['drug'])
            di_LLM_emb.append(h_llm['disease'])

            if self.model_type == 'cross_graph':
                h['drug'], h_llm['drug'] = dr_emb[0], dr_LLM_emb[0]
                h['disease'], h_llm['disease'] = di_emb[0], di_LLM_emb[0]
                h = self.encoder_1(g_llm, h, bn=True, dp=True)
                dr_emb.append(h['drug'])
                di_emb.append(h['disease'])
                h = self.encoder_2(g_llm, h, bn=True, dp=True)
                dr_emb.append(h['drug'])
                di_emb.append(h['disease'])

                h_llm = self.encoder_LLM_1(g, h_llm, bn=True, dp=True)
                dr_LLM_emb.append(h_llm['drug'])
                di_LLM_emb.append(h_llm['disease'])
                h_llm = self.encoder_LLM_2(g, h_llm, bn=True, dp=True)
                dr_LLM_emb.append(h_llm['drug'])
                di_LLM_emb.append(h_llm['disease'])
                
        elif self.model_type == 'graph_ae':
            h_llm['drug'] = self.drug_linear_LLM(h_llm['drug'])
            h_llm['disease'] = self.disease_linear_LLM(h_llm['disease'])
            dr_LLM_emb.append(h_llm['drug'])
            di_LLM_emb.append(h_llm['disease'])
            h_llm['drug'] = self.encoder_LLM_1(h_llm['drug'])
            h_llm['disease'] = self.encoder_LLM_1(h_llm['disease'])
            dr_LLM_emb.append(h['drug'])
            di_LLM_emb.append(h['disease'])
            h_llm['drug'] = self.encoder_LLM_2(h_llm['drug'])
            h_llm['disease'] = self.encoder_LLM_2(h_llm['disease'])
            dr_LLM_emb.append(h_llm['drug'])
            di_LLM_emb.append(h_llm['disease'])

        drug_emb_list = dr_emb + dr_LLM_emb
        dis_emb_list = di_emb + di_LLM_emb
        
        if self.model_type != 'none':
            h['drug'] = self.layer_attention_layer_drug(torch.stack(drug_emb_list, dim=1))
            h['disease'] = self.layer_attention_layer_dis(torch.stack(dis_emb_list, dim=1))

        return self.predict(h)
        