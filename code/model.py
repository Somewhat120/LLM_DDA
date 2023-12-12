import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from layers import GraphConvolution_2, GraphConvolutionSparse_2, InnerProductDecoder_2
from layers import AttentionAggregator
from utils import *


class GCNModel():

    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.num_r = num_r
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden1)

        self.emb = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)

        self.embeddings = self.hidden1 * \
            self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.embeddings)

from tensorflow.keras.models import Model
class GCNModel_2(Model):
    def __init__(self, num_features, emb_dim, features_nonzero, adj, adj_nonzero, adjdp, dp, num_r, act=tf.nn.elu):
        super(GCNModel_2, self).__init__()
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.act = act
        self.num_r = num_r
        # self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.adj = adj
        self.adjdp = adjdp
        self.dropout = dp

        # define layers
        self.gcn_sparse_layer1 = GraphConvolutionSparse_2(input_dim=num_features,
                                                        output_dim=self.emb_dim,
                                                        features_nonzero=self.features_nonzero,
                                                        act=self.act, 
                                                        adj=self.adj, 
                                                        dropout=self.dropout)
        self.gcn_dense_layer2 = GraphConvolution_2(input_dim=self.emb_dim,
                                                output_dim=self.emb_dim,
                                                act=self.act, 
                                                adj=self.adj, 
                                                dropout=self.dropout)
        self.gcn_dense_layer3 = GraphConvolution_2(input_dim=emb_dim,
                                                output_dim=self.emb_dim,
                                                act=self.act, 
                                                adj=self.adj, 
                                                dropout=self.dropout)
        self.layer_attn1 = AttentionAggregator(num_vectors=3)
        
        self.gcn_dense_layer4 = GraphConvolution_2(input_dim=768,
                                                    output_dim=self.emb_dim,
                                                    act=self.act, 
                                                    adj=self.adj, 
                                                    dropout=self.dropout)
        self.gcn_dense_layer5 = GraphConvolution_2(input_dim=self.emb_dim,
                                                output_dim=self.emb_dim,
                                                act=self.act, 
                                                adj=self.adj, 
                                                dropout=self.dropout)
        self.gcn_dense_layer6 = GraphConvolution_2(input_dim=self.emb_dim,
                                                output_dim=self.emb_dim,
                                                act=self.act, 
                                                adj=self.adj, 
                                                dropout=self.dropout)
        # self.layer_attn2 = AttentionAggregator(num_vectors=3)
        
        self.decoder = InnerProductDecoder_2(input_dim=self.emb_dim,
                                             num_r=self.num_r,
                                             act=tf.nn.sigmoid,
                                             dropout=self.dropout)
        self.feat_attn_drug = AttentionAggregator(num_vectors=2)
        self.feat_attn_dis = AttentionAggregator(num_vectors=2)

    # main contribution: add new module 
    def call(self, inputs, training=False):
        # X is similarity matrix of drugs and diseases
        x, drug_emb, dis_emb = inputs
        # layer 1: sparse GCN layer 
        hidden1 = self.gcn_sparse_layer1(x, training=training)
        # layer 2: dense GCN layer with embedding of drug and disease similarity
        hidden2 = self.gcn_dense_layer2(hidden1, training=training)
        # layer 3: dense GCN layer with embedding of drug and disease similarity
        emb = self.gcn_dense_layer3(hidden2, training=training)
        # layer 4: attention layer
        embeddings = self.layer_attn1(tf.stack([hidden1, hidden2, emb], axis=0))
        # the original attention did not update weights during training
        # embeddings = hidden1 * self.att[0] + hidden2 * self.att[1] + emb * self.att[2] 
    
        """Add a new feature channel with input from embeddings generated by LLM and BioBert"""  
        # concatenate embeddings of drug and disease
        LLM_x = tf.concat([drug_emb, dis_emb], axis=0)
        # layer 5: dense GCN layer with embedding
        LLMhidden1 = self.gcn_dense_layer4(LLM_x, training=training)
        # layer 6: dense GCN layer with embedding
        LLMhidden2 = self.gcn_dense_layer5(LLMhidden1, training=training)
        # layer 7: dense GCN layer with embedding
        LLMemb = self.gcn_dense_layer6(LLMhidden2, training=training)
        # layer 8: attention layer
        LLM_embeddings = self.layer_attn1(tf.stack([LLMhidden1, LLMhidden2, LLMemb], axis=0))
        # the original attention did not update weights during training
        # LLM_embeddings = LLMhidden1 * self.att[0] + LLMhidden2 * self.att[1] + LLMemb * self.att[2]

        # embeddings of drug and disease from two different channels
        drug_1, drug_2 = embeddings[0:self.num_r, :], LLM_embeddings[0:self.num_r, :] 
        dis_1, dis_2 = embeddings[self.num_r:, :], LLM_embeddings[self.num_r:, :]

        # layer 9: aggregate embeddings from two channels
        drug_emb = self.feat_attn_drug(tf.stack([drug_1, drug_2], axis=0))
        dis_emb = self.feat_attn_dis(tf.stack([dis_1, dis_2], axis=0))
        
        # concatenate embeddings of drug and disease
        embeddings = tf.concat([drug_emb, dis_emb], axis=0)

        # layer 10: decoder
        reconstructions = self.decoder(embeddings, training=training)

        return reconstructions, embeddings, LLM_embeddings

