import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default=None, type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-da', '--dataset', type=str,
                    choices=['Bdataset', 'Cdataset', 'Fdataset', 'Rdataset'],
                    help='Set the data set for training.')
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='result')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed')
# Training Arguments
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=1000, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=100, type=int,
                    help='Early Stopping argument')
# Model Arguments
parser.add_argument('-ft', '--feature_type', default='concat', type=str,
                    choices = ['BERT', 'LLM'],
                    help='The type of feature used in the model')
parser.add_argument('-ct', '--concatenate_type', default='graph_graph', type=str,
                    choices = ['graph_graph', 'graph_ae', 'cross_graph', 'as_node', 'none'],
                    help='The type of concatenation in the model')
parser.add_argument('-hf', '--hidden_feats', default=128, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-dp', '--dropout', default=0.0, type=float,
                    help='The rate of dropout layer')

args = parser.parse_args()
args.saved_path = os.path.join('../result',
                    args.dataset+'_'+args.concatenate_type+'_'+args.feature_type+'_'+str(args.epoch) \
                    +'_'+str(args.dropout)+'_'+str(args.hidden_feats),
                    str(args.seed))
args.dr_fingerprint = True
args.dis_prot_assoc = True
if args.feature_type == 'BERT':
    args.BERT_emb, args.LLM_emb = True, False
elif args.feature_type == 'LLM':
    args.BERT_emb, args.LLM_emb = False, True