import os
import pickle
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel, GCNModel_2
from opt import Optimizer
from sklearn.model_selection import KFold

# original code for GCN
def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    # tf.reset_default_graph()
    # tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1],
            association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

from tensorflow.keras.optimizers import Adam
"""Modified code from TF 1.x to TF 2.x"""
# Define a function to train a GCN and predict the score of drug-disease association
def PredictScore_2(train_drug_dis_matrix, drug_matrix, dis_matrix, drug_emb, dis_emb, 
                   seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    adj_norm = preprocess_graph(adj)
    adj_ = tf.SparseTensor(*adj_norm)
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))

    # construct a GCN model instance 
    model = GCNModel_2(num_features=features[2][1], emb_dim=emb_dim, 
                     features_nonzero=features[1].shape[0], adj_nonzero=adj.nonzero()[0].shape[0], 
                     adj=adj_, adjdp=adjdp, dp=dp,
                     num_r=train_drug_dis_matrix.shape[0])

    # compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')

    # convert the input data to tensor
    features_input = tf.SparseTensor(*features)

    drug_emb = tf.convert_to_tensor(drug_emb, dtype=tf.float32)
    dis_emb = tf.convert_to_tensor(dis_emb, dtype=tf.float32)

    for epoch in range(epochs):
        # 训练模型
        with tf.GradientTape() as tape:
            reconstructions, embeddings, LLM_embeddings = model((features_input, drug_emb, dis_emb), training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(train_drug_dis_matrix.flatten(), reconstructions))

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.numpy()))

    print('Optimization Finished!')

    # 生成预测结果
    reconstructions, embeddings, LLM_embeddings = model((features_input, drug_emb, dis_emb), training=False)
    return reconstructions


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    ##############################################
    # df = pd.read_csv('/home/gyw/DDA_prediction/data_1pos_1neg.csv')
    # df = df[df['Label'] == 1].values[:, :-1]
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    # kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    # for train_idx, test_idx in kfold.split(df):
    #     random_index.append(df[train_idx].tolist())
    tr, pr = [], []

    # tr_index = np.load(f'../data/B-dataset/train_{seed}.npy')
    # te_index = np.load(f'../data/B-dataset/test_{seed}.npy')
    pred_matrix = np.zeros(drug_dis_matrix.shape)
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        # train_matrix[te_index] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
        test_index = np.where(train_matrix == 0)
        metric_tmp, true, pred = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)
        pred_matrix[test_index] = predict_y_proba[test_index]
        # np.save('../result/kang_t_pred_{}.npy'.format(seed), predict_y_proba[te_index])
        # np.save('../result/kang_t_label_{}.npy'.format(seed), drug_dis_matrix[te_index])
        # return
        pred_matrix[tuple(np.array(random_index[k]).T)] = predict_y_proba[tuple(np.array(random_index[k]).T)]
        tr.append(drug_dis_matrix[test_index])
        pr.append(predict_y_proba[test_index])

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    # AUPR AUC F1-Score Accuracy Recall Specificity Precision
    metric = np.array(metric / k_folds)
    return metric, pred_matrix


def cross_validation_experiment_2(drug_dis_matrix, drug_matrix, dis_matrix, drug_emb, dis_emb,
                                 seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    ##############################################
    # df = pd.read_csv('/home/gyw/DDA_prediction/data_1pos_1neg.csv')
    # df = df[df['Label'] == 1].values[:, :-1]
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    # kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    # for train_idx, test_idx in kfold.split(df):
    #     random_index.append(df[train_idx].tolist())
    tr, pr = [], []

    # tr_index = np.load(f'../data/B-dataset/train_{seed}.npy')
    # te_index = np.load(f'../data/B-dataset/test_{seed}.npy')
    pred_matrix = np.zeros(drug_dis_matrix.shape)
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        # train_matrix[te_index] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore_2(
            train_matrix, drug_matrix, dis_matrix, drug_emb, dis_emb, 
            seed, epochs, emb_dim, dp, lr, adjdp)
        predict_y_proba = tf.reshape(drug_disease_res, [drug_len, dis_len])
        predict_y_proba = predict_y_proba.numpy()
        test_index = np.where(train_matrix == 0)
        metric_tmp, true, pred = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)
        pred_matrix[test_index] = predict_y_proba[test_index]
        # np.save('../result/kang_t_pred_{}.npy'.format(seed), predict_y_proba[te_index])
        # np.save('../result/kang_t_label_{}.npy'.format(seed), drug_dis_matrix[te_index])
        # return
        pred_matrix[tuple(np.array(random_index[k]).T)] = predict_y_proba[tuple(np.array(random_index[k]).T)]
        tr.append(drug_dis_matrix[test_index])
        pr.append(predict_y_proba[test_index])

        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    # AUPR AUC F1-Score Accuracy Recall Specificity Precision
    metric = np.array(metric / k_folds)
    return metric, pred_matrix

if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(dataset='B-dataset', save_dir='result_LLM') # to save the result for vanilla GCN, to 'result_origin'
    save_path = f'../result/{args.save_dir}/{args.dataset}'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    drug_sim = np.loadtxt(f'../data/{args.dataset}/drug_sim.csv', delimiter=',')
    print(drug_sim.shape)
    dis_sim = np.loadtxt(f'../data/{args.dataset}/dis_sim.csv', delimiter=',')
    print(dis_sim.shape)
    drug_dis_matrix = np.loadtxt(f'../data/{args.dataset}/drug_dis.csv', delimiter=',')
    print(drug_dis_matrix.shape)
    epoch = 4000
    emb_dim = 64
    lr = 0.008
    adjdp = 0.6
    dp = 0.4
    simw = 6
    results = np.zeros((10, 7), float)
    average_result = np.zeros((10, 11), float)
    circle_time = 10

    drug = pickle.load(open('../feat/drug_embeddings.pkl', 'rb'))
    dis = pickle.load(open('../feat/disease_embeddings.pkl', 'rb'))
    drug_emb = np.array([i.squeeze().numpy() for i in drug['embeddings'].values])
    dis_emb = np.array([i.squeeze().numpy() for i in dis['embeddings'].values])

    for i in range(10):
        result, pred_matrix = cross_validation_experiment_2(
            drug_dis_matrix, drug_sim * simw, dis_sim * simw, drug_emb, dis_emb, i, epoch, emb_dim, dp, lr, adjdp)
        results[i] = result

        np.save(f'{save_path}/{args.dataset}_{i}.npy', pred_matrix)