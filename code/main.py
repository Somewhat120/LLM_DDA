import os
import logging
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from load_data import load_dataset, remove_graph, generate_feat
from utils import define_logging, get_metrics_auc, set_seed, plot_result_auc,\
    plot_result_aupr, EarlyStopping, get_metrics
from args import args


def train():
    set_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    # if os.path.exists(os.path.join(args.saved_path, 'result.csv')):
    #     # this means the result for the current setting has been saved, no need to re-run
    #     return
    simplefilter(action='ignore', category=FutureWarning)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    define_logging(args, logger)
    logger.info(args)
    
    if args.device_id:
        logger.info('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        logger.info('Training on CPU')
        device = th.device('cpu')
    args.device = device

    # load DDA data for Kfold splitting
    df = pd.read_csv('../data/{}/drug_dis.csv'.format(args.dataset),
                      header=None).values
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, 
                         random_state=args.seed)
    fold = 1
    pred_result = np.zeros(df.shape)

    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        logger.info('{}-Cross Validation: Fold {}'.format(args.nfold, fold))
        
        # get the index list for train and test set
        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]
        assert len(test_pos_idx[0]) + len(test_neg_idx[0]) + len(train_pos_idx[0]) + len(train_neg_idx[0]) == len(data)
        
        g = load_dataset(args)
        logger.info(g)
        # remove test set DDA from train graph
        g, g_llm = g[0], g[1]
        g = remove_graph(g, test_pos_id).to(args.device)
        g_llm = remove_graph(g_llm, test_pos_id).to(args.device)
        
        # generate features based on model type
        feature = generate_feat(args, [g, g_llm])    

        # get the mask list for train and test set that used for performance calculation
        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
        mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
        mask_test = np.where(mask_label == 0)
        mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
        mask_train = np.where(mask_label == 1)
        mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]

        logger.info('Number of total training samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_train[0]),
                                                                                              len(train_pos_idx[0]),
                                                                                              len(train_neg_idx[0])))
        logger.info('Number of total testing samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_test[0]),
                                                                                             len(test_pos_idx[0]),
                                                                                             len(test_neg_idx[0])))
        assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])
        label = th.tensor(df).float().to(args.device)
        
        # load model and optimizer
        if args.concatenate_type in ['none', 'as_node']:
            model = Model(args=args,
                          etypes=g.etypes, ntypes=g.ntypes,
                          in_feats=[feature['drug'].shape[1],
                                    feature['disease'].shape[1]])
        else:
            model = Model(args=args,
                          etypes=g.etypes, ntypes=g.ntypes,
                          in_feats=[feature['drug'].shape[1],
                                    feature['disease'].shape[1],
                                    feature['drug_LLM'].shape[1],
                                    feature['disease_LLM'].shape[1]])
        model.to(args.device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)
        criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        # no pos weight
        # criterion = th.nn.BCEWithLogitsLoss()
        logger.info('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        # stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)
        
        # model training
        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model([g, g_llm], feature)
            pred = th.sigmoid(score)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC_, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                      pred[mask_train].cpu().detach().numpy())
            # early_stop = stopper.step(loss.item(), AUC_, model)

            if epoch % 50 == 0:
                AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
                                            pred[mask_test].cpu().detach().numpy())
                logger.info('Epoch {} Loss: {:.3f}; Train AUC {:.3f}; AUC {:.3f}; AUPR: {:.3f}'.format(epoch, loss.item(),
                                                                                                 AUC_, AUC, AUPR))
                # print('-' * 50)
                # if early_stop:
                #     break
            
        # stopper.load_checkpoint(model)
        model.eval()
        pred = th.sigmoid(model([g, g_llm], feature)).cpu().detach().numpy()
        test_pos_idx, test_neg_idx = np.array(test_pos_idx), np.array(test_neg_idx)
        pred_result[test_pos_idx[0], test_pos_idx[1]] = pred[test_pos_idx[0], test_pos_idx[1]]
        pred_result[test_neg_idx[0], test_neg_idx[1]] = pred[test_neg_idx[0], test_neg_idx[1]]
        # save the model
        th.save(model.state_dict(), os.path.join(args.saved_path, 'model_fold_{}.pth'.format(fold)))
        fold += 1

    # save the result
    AUC, aupr, acc, f1, pre, rec, spec = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    logger.info(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}'.
            format(AUC, aupr, acc, f1, pre, rec))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path,
                                                  'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)

if __name__ == '__main__':
    train()