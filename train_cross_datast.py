import os
import random
import logging
import numpy as np
from scipy import stats

import torch
import torch.optim

from configs import Configs
from models.model import VIPNet
from data_loader import DataLoader
from train import validate

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
os.makedirs('./results', exist_ok=True)

def main(config):

    train_data = DataLoader(config, config.train_dataset, path=dataset_path[config.train_dataset],
                            img_indx=dataset_split_img_num[config.train_dataset], patch_num=config.train_patch_num, istrain=True)
    train_loader = train_data.get_data()

    test_data1 = DataLoader(config, config.test_dataset1, path=dataset_path[config.test_dataset1],
                           img_indx=dataset_split_img_num[config.test_dataset1],patch_num=config.test_patch_num, istrain=False)
    test_loader1 = test_data1.get_data()

    test_data2 = DataLoader(config, config.test_dataset2, path=dataset_path[config.test_dataset2],
                           img_indx=dataset_split_img_num[config.test_dataset2],patch_num=config.test_patch_num, istrain=False)
    test_loader2 = test_data2.get_data()

    # Create an instance of VIPNet model
    model = VIPNet(config)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    # Define the loss function
    criterion = torch.nn.L1Loss().to(device)

    logger.info('Epoch\tTrain_Loss\tTrain_SROCC\tTest_on_{}_SROCC\tTest_on_{}_SROCC'.format(config.test_dataset1,config.test_dataset2))
    all_test_srocc1 = []
    all_test_plcc1 = []
    all_test_srocc2 = []
    all_test_plcc2 = []

    for epoch in range(config.start_epoch, config.epochs + 1):
        # train one epoch
        test_srocc1, test_plcc1, test_srocc2, test_plcc2 = train(train_loader,model, optimizer, criterion, epoch, config, test_loader1,test_loader2)
        lr_scheduler.step()
        all_test_srocc1.append(test_srocc1)
        all_test_plcc1.append(test_plcc1)
        all_test_srocc2.append(test_srocc2)
        all_test_plcc2.append(test_plcc2)
    test_srocc1, test_plcc1 = np.max(all_test_srocc1), np.max(all_test_plcc1)
    test_srocc2, test_plcc2 = np.max(all_test_srocc2), np.max(all_test_plcc2)
    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset1,
                                                                                  test_srocc1, test_plcc1))

    logger.info('Training on {} and test on {}. SROCC: {}, PLCC: {}'.format(config.train_dataset, config.test_dataset2,
                                                                          test_srocc2, test_plcc2))

    return test_srocc1, test_plcc1, test_srocc2, test_plcc2

def train(train_loader, model, optimizer,criterion, epoch, config,data_loader1,data_loader2):

    model.train()
    pred_scores, gt_scores, epoch_loss = [], [], []
    for i, (images, labels) in enumerate(train_loader):

        imgs_rgb = images[0].to(device)
        imgs_ycbcr = images[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs_rgb, imgs_ycbcr)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        pred_scores = pred_scores + preds.cpu().tolist()
        gt_scores = gt_scores + labels.cpu().tolist()

    train_srocc, _ = stats.spearmanr(pred_scores, gt_scores)
    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    test_srocc1, test_plcc1, _, _ = validate(config, model, data_loader1)
    test_srocc2, test_plcc2, _, _ = validate(config, model, data_loader2)
    test_srocc1, test_plcc1 = np.abs(test_srocc1), np.abs(test_plcc1)
    test_srocc2, test_plcc2 = np.abs(test_srocc2), np.abs(test_plcc2)
    logger.info('%d\t%6.5f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (epoch, sum(epoch_loss) / len(epoch_loss), train_srocc, test_srocc1, test_srocc2))

    return test_srocc1, test_plcc1, test_srocc2, test_plcc2


if __name__ == '__main__':
    dataset_path = {
        'live': 'D:\iqadataset\LIVE',
        'csiq': 'D:\iqadataset\CSIQ',
        'tid2013': 'D:\iqadataset\TID2013',
        'kadid-10k': 'D:\iqadataset\kadid10k',
        'livemd': 'D:\iqadataset\LIVEMD',
        'livec': 'D:\iqadataset\LIVEC',
        'koniq-10k': 'D:\iqadataset\koniq10k',
    }

    dataset_split_img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid-10k': list(range(0, 81)),

        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
    }
    config = Configs()
    # config.train_dataset = 'live'
    # config.test_dataset1 = 'csiq'
    # config.test_dataset2 = 'tid2013'

    logger.info('Experimental Configurations : %s ', config)
    # Setup logging
    logger.setLevel(level=logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./results/train_on_{}_test_on_{}_and_{}.txt'.format(config.train_dataset,
                                                                                        config.test_dataset1,config.test_dataset2),mode='a', encoding='utf-8')
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("***** Training on {}, Testing on {} and {} *****".format(config.train_dataset, config.test_dataset1, config.test_dataset2, config.epochs))

    all_cross_performance = np.zeros((config.train_test_round, 4), dtype=np.float32)

    for i in range(1, config.train_test_round+1):
        if config.seed == 0:
            pass
        else:
            logger.info('Using the seed = {} for {}-th experiment'.format(config.seed*i, i))
            torch.manual_seed(i* config.seed)
            torch.cuda.manual_seed(i * config.seed)
            np.random.seed(i * config.seed)
            random.seed(i * config.seed)

        all_cross_performance[i-1, :] = main(config)
    results_median = np.median(all_cross_performance, axis=0)
    logger.info('Testing on {} and Testing on {}'.format(config.test_dataset1, config.test_dataset2))
    logger.info('Results: {}'.format(results_median))

