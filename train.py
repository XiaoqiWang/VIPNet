import os
import random
import logging
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import torch
import torch.optim

from configs import Configs
from models.model import VIPNet
from data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
os.makedirs('./results', exist_ok=True)

def main(config):

    best_srocc, best_plcc, best_epoch = 0, 0, 0

    # define data loaders
    train_data = DataLoader(config, dataset=config.dataset, path=config.path, img_indx=config.train_index, patch_num=config.train_patch_num, istrain=True)
    train_loader = train_data.get_data()
    test_data = DataLoader(config, dataset=config.dataset, path=config.path, img_indx=config.test_index, patch_num=config.test_patch_num, istrain=False)
    test_loader = test_data.get_data()

    # Create an instance of VIPNet model
    model = VIPNet(config)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
    # Define the loss function
    criterion = torch.nn.L1Loss().to(device)

    # Training loop
    logger.info('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTest_SRCC\tTest_PLCC')
    for epoch in range(config.start_epoch, config.epochs + 1):
        # train one epoch
        test_srocc, test_plcc, all_preds, all_labels = train(train_loader, test_loader, model, optimizer, criterion, epoch, config)
        lr_scheduler.step()
        # save the best results
        if best_srocc < test_srocc:
            best_srocc = test_srocc
            best_plcc = test_plcc
            # Save the trained model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_criterion': best_srocc,
                'optimizer': optimizer.state_dict(),
            }, config)
            best_epoch = epoch
            # Evaluating distortion type performance on TID2013 and KADID-10k Datasets.
            if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
                dist_type_num = 24 if config.dataset == 'tid2013' else 25
                type_results = evaluate_distortion_type_performance(len(config.test_index), all_preds, all_labels, dist_type_num)

        # Early Stopping for over-fitting
        if (epoch > (best_epoch + 10)) and (test_srocc < best_srocc) and epoch > int(config.epochs * 0.5):
            break

    logger.info('Best Performance: {}, {}'.format(best_srocc, best_plcc))
    logger.info("End Training!")

    if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
        return best_srocc, best_plcc, type_results
    else:
        return best_srocc, best_plcc


def train(train_loader,test_loader, model, optimizer,criterion, epoch, config):
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
    test_srocc, test_plcc, all_preds, all_labels = validate(config, model, test_loader)

    logger.info('\t%d\t%6.5f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                (epoch, sum(epoch_loss) / len(epoch_loss), train_srocc, train_plcc, test_srocc, test_plcc))

    return test_srocc, test_plcc,all_preds, all_labels

def validate(config, model, test_loader):
    logger.info("***** Running validation *****")
    model.eval()
    all_preds, all_labels = [], []

    for step, (images, labels) in enumerate(test_loader):
        imgs_rgb = images[0].to(device)
        imgs_ycbcr = images[1].to(device)
        labels = labels.to(device)
        preds = model(imgs_rgb, imgs_ycbcr)

        all_preds = all_preds + preds.cpu().tolist()
        all_labels = all_labels + labels.cpu().tolist()

    all_preds = np.mean(np.reshape(np.array(all_preds), (-1, config.test_patch_num)), axis=1)
    all_labels = np.mean(np.reshape(np.array(all_labels), (-1, config.test_patch_num)), axis=1)

    test_srocc = test_protocol(all_preds, all_labels, protocol='srocc')
    test_plcc = test_protocol(all_preds, all_labels, protocol='plcc')
    # test_rmse = test_protocol(all_preds, all_label, protocol='rmse')
    model.train()
    return test_srocc, test_plcc, all_preds, all_labels

def test_protocol(preds, labels, protocol='srocc'):
    if protocol == 'srocc':
        result = stats.spearmanr(preds, labels)[0]
    elif protocol == 'plcc':
        result = stats.pearsonr(preds, labels)[0]
    elif protocol == 'krcc':
        result = stats.kendalltau(preds, labels)[0]
    elif protocol == 'rmse':
        result = np.sqrt(mean_squared_error(preds, labels))
    else:
        result = None
        logger.info('Invalid evaluation criteria were provided.')
    return result

def evaluate_distortion_type_performance(test_img_num, pred_scores, ground_truth, dist_type_num, dist_level=5):
    '''
    Note: Ensure the shuffle option of the DataLoader is set to False when calling this function.
    '''
    def get_dist_type_matrix(test_img_num, pred_scores, dist_type_num):
        scores_per_img = dist_type_num * dist_level
        preds_matrix = pred_scores.reshape(test_img_num, scores_per_img)
        # mos_matrix = np.zeros((test_img_num, scores_per_img))
        # for i in range(test_img_num):
        #     mos_matrix[i,:] = pred_scores[i*scores_per_img:(i+1)*scores_per_img]
        dist_type_matrix = np.zeros((dist_type_num, test_img_num * dist_level))
        for j in range(dist_type_num):
            dist_type_matrix[j, :] = preds_matrix[:, j*dist_level:(j+1)*dist_level].flatten()
        return dist_type_matrix

    pred_dist_type = get_dist_type_matrix(test_img_num, pred_scores, dist_type_num)
    gt_dist_type = get_dist_type_matrix(test_img_num, ground_truth, dist_type_num)
    all_srocc_results = []
    all_plcc_results = []
    for k in range(dist_type_num):
        srocc = stats.spearmanr(pred_dist_type[k,:], gt_dist_type[k,:])[0]
        plcc = stats.pearsonr(pred_dist_type[k, :], gt_dist_type[k, :])[0]
        all_srocc_results.append(srocc)
        all_plcc_results.append(plcc)
    np.set_printoptions(precision=6)
    all_srocc_results = np.asarray(all_srocc_results)
    # all_plcc_results = np.asarray(all_plcc_results)
    return all_srocc_results

def save_checkpoint(state, config):
    model_checkpoint = os.path.join(config.output_dir, "%s_checkpoint.pth.tar" % (config.dataset))
    torch.save(state, model_checkpoint)
    logger.info("Saved model checkpoints to [DIR: %s]", config.output_dir)


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
    logger.info('Experimental Configurations : %s ', config)
    # Setup logging
    logger.setLevel(level=logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./results/training_log.txt', mode='a', encoding='utf-8')
    fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


    logger.info("\033[91m" + "=" * 40)
    logger.info("== Training on {} for {} epochs ==".format(config.dataset, config.epochs))
    logger.info("=" * 40 + "\033[0m")

    config.path = dataset_path[config.dataset]

    srocc_all = np.zeros(config.train_test_round, dtype=np.float32)
    plcc_all = np.zeros(config.train_test_round, dtype=np.float32)

    if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
        dist_type_num = 24 if config.dataset == 'tid2013' else 25
        type_results = np.zeros((config.train_test_round, dist_type_num), dtype=np.float32)

    for i in range(1, config.train_test_round+1):
        if config.seed == 0:
            pass
        else:
            logger.info('Using the seed = {} for {}-th experiment'.format(config.seed*i, i))
            torch.manual_seed(i* config.seed)
            torch.cuda.manual_seed(i * config.seed)
            np.random.seed(i * config.seed)
            random.seed(i * config.seed)

        total_num_images = dataset_split_img_num[config.dataset]
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(total_num_images)
        config.train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
        config.test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

        if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
            srocc_all[i-1], plcc_all[i-1], type_results[i-1, :] = main(config)
        else:
            srocc_all[i - 1], plcc_all[i - 1] = main(config)

    if config.dataset == 'tid2013' or config.dataset == 'kadid-10k':
        np.savetxt('./results/{}_type_performance.csv'.format(config.dataset), type_results, fmt='%f', delimiter='\n')

    logger.info('{}: all srocc: {}'.format(config.dataset, srocc_all))
    logger.info('{}: all plcc: {}'.format(config.dataset, plcc_all))
    srocc_median, plcc_median = np.median(srocc_all), np.median(plcc_all)
    logger.info('%s : Testing SRCC %4.4f,\t PLCC %4.4f' % (config.dataset, srocc_median, plcc_median))

