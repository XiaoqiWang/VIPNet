import argparse

def Configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013',
                        help='synthetic distortion (|live|csiq|tid2013|kadid-10k) and authentic distortion (livec|koniq)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='the path to save results')
    # settings for cpm and dpm
    parser.add_argument('--dpm_checkpoints',  type=str,default=r'./pretrained_model/botnet_model_best.pth.tar',
                        help='The path to pre-trained weights of distortion perception module (DPM)')
    parser.add_argument('--cpm', type=str, default='resnet50',
                        help='The backbone of content perception module (CPM)')
    parser.add_argument('--cpm_channels', type=int, default=2048,
                        help='the out_channels of CPM if use multi-scale features')
    parser.add_argument('--multi_scale', default=True, action='store_false')
    parser.add_argument('--is_freeze_cpm', default=True, action='store_false')
    parser.add_argument('--is_freeze_dpm', default=True, action='store_false')

    # settings of the visual interaction module
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=128,
                        help='embedded dimensions of visual interaction module')
    parser.add_argument('--depth', dest='depth', type=int, default=2,
                        help='network depth of visual interaction module')
    parser.add_argument('--num_heads', dest='num_heads', type=int, default=16,
                        help='number of heads of multi-head attention')
    # optimizer setup
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4,
                        help='weight decay')
    # dataloader setup
    parser.add_argument('--train_bs', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--eval_bs',  type=int, default=32,
                        help='batch size for validation')
    parser.add_argument('--train_patch_num', type=int, default=12,
                        help='the number of patch for training stage')
    parser.add_argument('--test_patch_num', type=int, default=12,
                        help='the number of patch for testing stage')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # training setup
    parser.add_argument('--epochs', dest='epochs', type=int, default=35,
                        help='totall epochs for training model')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=1,
                        help='start epoch')
    parser.add_argument('--seed', dest='seed', type=int, default=20220426,
                        help='Set random seeds for replication')
    parser.add_argument('--train_test_round', dest='train_test_round', type=int, default=10,
                        help='train-test times')
    parser.add_argument('--cuda', default=True, action='store_false',
                        help='using gpu for training model')

    # cross-dataset test settings
    parser.add_argument('--train_dataset', type=str, default='live',
                        help='the training dataset in cross-dataset test')
    parser.add_argument('--test_dataset1', type=str, default='csiq',
                        help='the 1st test dataset')
    parser.add_argument('--test_dataset2', type=str, default='tid2013',
                        help='The 2nd test dataset')
    return parser.parse_args()

if __name__ == '__main__':
    config = Configs()
    for arg in vars(config):
        print(arg, getattr(config, arg))
