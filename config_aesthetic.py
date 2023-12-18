import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--refer_img_path', type=str, default='/dataset/best_swin_v2_feat_1024/')
    parser.add_argument('--train_img_path', type=str, default='/dataset/best_swin_v2_feat_1024/')

    parser.add_argument('--train_csv_file', type=str, default='dataset/train_set.csv')
    parser.add_argument('--train_refer_file', type=str, default='/code/MILNet/refer_index/AIAA_semantic_style_train_refer_50.txt')
    parser.add_argument('--val_csv_file', type=str, default='dataset/val_set.csv')
    parser.add_argument('--val_refer_file', type=str, default='/code/MILNet/refer_index/AIAA_semantic_style_val_refer_50.txt')
    parser.add_argument('--test_csv_file', type=str, default='dataset/test_set.csv')
    parser.add_argument('--test_refer_file', type=str, default='/code/MILNet/refer_index/AIAA_semantic_style_test_refer_50.txt')

    parser.add_argument('--min_score', type=float, default=2.75695)
    parser.add_argument('--max_score', type=float, default=10.0)
    
    # training parameters`
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=3e-5)
    parser.add_argument('--dense_lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpt/baseline_mid_feat_1024/')
    # parser.add_argument('--result_path', type=str, default='result/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=0)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()
    return config
