# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
import torch.optim as optim

from scipy import stats
from tqdm import tqdm
from config_aesthetic import get_args

from model.gcn_dataloader_1024 import AVADataset, test_AVADataset
from model.single_rsgcn_loss_emd import RsgcnModel


def main():
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config
    config = get_args()

    # model
    model = RsgcnModel(1024, 1024, 1024, 5, 1)
    model = model.cuda()

    # warm start
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path,
                                                      'AIAA-semantic-GCN-mse-loss-model-epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded pretrain model')

    # setting lr
    conv_base_lr = config.conv_base_lr
    optimizer = optim.Adam(model.parameters(), conv_base_lr)

    # loss function
    criterion = torch.nn.SmoothL1Loss()

    # model size
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # 　training
    if config.train:
        # read dataset
        trainset = AVADataset(config.train_csv_file, config.train_img_path, config.train_refer_file)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers)
        
        valset = AVADataset(config.val_csv_file, config.train_img_path, config.val_refer_file)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.train_batch_size,
                                                   shuffle=False, num_workers=config.num_workers)

        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        train_srcc = []
        train_plcc = []
        
        val_srcc = []
        val_plcc = []

        # start training
        print('its learning time: ')

        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            training_loss = 0.0

            train_pred_score = []
            train_gt_score = []
            
            val_pred_score_list = []
            val_gt_score_list = []

            for i, data in tqdm(enumerate(train_loader)):
                refer_feature = data['refer_feature'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                # print('refer_feature', refer_feature.shape)
                norm_score = data['norm_score'].to(device).float()
                score = data['score'].to(device).float()

                # 输出分数分布                
                gcn_outputs = model(refer_feature)
                # print('gcn_outputs', gcn_outputs.shape)
                gcn_outputs = torch.flatten(gcn_outputs)
                pred_score = gcn_outputs.sigmoid()
                # print('pred_score', pred_score.shape)

                optimizer.zero_grad()

                # loss function
                loss_gcn = criterion(norm_score, gcn_outputs)
                # print('loss_gcn: ', loss_gcn)
                
                training_loss += loss_gcn.item()
                batch_losses.append(loss_gcn.item())

                train_pred_score += list(pred_score.cpu().detach().numpy())
                train_gt_score += list(score.cpu().detach().numpy())

                loss_gcn.backward()
                optimizer.step()

                if i % 50 == 49:
                    print('Epoch: %d/%d | Step: %d/%d | Training Rank loss: %.4f' % (epoch + 1,
                                                                                     config.epochs,
                                                                                     i + 1,
                                                                                     len(trainset) // config.train_batch_size + 1,
                                                                                     training_loss/50))
                    training_loss = 0.0

            train_pred_score = np.array(train_pred_score) * (config.max_score - config.min_score) + config.min_score
            
            srocc = stats.spearmanr(train_pred_score, train_gt_score)[0]
            plcc  = stats.pearsonr(train_pred_score, train_gt_score)[0]
            print('% train SROCC of mean: {}'.format(srocc))
            print('% train PLCC of mean: {}'.format(plcc))
            
            train_srcc.append(srocc)
            train_plcc.append(plcc)

            # compute mean loss
            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training Rank loss: %.4f' % (epoch + 1, avg_loss))
            # writer.add_scalars('Loss_group', {'train_loss': avg_loss}, epoch)
            print('Epoch %d gcn loss: %.4f' % (epoch + 1, loss_gcn))
            # writer.add_scalars('Loss_group', {'gcn_loss': loss_gcn}, epoch)

            # do validation after each epoch　
            batch_val_losses = []
            for j, data in enumerate(val_loader):
                refer_feature = data['refer_feature'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                norm_score = data['norm_score'].to(device).float()
                score = data['score'].to(device).float()

                optimizer.zero_grad()

                # 输出分数分布
                with torch.no_grad():
                    outputs = model(refer_feature)
                outputs = torch.flatten(outputs)
                val_pred_score = outputs.sigmoid()
                val_loss = criterion(val_pred_score, norm_score)

                batch_val_losses.append(val_loss.item())
                
                val_pred_score_list += list(val_pred_score.cpu().detach().numpy())
                val_gt_score_list += list(score.cpu().detach().numpy())
                
                # loss.backward()
                # optimizer.step()
                
            val_pred_score_array = np.array(val_pred_score_list) * (config.max_score - config.min_score) + config.min_score
            #             print(val_pred_score_array.shape)
            #             print(len(val_gt_score_list))
            
            val_srocc_value = stats.spearmanr(val_pred_score_array, val_gt_score_list)[0]
            val_plcc_value = stats.pearsonr(val_pred_score_array, val_gt_score_list)[0]
            print('% val SROCC of mean: {}'.format(val_srocc_value))
            print('% val PLCC of mean: {}'.format(val_plcc_value))
            
            val_srcc.append(val_srocc_value)
            val_plcc.append(val_plcc_value)

            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.train_batch_size + 1)
            val_losses.append(avg_val_loss)

            print('Epoch %d completed. Averaged regression loss on val set: %.4f.' % (epoch + 1, avg_val_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 3 == 0:
                conv_base_lr = conv_base_lr / 10
                optimizer = optim.Adam(model.parameters(), conv_base_lr)
            # writer.add_scalars('LR', {'learn_rate': conv_base_lr}, epoch)
            # Use early stopping to monitor training
            # print('Saving model...')
            torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'AIAA-semantic-GCN-mse-loss-model-epoch-%d.pkl' % (epoch + 1)))
            print('Done.\n')
            
        # save loss, srcc, plcc
        save_all_metric = '/userhome/Aesthetic_art/MILNet/baseline_mid_feat_1024/MILNet_baseline_loss_srcc_plcc.csv'
        with open(save_all_metric, 'w') as f:
            line = 'train_loss,' + 'val_loss,' + 'train_srcc,' + 'train_plcc,' + 'val_srcc,' + 'val_plcc' + '\n'
            for i in range(config.epochs):
                line += str(train_losses[i]) + ',' + str(val_losses[i]) + ',' + str(train_srcc[i]) + ',' + str(train_plcc[i]) + ',' + str(val_srcc[i]) +',' + str(val_plcc[i]) + '\n'
            f.write(line)

            
    # testing
    if config.test:
        model.eval()
        print('its test time: ')

        testset = test_AVADataset(config.test_csv_file, config.train_img_path, config.test_refer_file)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False,
                                                  num_workers=config.num_workers)

        for test_epoch in range(0, config.epochs):
            test_pred_score = []
            image_name = []

            model.load_state_dict(torch.load(config.ckpt_path + 'AIAA-semantic-GCN-mse-loss-model-epoch-%d.pkl' % (test_epoch + 1)))

            for data in tqdm(test_loader):
                # forward
                refer_feature = data['refer_feature'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                name = data['name']

                with torch.no_grad():
                    gcn_outputs = model(refer_feature)

                gcn_outputs = gcn_outputs.view(-1)

                test_pred_score += list(gcn_outputs.cpu().numpy())
                image_name += list(name)

            test_pred_score = np.array(test_pred_score) * (config.max_score - config.min_score) + config.min_score
            test_pred_score_list = test_pred_score.tolist()

            save_result_csv_path = "/userhome/Aesthetic_art/MILNet/baseline_mid_feat_1024/result_" + str(test_epoch) + '.csv'
            with open(save_result_csv_path, 'w') as f:
                first_line = 'image,score' + '\n'
                for i in range(len(test_pred_score_list)):
                    first_line += image_name[i] + ',' + str(test_pred_score_list[i]) + '\n'
                f.write(first_line)


if __name__=='__main__':
    main()



