from data_provider.data_factory import data_provider, my_data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, SimpleCNN, DInformer, DTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, MyVisual, test_params_flop, get_save_name
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time
from datetime import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')       # 忽略warnings信息


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.my_visual = MyVisual(5)
        self.train_save_dir, self.train_save_name = None, None
        self.res_save_dir = None

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'OurCNN': SimpleCNN,
            'DInformer': DInformer,
            'DTransformer': DTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _my_get_data(self, flag, scaler=None):
        data_set, data_loader, new_scaler = my_data_provider(self.args, flag, scaler=scaler)
        return data_set, data_loader, new_scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion == 'mse':
            return nn.MSELoss()
        elif self.args.criterion == 'mae':
            return nn.L1Loss()
        elif self.args.criterion == 'huber':
            def huber_loss(y_true, y_pred, delta=0.37):
                delta = torch.tensor(delta)
                residual = torch.abs(y_true - y_pred)
                condition = torch.lt(residual, delta)
                small_res = 0.5 * torch.square(residual)
                large_res = delta * residual - 0.5 * torch.square(delta)
                loss = torch.where(condition, small_res, large_res)
                # print(loss)
                return torch.mean(loss)
            return huber_loss

    def _init_seeds(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)  # sets the seed for generating random numbers.
            torch.cuda.manual_seed(
                seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
            torch.cuda.manual_seed_all(
                seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'CNN' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'CNN' in self.args.model:
                        outputs = self.model(batch_x.permute(0, 2, 1)).view(-1, self.args.pred_len, 1)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()       # .detach()：不需要计算其梯度，不具有grad
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        self._init_seeds()

        train_data, train_loader = self._get_data(flag='train')
        # train_data, train_loader, scaler = self._my_get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
            # vali_data, vali_loader, _ = self._my_get_data(flag='val', scaler=scaler)
            # test_data, test_loader, _ = self._my_get_data(flag='test', scaler=scaler)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, patience=self.args.patience, verbose=True)       # 用于提前结束训练的逻辑判断

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.train()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)       # 相比.cuda()方法，to()快10倍

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # former decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()     # 创建一个与输入形状相同的全零张量
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'CNN' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'former' in self.args.model:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'CNN' in self.args.model:
                        outputs = self.model(batch_x.permute(0, 2, 1)).view(-1, self.args.pred_len, 1)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:      # 每100个iteration输出一次结果
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}ms'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                x = batch_x.detach().cpu().numpy()
                pred, true = outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()
                gt = np.concatenate((x[0, -10:, -1], true[0, :, -1]), axis=0)
                prd = np.concatenate((x[0, -10:, -1], pred[0, :, -1]), axis=0)
                index = i * self.args.batch_size
                self.my_visual(gt, prd, index)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)     # loss的计算方式是mean，每个batch得到一个loss，这里是求整个epoch的平均loss
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)        # total loss
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                early_stopping.rec_info(train_loss, vali_loss, test_loss, epoch)
                early_stopping(vali_loss, self.model, self.args)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

                early_stopping(train_loss, self.model, self.args)

            self.train_save_dir, self.train_save_name = early_stopping.get_save_info()

            # 如果改epoch训练是有效的（验证集精度提高）则重新可视化训练过程
            if early_stopping.pic_save_en:
                pic_save_name = 'train.jpg' if self.args.pic_save_key is None else f'{self.args.pic_save_key}_train.jpg'
                self.my_visual.plot_all(os.path.join(self.train_save_dir, pic_save_name), self.args.data)
                early_stopping.turn_off_save_en()

            # 清空该epoch记录的训练过程
            self.my_visual.reset()

            if early_stopping.early_stop:
                print("Early stopping")
                break
            else:
                continue

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 2023.3.21注释 感觉没有太多实际作用
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        #
        # return self.model

    def test(self):
        best_model_path = os.path.join(self.train_save_dir, self.train_save_name + '.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        test_data, test_loader = self._get_data(flag='test')
        # _, _, scaler = self._my_get_data(flag='train')
        # test_data, test_loader, _ = self._my_get_data(flag='test', scaler=scaler)
        preds = []
        trues = []
        # inputx = []

        # 用于保存pred和gt比较图表的文件夹路径
        save_name = get_save_name(self.args)
        self.res_save_dir = os.path.join('./test_results/', self.args.model, save_name)
        if not os.path.exists(self.res_save_dir):
            os.makedirs(self.res_save_dir)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'CNN' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'CNN' in self.args.model:
                        outputs = self.model(batch_x.permute(0, 2, 1)).view(-1, self.args.pred_len, 1)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())

                # 把input后10个加进来是为了方便观测输入序列是否有断层现象
                x = batch_x.detach().cpu().numpy()
                gt = np.concatenate((x[0, -10:, -1], true[0, :, -1]), axis=0)
                prd = np.concatenate((x[0, -10:, -1], pred[0, :, -1]), axis=0)
                index = i * self.args.batch_size
                self.my_visual(gt, prd, index)
                # gt = np.tile(gt.reshape(-1, 1), reps=(1, 7))
                # prd = np.tile(prd.reshape(-1, 1), reps=(1, 7))
                # self.my_visual(test_data.inverse_transform(gt)[:, -1],
                #                test_data.inverse_transform(prd)[:, -1],
                #                index,
                #                )

                # original 注释时间3.21 22:51
                # Linear作者仅把第一个batch，以及最后一个变量进行可视化，网络训练并没有利用变量关的关系
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)      # 一次完成多个数组的拼接
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(save_dir, str(i) + '.pdf'))

        # 记录预测过程并保存为.jpg格式
        pic_save_name = 'test.jpg' if self.args.pic_save_key is None else f'{self.args.pic_save_key}_test.jpg'
        self.my_visual.plot_all(os.path.join(self.res_save_dir, pic_save_name), self.args.data)
        self.my_visual.reset()

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)     # dataloardnum, batch, pre_len, channel
        trues = np.array(trues)
        # inputx = np.array(inputx)

        # 这一步貌似是针对Transformer输出的格式不规范才使用的，对于Linear下面的reshape操作没有意义
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # 标准化数据的误差分析
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 还原数据的误差分析
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = np.array(trues)
        trues = trues.reshape(-1, trues.shape[-1])
        if test_data.scale:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)

        original_mae, original_mse, _, _, _, _, _ = metric(preds, trues)
        print('mse:{}, mae:{}'.format(original_mae, original_mse))

        # original
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # my record
        xlsx_path = r'./test_results/result.xlsx'
        df = pd.read_excel(xlsx_path)
        cols_data = df.columns[1:]
        df_data = df[cols_data]

        time_info = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_data = {'Data': self.args.data, 'Time': time_info, 'Model': self.args.model,
                    'SeqLen': self.args.seq_len, 'PredLen': self.args.pred_len,
                    'ID': self.args.model_id, 'Param': f'b_{self.args.batch_size}',
                    'MAE': mae, 'MSE': mse, 'MAE(ori)': original_mae, 'MSE(ori)': original_mse,
                    'RMSE': rmse, 'MAPE': mape, 'MSPE': mspe, 'RES': rse}

        df_data = df_data.append(new_data, ignore_index=True)
        df_data = df_data.sort_values(['Data', 'Model', 'SeqLen', 'PredLen'])
        df_data.insert(0, 'Index', range(1, len(df_data) + 1))
        df_data.to_excel(xlsx_path, index=False)

        # result save
        folder_path = os.path.join(self.res_save_dir, save_name+'.npy')
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path, preds)        # 默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

    # def predict(self, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
    #
    #     if load:
    #         best_model_path = os.path.join(self.train_save_dir, self.train_save_name + '.pth')
    #         self.model.load_state_dict(torch.load(best_model_path))
    #
    #     preds = []
    #     trues = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if 'Linear' in self.args.model:
    #                         outputs = self.model(batch_x)
    #                     else:
    #                         if self.args.output_attention:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         else:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if 'Linear' in self.args.model:
    #                     outputs = self.model(batch_x)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             batch_y = batch_y.detach().cpu().numpy()
    #             outputs = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(outputs)   # pred:{dataloard num, batch, label length， channel}
    #             trues.append(batch_y)
    #
    #             x = batch_x.detach().cpu().numpy()
    #             gt = np.concatenate((x[0, -10:, -1], batch_y[0, :, -1]), axis=0)
    #             prd = np.concatenate((x[0, -10:, -1], outputs[0, :, -1]), axis=0)
    #             index = i * self.args.batch_size
    #             self.my_visual(pred_data.inverse_transform(np.tile(gt.reshape(-1, 1), reps=(1, 7)))[:, -1],
    #                            pred_data.inverse_transform(np.tile(prd.reshape(-1, 1), reps=(1, 7)))[:, -1],
    #                            index)
    #
    #     self.my_visual.plot_all(os.path.join(self.res_save_dir, 'pred.jpg'), self.args.data)
    #     self.my_visual.reset()
    #
    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = np.array(trues)
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     if (pred_data.scale):
    #         preds = pred_data.inverse_transform(preds)
    #         trues = pred_data.inverse_transform(trues)
    #
    #     mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #
    #     np.save(os.path.join(self.res_save_dir, 'real_prediction.npy'), preds)
    #     pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
    #                  columns=pred_data.cols)\
    #         .to_csv(os.path.join(self.res_save_dir + 'real_prediction.csv'), index=False)
    #
    #     return

