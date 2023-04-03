import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import metric

plt.switch_backend('agg')


def get_save_name(args):
    amp = 1 if args.use_amp else 0
    if 'Linear' in args.model:
        save_name = 'id_{}-m_{}-d_{}-f_{}-s_{}-p_{}-m_{}-b_{}-amp_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.pred_len,
            args.moving_avg, args.batch_size, amp
        )
    elif 'CNN' in args.model:
        save_name = 'id_{}-m_{}-d_{}-f_{}-s_{}-p_{}-k_{}-b_{}-amp_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.pred_len,
            args.kernel_size, args.batch_size, amp
        )
    elif 'former' in args.model:
        save_name = 'id_{}-m_{}-d_{}-f_{}-s_{}-p_{}-amp_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.pred_len, amp
        )
    return save_name


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # lradj default 'type1'
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, args, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf      # 表示+∞
        self.delta = delta
        self.args = args
        self.writer = None
        self.pic_save_en = False
        self.path, self.save_dir, self.save_name = None, None, None
        self._get_path()

    def __call__(self, val_loss, model, args):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.writer.close()
        else:       # score为loss相反数，即loss越小，score越大模型表现越好
            self.best_score = score
            self.pic_save_en = True
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

    def rec_info(self, train_loss, val_loss, test_loss, epoch):
        self.writer.add_scalars(f'{self.save_name}', {
            'train loss': train_loss,
            'vali loss': val_loss,
            'test loss': test_loss,
        }, epoch)

    def _get_path(self):
        self.save_name = get_save_name(self.args)
        self.save_dir = os.path.join(self.args.checkpoints, self.args.model, self.save_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.writer = SummaryWriter(self.save_dir)
        self.save_path = os.path.join(self.save_dir, self.save_name + '.pth')

    def get_save_info(self):
        return self.save_dir, self.save_name

    def turn_off_save_en(self):
        self.pic_save_en = False


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()        # 添加图例
    plt.savefig(name, bbox_inches='tight')


class MyVisual():
    def __init__(self, num):
        self.true_list = []
        self.pred_list = []
        self.index_list = []        # 记录数据在测试集中的索引
        self.num = num
        self.show_num = num * num

    def __call__(self, true, pred, index):
        self.true_list.append(true)
        self.pred_list.append(pred)
        self.index_list.append(index)

    def plot_all(self, path, data):
        intervel = len(self.true_list)//self.show_num
        plot_true = self.true_list[::intervel]
        plot_pred = self.pred_list[::intervel]
        plot_index = self.index_list[::intervel]

        fig, axs = plt.subplots(self.num, self.num, figsize=(25, 25))
        for i, ax in enumerate(axs.flatten()):
            # 实际上输入的true和pred会把输入序列最后10个数据加入，所以这里需要-10
            stat = plot_index[i]-10
            x = np.arange(stat, stat+len(plot_true[i]))
            true = plot_true[i]
            pred = plot_pred[i]

            mae = np.mean(np.abs(pred - true))
            mse = np.mean((pred - true) ** 2)
            ax.text(1, 1, f"MAE = {mae:.2f}, MSE = {mse:.2f}",
                    ha='center', va='bottom', transform=ax.transAxes)

            # ax.set_ylim(-1.2, 1.2)
            ax.plot(x, true, label='GroundTruth', linewidth=2)
            ax.plot(x, pred, label='Prediction', linewidth=2)
            ax.set_title(f"{data}-Plot {i}")
            ax.legend()

        plt.show()
        plt.savefig(path, bbox_inches='tight')

    def reset(self):
        # reset
        self.true_list = []
        self.pred_list = []
        self.index_list = []


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
