import torch
import time
import os
import sys
from utils.utils import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from utils.SI_SNR import si_snr_loss
from utils.SI_SNR import sisnr
import matplotlib.pyplot as plt
from model.Conv_TasNet import check_parameters
from itertools import permutations

def to_device(dicts, device):
    '''
       load dict data to cuda
    '''
    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        elif isinstance(datas,list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')


class Trainer():
    '''
       Trainer of Conv-Tasnet
       input:
             net: load the Conv-Tasnet model
             checkpoint: save model path
             optimizer: name of opetimizer
             gpu_ids: (int/tuple) id of gpus
             optimizer_kwargs: the kwargs of optimizer
             clip_norm: maximum of clip norm, default: None
             min_lr: minimun of learning rate
             patience: Number of epochs with no improvement after which learning rate will be reduced
             factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
             logging_period: How long to print
             resume: the kwargs of resume, including path of model, Whether to restart
             stop: Stop training cause no improvement
    '''

    def __init__(self,
                 net,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 stop=10,
                 num_epochs=100):
        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid

        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # build the logger object
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=False)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # current epoch
        self.stop = stop

        # Whether to resume the model
        if resume['resume_state']:
            cpt = torch.load(os.path.join(
                resume['path'], self.checkpoint, 'best.pt'), map_location='cpu')
            self.cur_epoch = cpt['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume['path'], self.cur_epoch))
            net.load_state_dict(cpt['model_state_dict'])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs, state=cpt['optim_state_dict'])
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        # check model parameters
        self.param = check_parameters(self.net)

        # Reduce lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=patience, verbose=True, min_lr=min_lr)

        # logging
        self.logger.info("Starting preparing model ............")
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            self.gpuid, self.param))
        self.clip_norm = clip_norm
        # clip norm
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

        # number of epoch
        self.num_epochs = num_epochs
    

    def mse_sisnr_loss(self, ests, egs):
        # refs = egs["ref"]
        # num_spks = len(refs)
        # # mse_NxS = torch.nn.MSELoss()

        # def mse_NxS(s_, s):
        #     diff = s_ - s
        #     squared_diff = torch.pow(diff, 2)
        #     # squared_diff_sum = torch.sum(squared_diff, dim=0)
        #     # return torch.mean(squared_diff_sum)
        #     a = torch.mean(squared_diff, dim=1)
        #     return torch.mean(squared_diff, dim=1)
            
        # def mse_loss(permute):
        # # for one permute
        #     a = [mse_NxS(ests[s], refs[t])
        #             for s, t in enumerate(permute)]
            
        #     return sum(
        #         [mse_NxS(ests[s], refs[t])
        #         for s, t in enumerate(permute)]) / len(permute)

        # N = egs["mix"].size(0)
        # mse_mat = torch.stack(
        #     [mse_loss(p) for p in permutations(range(num_spks))])

        # max_perutt, _ = torch.max(mse_mat, dim=0)
        # return max_perutt / N

        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [sisnr(ests[s], refs[t])
                for s, t in enumerate(permute)]) / len(permute)
            # average the value

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, max_indexes = torch.max(sisnr_mat, dim=0)

        # Calculate MSE between ests and refs for the permute with maximum sisnr
        permutes = [p for p in permutations(range(num_spks))]
        mse_mat = torch.stack(
            [torch.mean((ests[s] - refs[permutes[max_idx.item()][t]]) ** 2)
            for s, t, max_idx in zip(range(num_spks), range(num_spks), max_indexes)])
        mse = torch.mean(mse_mat)

        return -torch.sum(max_perutt) / N + mse * 10
            
            # return mse

    def create_optimizer(self, optimizer, kwargs, state=None):
        '''
           create optimizer
           optimizer: (str) name of optimizer
           kwargs: the kwargs of optimizer
           state: the load model optimizer state
        '''
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, best=True):
        '''
            save model
            best: the best model
        '''
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoint,
                         "{0}.pt".format("best" if best else "last")))

    def train(self, train_dataloader):
        '''
           training model
        '''
        self.logger.info('Training model ......')
        losses = []
        start = time.time()
        current_step = 0
        for egs in train_dataloader:
            current_step += 1
            egs = to_device(egs, self.device)
            self.optimizer.zero_grad()
            ests = data_parallel(self.net, egs['mix'], device_ids=self.gpuid)
            # loss = si_snr_loss(ests, egs) + self.mse(ests, egs)
            loss = self.mse_sisnr_loss(ests, egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses.append(loss.item())
            if len(losses)%self.logging_period == 0:
                avg_loss = sum(
                    losses[-self.logging_period:])/self.logging_period
                self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))
        return total_loss_avg

    # 验证函数
    def val(self, val_dataloader):
        '''
           validation model
        '''
        self.logger.info('Validation model ......')
        self.net.eval()
        losses = []
        current_step = 0
        start = time.time()
        with torch.no_grad():
            for egs in val_dataloader:
                current_step += 1
                egs = to_device(egs, self.device)
                ests = data_parallel(self.net, egs['mix'], device_ids=self.gpuid)
                loss = si_snr_loss(ests, egs)
                losses.append(loss.item())
                if len(losses)%self.logging_period == 0:
                    avg_loss = sum(
                        losses[-self.logging_period:])/self.logging_period
                    self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))
        return total_loss_avg

    # 执行训练并验证的函数
    def run(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []
        
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(best=False)
            val_loss = self.val(val_dataloader)
            best_loss = val_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0

            self.scheduler.best = best_loss
            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1
                train_loss = self.train(train_dataloader)
                val_loss = self.val(val_dataloader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss > best_loss:
                    no_impr += 1
                    self.logger.info('no improvement, best loss: {:.4f}'.format(self.scheduler.best))
                else:
                    best_loss = val_loss
                    no_impr = 0
                    self.save_checkpoint(best=True)
                    self.logger.info('Epoch: {:d}, now best loss change: {:.4f}'.format(self.cur_epoch,best_loss))
                # schedule here
                self.scheduler.step(val_loss)
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.num_epochs))
            
         # loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_losses, 'b-', label=u'train_loss',linewidth=0.8)
        plt.plot(x, val_losses, 'c-', label=u'val_loss',linewidth=0.8)
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('conv_tasnet_LRS.png')
