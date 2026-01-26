from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import random




def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class Trainer_Stage1(object):
    def __init__(self, encoder, net_modal_classifer,memory=None):
        super(Trainer_Stage1, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory
        self.memory_instance_ir = memory
        self.memory_instance_rgb = memory
        self.net_modal_classifer = net_modal_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.cuda()

    def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, modal_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=False):
        self.encoder.train()
        self.net_modal_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_ir = AverageMeter()
        losses_rgb = AverageMeter()
        losses_instance_ir = AverageMeter()
        losses_instance_rgb = AverageMeter()
        losses_modal = AverageMeter()
        acc_modal = AverageMeter()
        end = time.time()
        modal_i_labels = Variable(torch.ones(128).long().cuda())
        modal_v_labels = Variable(torch.zeros(128).long().cuda())
        modal_g_labels = Variable(2 * torch.ones(128).long().cuda())
        true_label = torch.cat([modal_v_labels, modal_g_labels, modal_i_labels], dim=0)
        loss_all_ir=0
        loss_all_rgb=0
        loss = 0
        loss_ir=0
        loss_rgb=0
        modal_classifier_acc=0
        modal_loss=0
        loss_instance_ir=0
        loss_instance_rgb = 0
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward

            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)



            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            ###random features labels indexes
            random_indexes = list(range(labels_rgb.shape[0]))
            random.shuffle(random_indexes)
            f_out_rgb = f_out_rgb[random_indexes]
            labels_rgb = labels_rgb[random_indexes]
            indexes_rgb = indexes_rgb[random_indexes]

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_instance_ir = self.memory_instance_ir(f_out_ir, indexes_ir)
            loss_instance_rgb = self.memory_instance_rgb(f_out_rgb, indexes_rgb)
            loss = loss_ir+ loss_rgb+ loss_instance_ir+loss_instance_rgb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_ir.update(loss_ir.item())
            losses_rgb.update(loss_rgb.item())
            losses_instance_ir.update(loss_instance_ir.item())
            losses_instance_rgb.update(loss_instance_rgb.item())
            acc_modal.update(modal_classifier_acc)
            # losses_modal.update(modal_loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss rgb instance {:.3f}\t'
                      'Loss ir instance {:.3f}\t'
                      ' modal class acc {:.3f}\t'
                      ' modal_loss {:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,losses_ir.avg,losses_rgb.avg,losses_instance_rgb.avg,losses_instance_ir.avg,acc_modal.avg,losses_modal.avg))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


class Trainer_Stage2(object):
    def __init__(self, encoder, net_modal_classifer,memory=None):
        super(Trainer_Stage2, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory
        self.memory_instance_ir = memory
        self.memory_instance_rgb = memory
        self.memory_instance_all = memory
        self.net_modal_classifer = net_modal_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.cuda()
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, modal_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=False):
        self.encoder.train()
        self.net_modal_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_ir = AverageMeter()
        losses_rgb = AverageMeter()
        losses_instance_ir = AverageMeter()
        losses_instance_rgb = AverageMeter()
        losses_all_rgb_ir = AverageMeter()
        losses_all_instance_ir = AverageMeter()
        losses_all_instance_rgb = AverageMeter()
        losses_modal = AverageMeter()

        end = time.time()
        modal_i_labels = Variable(torch.ones(128).long().cuda())
        modal_v_labels = Variable(torch.zeros(128).long().cuda())
        modal_g_labels = Variable(2 * torch.ones(128).long().cuda())
        true_label = torch.cat([modal_v_labels, modal_g_labels, modal_i_labels], dim=0)
        loss_all_ir=0
        loss_all_rgb=0
        loss = 0
        loss_ir=0
        loss_rgb=0
        modal_classifier_acc=0
        modal_loss=0

        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)



            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            ###random features labels indexes
            random_indexes = list(range(labels_rgb.shape[0]))
            random.shuffle(random_indexes)
            f_out_rgb = f_out_rgb[random_indexes]
            labels_rgb = labels_rgb[random_indexes]
            indexes_rgb = indexes_rgb[random_indexes]

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_instance_ir = self.memory_instance_ir(f_out_ir, indexes_ir)

            loss_instance_rgb = self.memory_instance_rgb(f_out_rgb, indexes_rgb)

            loss = loss_ir + loss_rgb +loss_instance_ir +loss_instance_rgb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_ir.update(loss_ir.item())
            losses_rgb.update(loss_rgb.item())
            losses_instance_ir.update(loss_instance_ir.item())
            losses_instance_rgb.update(loss_instance_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                       'loss_instance_ir {:.3f}\t'
                      'loss_instance_rgb {:.3f}\t'
                        'Loss all {:.3f}\t'
                      'loss_all_instance_ir {:.3f}\t'
                      'loss_all_instance_rgb {:.3f}\t'
                      ' modal class acc {:.3f}\t'
                      ' modal_loss {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,losses_ir.avg,losses_rgb.avg,losses_instance_ir.avg,losses_instance_rgb.avg,losses_all_rgb_ir.avg,losses_all_instance_ir.avg,losses_all_instance_rgb.avg,modal_classifier_acc,losses_modal.avg))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


class Trainer_Stage3(object):
    def __init__(self, encoder, net_modal_classifer,memory=None):
        super(Trainer_Stage3, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory
        self.memory_instance_ir = memory
        self.memory_instance_rgb = memory
        self.memory_instance_all = memory
        self.net_modal_classifer = net_modal_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.cuda()
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, modal_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=False):
        self.encoder.train()
        self.net_modal_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_ir = AverageMeter()
        losses_rgb = AverageMeter()
        losses_instance_ir = AverageMeter()
        losses_instance_rgb = AverageMeter()
        losses_all_rgb_ir = AverageMeter()
        losses_all_instance_ir = AverageMeter()
        losses_all_instance_rgb = AverageMeter()
        losses_modal = AverageMeter()

        end = time.time()
        modal_i_labels = Variable(torch.ones(128).long().cuda())
        modal_v_labels = Variable(torch.zeros(128).long().cuda())
        modal_g_labels = Variable(2 * torch.ones(128).long().cuda())
        true_label = torch.cat([modal_v_labels, modal_g_labels, modal_i_labels], dim=0)
        loss_all_ir=0
        loss_all_rgb=0
        loss = 0
        loss_ir=0
        loss_rgb=0
        modal_classifier_acc=0
        modal_loss=0

        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)


            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            ###random features labels indexes
            random_indexes = list(range(labels_rgb.shape[0]))
            random.shuffle(random_indexes)
            f_out_rgb = f_out_rgb[random_indexes]
            labels_rgb = labels_rgb[random_indexes]
            indexes_rgb = indexes_rgb[random_indexes]

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_instance_ir = self.memory_instance_ir(f_out_ir, indexes_ir)

            loss_instance_rgb = self.memory_instance_rgb(f_out_rgb, indexes_rgb)


            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            # process inputs
            inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
            # forward
            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)
            indexes_all_rgb = torch.cat((indexes_all_rgb, indexes_all_rgb), -1)
            indexes_all_ir = indexes_all_ir + self.memory_instance_rgb.num_samples


            _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, pool_all_rgb, pool_all_ir = self._forward(inputs_all_rgb, inputs_all_ir,
                                                                                             label_1=labels_all_rgb,
                                                                                             label_2=labels_all_ir, modal=0)
            ##concat f_out_all_ir and f_out_all_rgb and random them
            # loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            # loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            f_out_all_concat =   torch.cat([f_out_all_rgb,f_out_all_ir], dim=0)
            labels_all_concat = torch.cat((labels_all_rgb, labels_all_ir), -1)

            ###random features labels indexes
            random_indexes = list(range(labels_all_concat.shape[0]))
            random.shuffle(random_indexes)
            f_out_all_concat = f_out_all_concat[random_indexes]
            labels_all_concat = labels_all_concat[random_indexes]


            loss_all_rgb_ir = self.memory_all(f_out_all_concat, labels_all_concat)


            ###random features labels indexes
            random_indexes = list(range(labels_all_rgb.shape[0]))
            random.shuffle(random_indexes)
            f_out_all_rgb = f_out_all_rgb[random_indexes]
            labels_all_rgb = labels_all_rgb[random_indexes]
            indexes_all_rgb = indexes_all_rgb[random_indexes]


            loss_all_instance_ir = self.memory_instance_all(f_out_all_ir,indexes_all_ir)
            loss_all_instance_rgb = self.memory_instance_all(f_out_all_rgb,indexes_all_rgb)

            
            loss = loss_ir + loss_rgb + loss_instance_ir + loss_instance_rgb + 0.7*(loss_all_rgb_ir + loss_all_instance_ir + loss_all_instance_rgb)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_ir.update(loss_ir.item())
            losses_rgb.update(loss_rgb.item())
            losses_instance_ir.update(loss_instance_ir.item())
            losses_instance_rgb.update(loss_instance_rgb.item())
            losses_all_rgb_ir.update(loss_all_rgb_ir.item())
            losses_all_instance_ir.update(loss_all_instance_ir.item())
            losses_all_instance_rgb.update(loss_all_instance_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                       'loss_instance_ir {:.3f}\t'
                      'loss_instance_rgb {:.3f}\t'
                        'Loss all {:.3f}\t'
                      'loss_all_instance_ir {:.3f}\t'
                      'loss_all_instance_rgb {:.3f}\t'
                      ' modal class acc {:.3f}\t'
                      ' modal_loss {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,losses_ir.avg,losses_rgb.avg,losses_instance_ir.avg,losses_instance_rgb.avg,losses_all_rgb_ir.avg,losses_all_instance_ir.avg,losses_all_instance_rgb.avg,modal_classifier_acc,losses_modal.avg))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


def update_ema_variables(net, net_ema, alpha, global_step):
    alpha = min(1-1/(global_step + 1), alpha)
    print('alpha',alpha)
    for ema_param, param in zip(net_ema.parameters(),net.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha,param.data)