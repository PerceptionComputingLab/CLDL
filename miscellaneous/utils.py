import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numbers
from torch.autograd import Variable
import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import random
import imageio
from torch.nn import KLDivLoss
from skimage.transform import resize, downscale_local_mean
import nibabel as nib
# from src.miscellaneous.metrics import dice
# from miscellaneous.metrics import dice
# from metrics import dice
# from src.data.datagenerator import CTData
from glob import glob
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import copy
from scipy import ndimage
import math


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class KLDivLossSeg(KLDivLoss):
    '''
    F.kl_div(input, target)  ----  target*log(target/input)
    Example:
        a = torch.randn(1, 4, 64,64,64)
        b = torch.randn(1,4,64,64,64)
        q = F.log_softmax(a, dim=1)
        q1 = F.softmax(a, dim=1)
        p = F.softmax(b, dim=1)
        # pytorch implementation
        kl = F.kl_div(q, target=p, reduction="none")
        kl_torch = F.kl_div(q, p,reduction="batchmean")
        # my implementation (the same as pytorch)
        kl_my = p*torch.log(p/q1)
        kl_my_f = torch.sum(torch.sum(kl_my, dim=1))
    '''
    def __init__(self, size_average=None, reduce=None, reduction="mean", log_target=False):
        super(KLDivLossSeg, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction,
                                           log_target=log_target)

    def forward(self, input, target):
        pixel_wise_kl = F.kl_div(input, target, reduction="none", log_target=self.log_target)

        # TODO test nan error
        flag = torch.any(torch.isnan(input))
        if flag:
            print("NaN occur!")

        # Add along the distribution channel
        kl = torch.sum(pixel_wise_kl, dim=1)
        # reduce dimention by batch and image dimensions
        if self.reduction == "mean":
            mean_kl = torch.mean(kl)
            return mean_kl
        elif self.reduction == "sum":
            sum_kl = torch.sum(kl)
            return sum_kl
        else:
            return kl


class KLDivLossSegSmooth(KLDivLoss):
    '''
    F.kl_div(input, target)  ----  target*log(target/input)
    Example:
        a = torch.randn(1, 4, 64,64,64)
        b = torch.randn(1,4,64,64,64)
        q = F.log_softmax(a, dim=1)
        q1 = F.softmax(a, dim=1)
        p = F.softmax(b, dim=1)
        # pytorch implementation
        kl = F.kl_div(q, target=p, reduction="none")
        kl_torch = F.kl_div(q, p,reduction="batchmean")
        # my implementation (the same as pytorch)
        kl_my = p*torch.log(p/q1)
        kl_my_f = torch.sum(torch.sum(kl_my, dim=1))
    '''
    def __init__(self, size_average=None, reduce=None, reduction="mean", epsilon=0.1):
        super(KLDivLossSegSmooth, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.epsilon = epsilon

    def forward(self, input, target):
        # add general label smoothing
        # calculate the number of non-zero categories at each data point
        count = torch.count_nonzero(target, dim=1)
        n_class = list(target.size())[1]

        # get mask which have to be smoothed
        mask = count != n_class
        # flag = torch.any(mask)
        if torch.any(mask):
            extra_pos = torch.zeros(count.shape, device=count.device)
            extra_neg = torch.zeros(count.shape, device=count.device)

            extra_pos[mask] = -self.epsilon/(torch.pow(count[mask], 2))
            extra_neg[mask] = self.epsilon/(count[mask]*(n_class-count[mask]))
            positive_mask = target > 0
            negative_mask = ~positive_mask
            # tt1 = torch.broadcast_to(extra_pos, target.shape)
            # tt2 = torch.broadcast_to(extra_neg, target.shape)
            target[positive_mask] = target[positive_mask] + torch.broadcast_to(extra_pos, target.shape)[positive_mask]
            target[negative_mask] = target[negative_mask] + torch.broadcast_to(extra_neg, target.shape)[negative_mask]

        # test = torch.sum(target, dim=1)
        pixel_wise_kl = F.kl_div(input, target, reduction="none")
        # Add along the distribution channel
        kl = torch.sum(pixel_wise_kl, dim=1)
        # reduce dimention by batch and image dimensions
        if self.reduction == "mean":
            mean_kl = torch.mean(kl)
            return mean_kl
        elif self.reduction == "sum":
            sum_kl = torch.sum(kl)
            return sum_kl
        else:
            return kl


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, prediction, label):
        label_one_hot = torch.zeros(prediction.shape)
        if prediction.device.type == "cuda":
                label_one_hot = label_one_hot.cuda(prediction.device.index)
        gt = torch.unsqueeze(label, dim=1)
        gt = gt.long()
        label_one_hot.scatter_(1, gt, 1)
        logsoftmax_pre = F.log_softmax(prediction, dim=1)
        weight_list = []
        loss = 0
        for i in range(self.n_classes):
            gti = label_one_hot[:,i, ...]
            predi = logsoftmax_pre[:,i,...]
            weight_i = 1- torch.sum(gti)/torch.sum(label_one_hot)
            weight_list.append(weight_i)
            loss = loss + -torch.mean(weight_i*gti*predi)
    
        return loss


def label_smooth(target, epsilon):
    # add general label smoothing
    # calculate the number of non-zero categories at each data point
    count = torch.count_nonzero(target, dim=1)
    n_class = list(target.size())[1]

    # get mask which have to be smoothed
    mask = count != n_class
    # flag = torch.any(mask)
    if torch.any(mask):
        extra_pos = torch.zeros(count.shape, device=count.device)
        extra_neg = torch.zeros(count.shape, device=count.device)

        extra_pos[mask] = -epsilon / (torch.pow(count[mask], 2))
        extra_neg[mask] = epsilon / (count[mask] * (n_class - count[mask]))
        positive_mask = target > 0
        negative_mask = ~positive_mask

        extra_pos = torch.unsqueeze(extra_pos, dim=1)
        extra_neg = torch.unsqueeze(extra_neg, dim=1)
        
        target[positive_mask] = target[positive_mask] + torch.broadcast_to(extra_pos, target.shape)[positive_mask]
        target[negative_mask] = target[negative_mask] + torch.broadcast_to(extra_neg, target.shape)[negative_mask]

    # make sures all the entries are above zero
    # if torch.all(target>0):
    #     pass
    # else:
    #     mask_calib = target <= 0
    #     target[mask_calib] = target[mask_calib]+ epsilon
    # test = torch.sum(target, dim=1)
    # ttt = torch.all(target>0)
    # print(f"Flag:{ttt}")
    return target


class JensenDivLossSeg(nn.Module):
    '''
    Jensen-Shannon divergence
    '''
    def __init__(self, reduction="mean"):
        super(JensenDivLossSeg, self).__init__()
        self.kl_div = KLDivLossSeg(reduction=reduction)
        self.kl_div_log = KLDivLossSeg(reduction=reduction, log_target=True)

    def forward(self, input, target):
        '''

        :param input:
        :param target: the target prob dist
        :return:
        '''
        eps = 1e-6
        # transfer logsoftmax to softmax
        input_softmax = torch.exp(input)
        m = (input_softmax+target)/2
        log_m = torch.log(m)
        # jensen_div = (self.kl_div(input, F.softmax(m, dim=1))+self.kl_div(target, F.softmax(m, dim=1)))/2
        jensen_div = (self.kl_div_log(log_m, input) + self.kl_div(log_m, target)) / 2
        return jensen_div


class JensenDivLossSeg_old(nn.Module):
    '''
    Jensen-Shannon divergence
    '''
    def __init__(self, reduction="mean"):
        super(JensenDivLossSeg_old, self).__init__()
        self.kl_div = KLDivLossSeg(reduction=reduction)

    def forward(self, input, target):
        '''

        :param input:
        :param target: the target prob dist
        :return:
        '''
        eps = 1e-6
        # transfer logsoftmax to softmax
        input_softmax = torch.exp(input)
        m = (input_softmax+target)/2
        # log_m = torch.log(m)
        target = torch.add(target, eps)
        # input = torch.log(input)
        target = torch.log(target)
        jensen_div = (self.kl_div(input, F.softmax(m, dim=1))+self.kl_div(target, F.softmax(m, dim=1)))/2
        # jensen_div = (self.kl_div(log_m, input) + self.kl_div(log_m, target)) / 2
        return jensen_div


class EntropyWeightedKLDivLossSeg(KLDivLoss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", eps=1e-3):
        super(EntropyWeightedKLDivLossSeg, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.eps = eps

    def forward(self, input, target):
        pixel_wise_kl = F.kl_div(input, target, reduction="none")
        # Add along the distribution channel
        kl = torch.sum(pixel_wise_kl, dim=1)
        # Calculate the entropy of each data point
        entropy_dist = entropy(target, dim=1, eps=self.eps)
        # weighted the loss with pixel-wise entropy
        weighted_kl = entropy_dist*kl
        # reduce dimention by batch and image dimensions
        if self.reduction == "mean":
            mean_kl = torch.mean(weighted_kl)
            return mean_kl
        elif self.reduction == "sum":
            sum_kl = torch.sum(weighted_kl)
            return sum_kl
        else:
            return weighted_kl


class FGKLDivLossSeg(KLDivLoss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", eps=1e-3):
        super(FGKLDivLossSeg, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.eps = eps

    def forward(self, input, target):
        pixel_wise_kl = F.kl_div(input, target, reduction="none")
        # Add along the distribution channel
        kl = torch.sum(pixel_wise_kl, dim=1)
        # Calculate the probability of the foreground pixels
        probs = torch.sum(target[:, 1:, ...], dim=1)
        # weighted the loss with pixel-wise probability
        weighted_kl = probs*kl
        # reduce dimention by batch and image dimensions
        if self.reduction == "mean":
            mean_kl = torch.mean(weighted_kl)
            return mean_kl
        elif self.reduction == "sum":
            sum_kl = torch.sum(weighted_kl)
            return sum_kl
        else:
            return weighted_kl


def entropy(input, dim=1, eps=1e-3):
    '''
    Calculate the pixel-wise entropy of the input
    :param input: type torch tensor of shape N*C*d1*d2*d3....
    :param eps:
    :return: pixel-wise entropy
    '''
    x = torch.add(input, eps)
    flag = torch.all(torch.gt(x, 0))
    if flag != True:
        a = 1
    assert flag == True
    logx = torch.log(x)
    entropy = torch.sum(-x*logx, dim=dim)
    return entropy


def mask_cross_entropy(y_true, y_pred, alpha, mask=None):
    '''
    calculate the croos-entropy loss function with mask
    :param y_true: tensor, size=N*D*H*W
    :param y_pred: tensor, size=N*class_n*D*H*W
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    log_prob = F.log_softmax(y_pred, dim=1)
    prob = F.softmax(y_pred, dim=1)
    shape = y_pred.shape
    y_true_tensor = onehot(y_true, shape[1])
    # loss = torch.cuda.FloatTensor([0])
    loss = 0
    for i in range(shape[1]):
        y_task = y_true_tensor[:,i,...]
        y_prob = log_prob[:, i, ...]
        if torch.is_tensor(mask):
            loss += torch.mean(-y_task * y_prob * mask)*alpha[i]
        else:
            loss += torch.mean(-y_task * y_prob)*alpha[i]
    return loss


def mask_cross_entropy_3d(y_true_dist, y_pred_logprob, mask=None):
    '''
    calculate the cross-entropy loss function with mask
    :param y_true: tensor, size=N*C*H*W*D
    :param y_pred: tensor, size=N*C*H*W*D
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    loss = 0
    shape = list(y_pred_logprob.size())
    for i in range(shape[1]):
        y_task = y_true_dist[:, i, ...]
        y_prob = y_pred_logprob[:, i, ...]
        if mask is not None:
            loss += torch.mean(-y_task * y_prob * mask)
        else:
            loss += torch.mean(-y_task * y_prob)
    return loss


def mask_focal_loss(y_true, y_pred, alpha, gamma=0, mask=None):
    '''
    calculate the croos-entropy loss function with mask
    :param y_true: tensor, size=N*D*H*W
    :param y_pred: tensor, size=N*class_n*D*H*W
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    log_prob = F.log_softmax(y_pred, dim=1)
    prob = F.softmax(y_pred, dim=1)
    shape = y_pred.shape
    y_true_tensor = onehot(y_true, shape[1])
    # loss = torch.cuda.FloatTensor([0])
    loss = 0
    assert isinstance(alpha, list)
    alpha = torch.Tensor(alpha)
    # gamma = torch.Tensor(gamma)
    for i in range(shape[1]):
        y_task = y_true_tensor[:,i,...]
        y_prob = log_prob[:, i, ...]
        focal_weight = (1-prob)**gamma
        if torch.is_tensor(mask):
            loss += torch.mean(-y_task * y_prob * mask*focal_weight)*alpha[i]
        else:
            loss += torch.mean(-y_task * y_prob*focal_weight)*alpha[i]
    return loss


def onehot(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W
    :param class_n:
    :return:N*n_class*H*W
    '''
    shape = input.shape
    onehot = torch.zeros((class_n,)+shape)
    # onehot = torch.zeros((class_n,) + shape)
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    onehot_trans = onehot.permute(1,0,2,3)
    return onehot_trans


def onehot3d(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W*D
    :param class_n:
    :return:N*n_class*H*W*D
    '''
    shape = input.shape
    # onehot = torch.zeros((class_n,)+shape).cuda()
    onehot = torch.zeros((class_n,) + shape)
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    onehot_trans = onehot.permute(1,0,2,3,4)
    return onehot_trans


# one-hot encoding method(efficient)
def onehot_encoding(array, class_num):
    '''
    the function turn a regular array to a one-hot representation form
    :param array: input array
    :param class_num: number of classes
    :return: one-hot encoding of array
    '''
    label_one_hot = np.zeros(array.shape + (class_num,), dtype="int16")
    for k in range(class_num):
        label_one_hot[..., k] = (array == k)
    return label_one_hot


def filter3D(image, kernel):
    '''
    do 3D convolution to an input ndarray
    :param image: 3d volume data W*H*D
    :param kernel: 2d filter
    :return: convolved result
    '''
    shape = image.shape
    convled_list = []
    for i in range(shape[0]):
        convolve = cv.filter2D(image[i, ...], -1, kernel)
        convled_list.append(convolve)
    out = np.array(convled_list)
    return out


def Canny3D(image, thresh1, thresh2):
    '''
    apply canny algorithm in a 3D fashion
    :param image:
    :param thresh1:
    :param thresh2:
    :return:
    '''
    shape = image.shape
    convled_list = []
    for i in range(shape[0]):
        convolve = cv.Canny(image[i, ...], threshold1=thresh1, threshold2=thresh2)
        convled_list.append(convolve)
    out = np.array(convled_list)
    out = out/255
    return out


def normalize(data):
    max = np.max(data)
    min = np.min(data)
    normalized = (data-min)/(max-min+0.00001)
    return normalized


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Evaluation(object):
    def __init__(self):
        pass

    # save 3d volume as slices
    def save_slice_img(self, volume_path, output_path):
        file_name = os.path.basename(volume_path)
        output_dir  = os.path.join(output_path, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            pass
        input_volume = nib.load(volume_path).get_data()
        # mapping to 0-1
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        input_unit = (input_volume-vol_min)/(vol_max - vol_min)
        width, height, depth= input_unit.shape
        for i in range(0, depth):
            slice_path = os.path.join(output_dir, str(i)+'.png')
            img_i = input_unit[:, :, i]
            # normalize to 0-255
            img_i = (img_i*255).astype('uint8')
            # cv.imwrite(slice_path, img_i)
        return input_unit

    def save_slice_img_label(self, img_volume, pre_volume, gt_volume,
                             output_path, file_name, show_mask=False, show_gt = False):
        assert img_volume.shape == pre_volume.shape
        if show_gt:
            assert img_volume.shape == gt_volume.shape
        width, height, depth = img_volume.shape
        # gray value mapping   from MRI value to pixel value(0-255)
        volume_max = np.max(img_volume)
        volume_min = np.min(img_volume)
        volum_mapped = (img_volume-volume_min)/(volume_max-volume_min)
        volum_mapped = (255*volum_mapped).astype('uint8')
        # construct a directory for each volume to save slices
        dir_volume = os.path.join(output_path, file_name)
        if not os.path.exists(dir_volume):
            os.makedirs(dir_volume)
        else:
            pass
        for i in range(depth):
            img_slice = volum_mapped[:, :, i]
            pre_slice = pre_volume[:, :, i]
            if show_gt:
                gt_slice = gt_volume[:, :, i]
            else:
                gt_slice = []
            self.save_contour_label(img=img_slice, pre=pre_slice, gt=gt_slice,
                                    save_path=dir_volume, file_name=i,show_mask=show_mask,show_gt=show_gt)

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def save_contour_label(self, img, pre, save_path='',color="red", file_name=None, show_mask=False):
        # single channel to multi-channel
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))

        # Generate random colors
        # colors = self.random_colors(4)
        # Prediction result is illustrated as red and the groundtruth is illustrated as blue
        colors = [[0, 1.0, 0], [0, 0, 1.0]]
        if color == "red":
            color_used = colors[0]
        elif color == "blue":
            color_used = colors[1]
        else:
            raise Exception("unkown color")
        # Show area outside image boundaries.

        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')
        # ax.set_title("volume mask")
        masked_image = img.astype(np.uint32).copy()

        if show_mask:
            masked_image = self.apply_mask(masked_image, pre, color_used)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask_pre = np.zeros(
            (pre.shape[0] + 2, pre.shape[1] + 2), dtype=np.uint8)
        padded_mask_pre[1:-1, 1:-1] = pre
        contours = find_contours(padded_mask_pre, 0.5)
        for verts in contours:
            # reduce padding and  flipping from (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color_used, linewidth=1)
            ax.add_patch(p)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        ax.imshow(masked_image.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, file_name))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def output_gif(input_path, output_path, gif_name):
    '''
    generate gif
    :param input_path: path of the input inages series
    :param output_path: gif output path
    :param gif_name: name of the gif
    :return:
    '''

    outfilename = os.path.join(output_path, '{}.gif'.format(gif_name))
    frames = []
    paths = os.listdir(input_path)
    paths_sort = sorted(paths, key=lambda x: int((os.path.splitext(x))[0]))
    for path in paths_sort:
        fullpath = os.path.join(input_path, path)
        im = imageio.imread(fullpath)
        frames.append(im)
    imageio.mimsave(outfilename, frames, 'GIF', duration=0.5)


def label_discrete2distribution(input_label, scale, stride, padding, n_class):
    '''
    This function tranfers the pixel-wise discrete label to that of label distributions
    :param input_label: pixel-wise discrete label(W,H,D...)
    :param scale: down-sampling scale
    :param n_class: number of the classes
    :return: The generated label distribution
    '''
    # We only consider 2-D and 3-D input data for image label and volume label
    shape = input_label.shape
    assert len(shape) == 2 or len(shape) == 3
    # First we pad the input_label so that the shape can be divided by the scale
    scale_tuple = np.ones(len(shape), dtype="uint8")*scale
    sub = np.mod(shape, scale_tuple)
    '''
    if np.any(sub):
        if len(shape)==2:
            gap = scale_tuple - sub
            rem_h = gap[0]%2
            rem_w = gap[1]%2
            pad_h = (gap[0]//2, gap[0]//2+ rem_h)
            pad_w = (gap[1]//2, gap[1]//2+ rem_w)
            n_pad = (pad_h, pad_w)
        elif len(shape)==3:
            gap = scale_tuple - sub
            rem_h = gap[0] % 2
            rem_w = gap[1] % 2
            rem_d = gap[2] % 2
            pad_h = (gap[0] // 2, gap[0] // 2 + rem_h)
            pad_w = (gap[1] // 2, gap[1] // 2 + rem_w)
            pad_d = (gap[2]//2, gap[2]//2 + rem_d)
            n_pad = (pad_h, pad_w, pad_d)
        else:
            raise Exception("Data Dimension Error!")
        padded_label = np.pad(input_label, pad_width=n_pad, mode="edge")
    else:
        padded_label = input_label    
   '''
    # Try to pad at the right side
    if np.any(sub):
        if len(shape)==2:
            gap = scale_tuple - sub
            pad_length = np.mod(gap, scale_tuple)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            n_pad = (pad_h, pad_w)
        elif len(shape)==3:
            gap = scale_tuple - sub
            pad_length = np.mod(gap, scale_tuple)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            pad_d = (0, pad_length[2])
            n_pad = (pad_h, pad_w, pad_d)
        else:
            raise Exception("Data Dimension Error!")
        padded_label = np.pad(input_label, pad_width=n_pad, mode="edge")
    else:
        padded_label = input_label

    # transfer numpy to tensor
    label_tensor = torch.from_numpy(padded_label)
    label_tensor = torch.unsqueeze(label_tensor, dim=0)
    if len(shape)==2:
        label_onehot = onehot(label_tensor, n_class)
        onehot_trans = label_onehot.permute(1, 0, 2, 3)
        # kernel = torch.ones(1, 1, scale, scale).cuda()
        kernel = torch.ones(1, 1, scale, scale)
        result = F.conv2d(onehot_trans, kernel/(scale*scale), stride=stride, padding=padding)
    elif len(shape)==3:
        label_onehot = onehot3d(label_tensor, n_class)
        # We exchange the batch and channel axis to conv the volume channel-wise
        onehot_trans = label_onehot.permute(1, 0, 2, 3, 4)
        # kernel = torch.ones(1, 1, scale,scale,scale).cuda()
        kernel = torch.ones(1, 1, scale, scale, scale)
        result = F.conv3d(onehot_trans, kernel/(scale*scale*scale), stride = stride, padding=padding)
    else:
        raise Exception("Data Dimension Error!")

    data = torch.squeeze(result, dim=1)
    # out = data.data.cpu().numpy()

    return data


def label_discrete2GaussianDist(input_label, n_class):
    '''
    This function tranfers the pixel-wise discrete label to that of label distributions via existing Gaussian filtering
    :param input_label: pixel-wise discrete label(W,H,D...)
    :param n_class: number of the classes
    :return: The generated label distribution
    '''
    # We only consider 2-D and 3-D input data for image label and volume label
    shape = input_label.shape
    assert len(shape) == 2 or len(shape) == 3

    # transfer numpy to tensor
    label_tensor = torch.from_numpy(input_label)
    label_tensor = torch.unsqueeze(label_tensor, dim=0)
    if len(shape)==2:
        label_onehot = onehot(label_tensor, n_class)
        smoothing = GaussianSmoothing(n_class, 5, 0.5, dim=2)
        input = F.pad(label_onehot, (2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
        result = output / torch.sum(output, dim=1)
    
    elif len(shape)==3:
        label_onehot = onehot3d(label_tensor, n_class)
        smoothing = GaussianSmoothing(n_class, 5, 0.5, dim=3)
        input = F.pad(label_onehot, (2, 2, 2, 2, 2, 2), mode='reflect')
        output = smoothing(input)
        result = output / torch.sum(output, dim=1)
    else:
        raise Exception("Data Dimension Error!")

    data = torch.squeeze(result, dim=0)

    return data


class GaussianSmoothing(nn.Module):
    """
    Implementation adopd from "https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8"
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def downsample_image(input, scale):
    '''
    Downsample the input image.
    :param input: Input image, H*W*C  D*H*W*C
    :param scale: scale factor, dtype of int
    :return:
    '''
    # Pad the input image if the input dimensions are not perfectly divisible by the
    # scale factors
    shape = input.shape
    assert len(shape) == 3 or len(shape) == 4
    scale_tuple = (scale, )*(len(shape)-1)+(1,)
    scale_array = np.array(scale_tuple, dtype="uint8")
    sub = np.mod(shape, scale_array)
    # Try to pad at the right side
    if np.any(sub):
        if len(shape) == 3:
            gap = scale_array - sub
            pad_length = np.mod(gap, scale_array)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            n_pad = (pad_h, pad_w, (0, 0))
        elif len(shape) == 4:
            gap = scale_array - sub
            pad_length = np.mod(gap, scale_array)
            pad_h = (0, pad_length[0])
            pad_w = (0, pad_length[1])
            pad_d = (0, pad_length[2])
            n_pad = (pad_h, pad_w, pad_d, (0, 0))
        else:
            raise Exception("Data Dimension Error!")
        padded_input = np.pad(input, pad_width=n_pad, mode="edge")
    else:
        padded_input = input

    img_donwsample = downscale_local_mean(padded_input, scale_tuple)
    return img_donwsample


def euclidean(x, y):
    '''
    Pixel-vise euclidean distance,
    :param x: N*C*D*W*H
    :param y: N*C*D*W*H
    :return:
    '''
    return np.sqrt(np.sum(np.power((x-y), 2), axis=1))


def euclidean_mean(x,y):
    euclidean_vise = euclidean(x,y)
    mean_euclidean = np.mean(euclidean_vise)
    return mean_euclidean


def save_prediction_niftil(out_path, file_name, affine, prediction):
    '''
    save the segmentation result to a specific path
    :param out_path:
    :param file_name:
    :param affine:
    :param prediction:
    :return:
    '''
    c_map_path = os.path.join(out_path, file_name+".nii.gz")
    label_volume = nib.Nifti1Image(prediction, affine)
    nib.save(label_volume, c_map_path)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def postprocessing_brats(prediction, threshold=1):
    '''
   This function replace all enhancing tumor voxels with necrosis if the total
    number of predicted enhancing tumor is less than some threshold.
    :param prediction: prediction: predicted result of the data volume, label {0，1，2，4}
    :param threshold:
    :return:
    '''
    # get the number of enhancing tumor pixels
    num_et = np.sum((prediction==4))
    if num_et<threshold:
        prediction[prediction==4] = 1
    return prediction


def postprocess_main(path):
    out_path = "submitNew"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    file_paths = glob(f'{path}/*.nii.gz')

    for file in file_paths:
        data = nib.load(file)
        ref_affine = data.affine
        data_array = data.get_data()
        out_data = postprocessing_brats(data_array, threshold=50)
        label_volume = nib.Nifti1Image(out_data, ref_affine)
        nib.save(label_volume, os.path.join(out_path, os.path.basename(file)))


def fillhole_main3D(path):
    out_path = "submit_fillhole3D"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    file_paths = glob(f'{path}/*.nii.gz')

    for file in file_paths:
        data = nib.load(file)
        ref_affine = data.affine
        input_volume = data.get_data()
        _,_, slices = input_volume.shape
        out_volume = np.zeros(input_volume.shape, dtype="uint8")

        # mask_1 and mask_4
        mask1 = input_volume == 1
        mask4 = input_volume == 4
    
        edma_mask = input_volume == 2
        filled_out = binary_fill_holes(edma_mask.astype("uint8"))
        out_volume[filled_out] = 2

        out_volume[mask1] = 1
        out_volume[mask4] = 4
        label_volume = nib.Nifti1Image(out_volume, ref_affine)
        nib.save(label_volume, os.path.join(out_path, os.path.basename(file)))

    return out_volume


def fillhole_main2D(path):
    '''
    slice-wise hole filling !
    '''

    out_path = "submit_fillhole"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    file_paths = glob(f'{path}/*.nii.gz')

    for file in file_paths:
        data = nib.load(file)
        ref_affine = data.affine
        input_volume = data.get_data()
        _,_, slices = input_volume.shape
        out_volume = np.zeros(input_volume.shape, dtype="uint8")

        # mask_1 and mask_4
        mask1 = input_volume == 1
        mask4 = input_volume == 4
        for i in range(slices):
            slice_i = input_volume[:,:,i]
            slice_i_clone = copy.deepcopy(slice_i)
            edma_mask = slice_i == 2
            filled_out = binary_fill_holes(edma_mask.astype("uint8"))
            slice_i_clone[filled_out] = 2
            out_volume[:,:,i] = slice_i_clone

        out_volume[mask1] = 1
        out_volume[mask4] = 4
        label_volume = nib.Nifti1Image(out_volume, ref_affine)
        nib.save(label_volume, os.path.join(out_path, os.path.basename(file)))

    return out_volume


def find_postprocess_threshold_brats(predict_path, gt_path, threshold_range=(0,2000)):
    '''
    This function find the threshold used for postprocessing by optimizing the mean Dice (using
    the above mentioned convention) on the BraTS 2018 training cases.
    :param predict_path:  data_path of the predicted path
    :param gt_path: data_path of the groundtruth
    :param threshold_range: Range of the threshold
    :return:
    '''
    with open("../threshold.txt", "a") as log:
        step = 20
        mean_dice_max = 0
        out_threshold = 0
        for i in range(threshold_range[0], threshold_range[1], step):
            dice_et_list = []
            dice_tc_list = []
            dataloader = CTData(gt_path)
            data_lists = dataloader.data_list
            for data_list in data_lists:
                single_dir_path = data_list["path"]
                base = os.path.basename(single_dir_path)
                _, gt, _ = dataloader.load_volumes_label(single_dir_path, False)
                if data_list["category"] == "HGG":
                    predict_single_path = os.path.join(predict_path, "hgg/HGG", base+".nii.gz")
                elif data_list["category"] == "LGG":
                    predict_single_path = os.path.join(predict_path, "lgg/LGG", base+".nii.gz")

                predict = nib.load(predict_single_path).get_data()
                predicted_new = postprocessing_brats(predict, i)

                # tumor core
                gt[gt == 2] = 0
                predicted_new[predicted_new == 2] = 0
                TC_target = gt > 0
                TC_predict = predicted_new > 0
                dsc_tc = dice(TC_predict, TC_target, 1)

                # Enhancing tumor
                gt[gt == 1] = 0
                predicted_new[predicted_new == 1] = 0
                ET_target = gt > 0
                ET_predict = predicted_new > 0
                dsc_et = dice(ET_predict, ET_target, 1)
                dice_et_list.append(dsc_et)
                dice_tc_list.append(dsc_tc)

            mean_dice_et = np.mean(np.array(dice_et_list))
            mean_dice_tc = np.mean(np.array(dice_tc_list))
            mean_dice = (mean_dice_et+mean_dice_tc)/2
            if mean_dice>mean_dice_max:
                out_threshold = i
                mean_dice_max = mean_dice
            log.write(f"Threshold:{i}  tc:{mean_dice_tc}  et:{mean_dice_et}  mean:{mean_dice}")
    print(f"Best_threshold:{out_threshold}")
    return out_threshold


def find_threshold_range_brats(path):
    '''
    determine the threshold range for the post-processing process
    :param path:
    :return:
    '''
    with open("et_count.txt", "a") as txtlog:
        num_list = []
        dataloader = CTData(path)
        data_lists = dataloader.data_list
        for data_list in data_lists:
            single_dir_path = data_list["path"]
            _, label_data, _ = dataloader.load_volumes_label(single_dir_path, False)
            num_et = np.sum((label_data == 4))
            num_list.append(num_et)
            txtlog.write(str(num_et)+"\n")
        count = np.array(num_list)
        plt.hist(count, bins=40)
        plt.show()
        a = 1


def find_threshold_range(path):
    # range:0-2000
    with open("et_count_predict.txt", "a") as txtlog:
        num_list = []
        data_lists = glob(pathname=os.path.join(path, "*.nii.gz"))
        for data_list in data_lists:
            label_data = nib.load(data_list).get_data()
            num_et = np.sum((label_data == 4))
            num_list.append(num_et)
            txtlog.write(str(num_et)+"\n")
        count = np.array(num_list)
        plt.hist(count, bins=40)
        plt.show()


def test_postprocessing(path):
    '''
    Test the num of the class 4 in the predictions
    '''
    file_paths = glob(f'{path}/*.nii.gz')
    subject_list = []
    count_list = []
    for file in file_paths:
        data = nib.load(file)
        data_array = data.get_data()
        label_4 = data_array==4
        label_4_count = np.sum(label_4)
        subject_list.append(os.path.basename(file))
        count_list.append(label_4_count)
    
    data_frame = pd.DataFrame({"subject": subject_list, "count": count_list})
    data_frame.to_csv("postprocessing.csv")



if __name__ == "__main__":
    
    # input = torch.rand(96, 96, 96)
    a = [[0,0,1,0],[1,0,0,0]]
    a_tensor = torch.from_numpy(np.array(a))
    en = entropy(a_tensor)


    input = np.random.randint(0, 4, (96, 96, 96))
    ddd = label_discrete2GaussianDist(input, 4)
    test1 = ddd[:,15,0,0]
    test2 = ddd[:,48,48,48]


    # postprocess_main(data_path)
    # test_postprocessing(data_path)
    # data_path_predict = "../MultiScaleAttentionNoNewNet_trainingset"
    # find_postprocess_threshold_brats(data_path_predict, data_path)
    # input = torch.ones(1,1,9,9)
    # input2 = torch.ones(1,1,9,9)*2
    # dd = torch.cat((input, input2), dim=0)
    # filter = torch.ones(1,1,3,3)
    # a = F.conv2d(dd, filter, stride=3, padding=0)
    # f = a.data.numpy()
    # x = np.random.randint(0, 4, (491,491, 490))
    # trans = label_discrete2distribution(x, scale=7, stride=7, padding=0, n_class=4)
    # y = np.random.randn(480, 490, 494, 3)
    # fdf = downsample_image(y, 7)
    # input = torch.randn(1,4,96,96,96, requires_grad=True)
    # target = torch.randn(1,4,96,96,96)
    # loss = KLDivLossSeg(reduction="mean")
    # loss2 = WeightedKLDivLossSeg(reduction="mean", weight=[0.1,0.2,0.3,0.4])
    # loo_re = loss(input, target)
    # loss2_re = loss2(input, target)
    # loo_re.backward()
    v = 1
