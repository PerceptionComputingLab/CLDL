import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from config import Config
from models.unet_3d import UNetLDL, UNetLDLWithoutPPA
from miscellaneous.metrics import dice
from miscellaneous.logger import Logger
from miscellaneous.utils import KLDivLossSeg, JensenDivLossSeg, SoftDiceLoss
from data.data import BratsDataloaderNnunet, get_train_transform, get_img_info,train_validate_split
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from miscellaneous.utils import label_discrete2distribution, downsample_image, entropy, label_smooth
import time
import os
import copy
import torch.nn.functional as F
# from ttest import test_ldl_gt_predict


def log_tenserboard(info):
    structure = ["DSC_bg", "DSC_wc", "DSC_tc", "DSC_et"]
    board_info= {structure[i]:info[i] for i in range(len(info))}
    return board_info


def Normalization(volume):
    '''
    Volume shape ： W*H*D*C
    :param volume:
    :param axis:
    :return:
    '''
    batch, c, _,_,_, = volume.shape
    bg_mask = volume == 0
    mean_arr = np.zeros(c, dtype="float32")
    std_arr = np.zeros(c, dtype="float32")
    norm_volume = copy.deepcopy(volume.transpose(0, 2, 3, 4, 1))
    for j in range(batch):
        for i in range(c):
            data = volume[j, i, ...]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std
        norm_volume[j] = (volume[j].transpose(1, 2, 3, 0) - mean_arr) / std_arr

    norm_volume = norm_volume.transpose(0,4,1,2,3)
    norm_volume[bg_mask] = 0

    return norm_volume


def pre_processing(data_dict, scale, stride, padding, n_class):
    '''
    transfer numpy data to that of tensor
    :param data:
    :return:
    '''
    img_array = data_dict["data"]
    label_array = data_dict["seg"]
    label_array = label_array.astype("int16")
    img_normed = Normalization(img_array)
    batch, _,_,_,_ = img_array.shape
    label_dist_list1 = []
    entropy_list1 = []
    label_dist_list2 = []
    entropy_list2 = []
    label_dist_list3 = []
    entropy_list3 = []
    for i in range(batch):
        label_dist1 = label_discrete2distribution(label_array[i, 0, ...], scale*4, stride*4, padding, n_class)
        label_dist2 = label_discrete2distribution(label_array[i, 0, ...], scale*2, stride*2, padding, n_class)
        label_dist3 = label_discrete2distribution(label_array[i, 0, ...], scale, stride, padding, n_class)

        label_entropy1 = entropy(torch.unsqueeze(label_dist1, dim=0))
        label_entropy2 = entropy(torch.unsqueeze(label_dist2, dim=0))
        label_entropy3 = entropy(torch.unsqueeze(label_dist3, dim=0))

        label_dist_list1.append(label_dist1)
        entropy_list1.append(label_entropy1)
        label_dist_list2.append(label_dist2)
        entropy_list2.append(label_entropy2)
        label_dist_list3.append(label_dist3)
        entropy_list3.append(label_entropy3)

    label_dist_out1 = torch.stack(label_dist_list1, dim=0)
    label_entropy_out1 = torch.stack(entropy_list1, dim=0)
    label_dist_out2 = torch.stack(label_dist_list2, dim=0)
    label_entropy_out2 = torch.stack(entropy_list2, dim=0)
    label_dist_out3 = torch.stack(label_dist_list3, dim=0)
    label_entropy_out3 = torch.stack(entropy_list3, dim=0)

    out_label_dist_entropy = {'aux1': [label_dist_out1, label_entropy_out1], 'aux2': [label_dist_out2,label_entropy_out2],
                              'aux3': [label_dist_out3, label_entropy_out3]}
    return (torch.from_numpy(img_normed).float(),
            torch.from_numpy(label_array[:,0,...]).long(),
            out_label_dist_entropy
            )


class Trainer(object):
    '''
    Trainer Class
    '''
    def __init__(self, debug, config, model, criterion, optimizer, train_data_loader,
                 val_data_loader=None, lr_schedule=None, log=None):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.lr_schedule = lr_schedule
        self.debug = debug
        self.do_validation = self.val_data_loader is not None
        self.scaler = [] # AMP test
        self.log = log

    def training(self):
        # Test Automatic Mixed Precision training(AMP)
        self.scaler = torch.cuda.amp.GradScaler()
        # iteration = 0
        best_dice_train = 0
        best_dice_val = 0
        best_loss_val = 1000
        best_train_num = -1
        best_val_num = -1
        best_val_loss_num = -1
        for epoch in range(self.config.num_epoch):
            # training iteration
            epoch_start_time = time.time()
            loss_train = self.train_epoch(epoch)
            epoch_train_end = time.time()
            # validation iteration
            if self.do_validation:
                dice_val, loss_val = self.validate_epoch(epoch)
            evaluation_time = time.time()
            if self.debug:
                print(f"evaluation:{(evaluation_time-epoch_train_end):8.2f}")
            # save training information
            self.log.log('%0.6f | %6d | %0.5f |%0.5f |%0.5f | %0.5f| %0.5f| %0.5f| %0.5f| %0.5f ' % (
                self.optimizer.state_dict()["param_groups"][0]['lr'], epoch,
                loss_train[0].data.cpu().numpy(), loss_train[1].data.cpu().numpy(), loss_train[2].data.cpu().numpy(),
                loss_train[3].data.cpu().numpy(),
                dice_val[0], dice_val[1], dice_val[2], dice_val[3]))
            board_info = log_tenserboard(dice_val)
            self.log.write_to_board(f"Validation/score", board_info, epoch)
            # board_info_train = log_tenserboard(dice_train_val)
            self.log.write_to_board(f"Val_Loss", {"loss": loss_val[0], "loss_ldl1": loss_val[1],"loss_ldl2": loss_val[2], "loss_ce": loss_val[3],
                                         "loss_dice": loss_val[4]}, epoch)
            self.log.write_to_board(f"Train_Loss", {"loss": loss_train[0], "loss_ldl1": loss_train[1],"loss_ldl2": loss_val[2], "loss_ce": loss_train[3],
                                         "loss_dice": loss_train[4]}, epoch)

            # save checkpoints
            if epoch % self.config.step_size_S == 0:
                self.log.save_model(self.model.state_dict(), f"checkpoint_{epoch}.pt", forced=True)

            # save best checkpoint
            if np.mean(dice_val[1:]) > best_dice_val:
                best_checkpoint_path = os.path.join(self.log.dir, f"best_checkpoint_{best_val_num}.pt")
                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                best_val_num = epoch
                self.log.save_model(self.model.state_dict(), f"best_checkpoint_{best_val_num}.pt", forced=True)
                best_dice_val = np.mean(dice_val[1:])
            if loss_val[0]<best_loss_val:
                best_loss_checkpoint_path = os.path.join(self.log.dir, f"best_checkpoint_loss_{best_val_loss_num}.pt")
                if os.path.exists(best_loss_checkpoint_path):
                    os.remove(best_loss_checkpoint_path)
                best_val_loss_num = epoch
                self.log.save_model(self.model.state_dict(), f"best_checkpoint_loss_{best_val_loss_num}.pt", forced=True)
                best_loss_val = loss_val[0]
            # save the latest checkpoint
            self.log.save_model(self.model.state_dict(), "latest_checkpoint.pt", forced=True)
            # learning rate decay
            self.lr_schedule.step()
            epoch_end_time = time.time()
            if self.debug:
                print(f"model_save_time:{(epoch_end_time-evaluation_time):8.2f}")
                print(f"Epoch_whole_time:{(epoch_end_time-epoch_start_time):8.2f}")
            # print(f"Whole_time{(epoch_end_time-epoch_start_time):8.2f}--Trining_time{(epoch_train_end-epoch_start_time):8.2f}--"
            #       f"epoch_validate:{(epoch_validate_train_start-epoch_train_end):8.2f}--epoch_"
            #       f"validate_training:{(epoch_validate_train_end-epoch_validate_train_start):8.2f}--"
            #       f"epoch_save_model:{(epoch_end_time-epoch_validate_train_end):8.2f}")

    def train_epoch(self, epoch):
        self.model.train()
        loss_list = []
        # Test time
        epoch_start_time = time.time()
        for i in range(self.config.num_batches_per_epoch):
            # Clear the gradient
            self.optimizer.zero_grad()
            loss, loss_ldl1, loss_ldl2, loss_ce, loss_dice = self.forward()
            loss_list.append([loss.item(), loss_ldl1.item(), loss_ldl2.item(), loss_ce.item(), loss_dice.item()])

            # backward and optimize(AMP)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            # optimizer_S.step()
            # TODO cycling lr_rate decay
            # learning rate decay
            # scheduler_S.step(iteration)
            # iteration += 1

        loss_array = np.array(loss_list)
        loss_mean = np.mean(loss_array, axis=0)
        epoch_train_end = time.time()
        if self.debug:
            print(f"epoch_training:{(epoch_train_end-epoch_start_time):8.2f}")

        return loss_mean

    def validate_epoch(self, epoch):
        # Validation
        self.model.eval()
        dice_new_list = []
        loss_list = []
        for b in range(self.config.num_validation_batches_per_epoch):
            with torch.no_grad():
                data_out = next(self.val_data_loader)
                images, targets, target_dist_entropy = pre_processing(data_out, self.config.down_scale,
                                                                      self.config.down_scale, 0,
                                                                      self.config.num_classes)
                images_val = images.cuda()
                targets = targets.cuda()
                targets_val = targets.data.cpu().numpy()

                aux_out, sr_out = self.model(images_val)
                loss_ldl = torch.tensor(0.0).cuda()
                loss_ldl_1_all = torch.tensor(0.0).cuda()
                loss_ldl_2_all = torch.tensor(0.0).cuda()

                for item in target_dist_entropy.keys():
                    targets_dist = target_dist_entropy[item][0].cuda()
                    targets_dist_smooth = label_smooth(targets_dist, epsilon=0.001)
                    ldl1 = self.criterion[0](aux_out[item][0], targets_dist)
                    ldl2 = self.criterion[3](torch.log(targets_dist_smooth), aux_out[item][0])
                    loss_ldl_1_all = loss_ldl_1_all + ldl1
                    loss_ldl_2_all = loss_ldl_2_all + ldl2
                    loss_ldl = loss_ldl + ldl1 + ldl2

                loss_ce = self.criterion[1](sr_out, targets)
                loss_dice = self.criterion[2](F.softmax(sr_out, dim=1), targets)
                loss = self.config.lamda_ldl * loss_ldl + self.config.lamda_sr * loss_ce + loss_dice

                _, predicted = torch.max(sr_out.data, 1)
                whole_target = targets_val
                whole_pred = predicted.data.cpu().numpy()
                # calculate clinical-specific metrics
                # whole tumor
                WT_target = whole_target > 0
                WT_predict = whole_pred > 0

                dsc_bg = dice(WT_predict, WT_target, 0)
                dsc_wt = dice(WT_predict, WT_target, 1)

                # tumor core
                whole_target[whole_target == 2] = 0
                whole_pred[whole_pred == 2] = 0
                TC_target = whole_target > 0
                TC_predict = whole_pred > 0
                dsc_tc = dice(TC_predict, TC_target, 1)

                # Enhancing tumor
                whole_target[whole_target == 1] = 0
                whole_pred[whole_pred == 1] = 0
                ET_target = whole_target > 0
                ET_predict = whole_pred > 0
                dsc_et = dice(ET_predict, ET_target, 1)

                dice_new_list.append([dsc_bg, dsc_wt, dsc_tc, dsc_et])
                loss_list.append([loss, loss_ldl_1_all, loss_ldl_2_all, loss_ce, loss_dice])

        dice_array = np.array(dice_new_list)
        dice_mean = np.mean(dice_array, axis=0)
        loss_array = np.array(loss_list)
        loss_mean = np.mean(loss_array, axis=0)

        return dice_mean, loss_mean

    def forward(self, phase="train"):
        start_time = time.time()
        if phase == "train":
            dataloader = self.train_data_loader
        elif phase == "validate":
            dataloader = self.val_data_loader
        else:
            raise NotImplementedError("Check arguments!")
        # load data
        data_out = next(dataloader)
        images, targets, target_dist_entropy = pre_processing(data_out, self.config.down_scale,
                                                              self.config.down_scale, 0,
                                                              self.config.num_classes)

        # test = test_ldl_gt_predict(images, targets, target_dist_entropy['aux1'][0])
        
        pre_process_time = time.time()
        if self.debug:
            print(f"pre_process_time:{(pre_process_time-start_time):8.2f}")
        images = images.cuda()
        targets = targets.cuda()

        # Test AMP
        with torch.cuda.amp.autocast():
            aux_out, sr_out = self.model(images)
            forward_time = time.time()
            if self.debug:
                print(f"forward_time:{(forward_time-pre_process_time):8.2f}")
            loss_ldl = torch.tensor(0.0).cuda()
            loss_ldl_1_all = torch.tensor(0.0).cuda()
            loss_ldl_2_all = torch.tensor(0.0).cuda()
            cc_list = []
            for item in target_dist_entropy.keys():
                # target_entropy = torch.squeeze(target_dist_entropy[item][1], dim=1)
                # target_entropy = target_entropy.cuda()
                targets_dist = target_dist_entropy[item][0].cuda()
                # TODO epsilon==0.01 would trigger NaN bug since this would cause non-positive value in targets.
                targets_dist_smooth = label_smooth(targets_dist, epsilon=0.001)
                # loss_ldl = loss_ldl + self.criterion[0](aux_out[item][0], targets_dist)
                # TODO Test bidirectional kl_divergency loss
                # aa = torch.log(targets_dist_smooth)
                # bb = torch.exp(aux_out[item][0])
                ldl1 = self.criterion[0](aux_out[item][0], targets_dist)
                ldl2 = self.criterion[3](torch.log(targets_dist_smooth), aux_out[item][0])
                loss_ldl_1_all = loss_ldl_1_all + ldl1
                loss_ldl_2_all = loss_ldl_2_all + ldl2
                loss_ldl = loss_ldl + ldl1+ ldl2

            loss_ce = self.criterion[1](sr_out, targets)
            loss_dice = self.criterion[2](F.softmax(sr_out, dim=1), targets)
            loss = self.config.lamda_ldl * loss_ldl + self.config.lamda_sr * loss_ce + loss_dice

        return loss, loss_ldl_1_all, loss_ldl_2_all, loss_ce, loss_dice


def main():
    num_threads = 12
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = True
    config = Config()
    debug = False
    # write and display config information
    log = Logger("runs", write=config.write_log, save_freq=4)
    log.log(config.get_str_config())

    # define the main model
    model_S = UNetLDL(input_dim=config.modalities, scale_factor=config.down_scale).cuda()
    criterion_S1 = KLDivLossSeg(reduction="mean").cuda()
    criterion_S2 = KLDivLossSeg(reduction="mean", log_target=True).cuda()
    criterion_S3 = nn.CrossEntropyLoss().cuda()
    criterion_S4 = SoftDiceLoss(apply_nonlin=None, **{'batch_dice': False, 'do_bg': True, 'smooth': 0}).cuda()
    optimizer_S = optim.Adam(model_S.parameters(), lr=config.lr_S, weight_decay=1e-5, betas=(0.97, 0.999))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=config.step_size_S, gamma=config.ratio)

    # dataloader
    file_list = get_img_info(config.data_path)
    train_list, validate_list = train_validate_split(file_list, config.training_data_ratio, seed=3)
    config.num_validation_batches_per_epoch = len(validate_list)

    dataloader_train = BratsDataloaderNnunet(train_list, config.batch_train, config.crop_size, num_threads)
    dataloader_validation = BratsDataloaderNnunet(validate_list, 1, config.crop_size, max(1, num_threads // 2))
    tr_transforms = get_train_transform(config.crop_size)

    # tr_gen = SingleThreadedAugmenter(dataloader_train, transform=tr_transforms)
    # val_gen = SingleThreadedAugmenter(dataloader_validation, transform=None)
    # finally we can create multithreaded transforms that we can actually use for training
    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=num_threads,
                                    num_cached_per_queue=3, seeds=None, pin_memory=True)
    # we need less processes for validation because we don't apply transformations
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,num_processes=max(1, num_threads // 2),
                                     num_cached_per_queue=1, seeds=None,pin_memory=True)
    # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # batches while other things run in the main thread
    tr_gen.restart()
    val_gen.restart()

    trainer = Trainer(debug=debug, config=config, model=model_S, criterion=[criterion_S1,criterion_S3, criterion_S4, criterion_S2],
                      optimizer=optimizer_S, train_data_loader=tr_gen, val_data_loader=val_gen,
                      lr_schedule=scheduler_S, log=log)
    log.log("start training!")
    log.log('Rate     | epoch  | Loss seg|Loss_ldl|Loss_sr|DSC_bg| DSC_WT| DSC_TC| DSC_ET ')
    trainer.training()


if __name__ == '__main__':
    main()
