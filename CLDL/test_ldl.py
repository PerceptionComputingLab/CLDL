import torch.utils.data as dataloader
import torch.optim as optim
from imgaug import augmenters as iaa
import numpy as np
import torch
import torch.nn as nn
from config import Config
from models.unet_3d import Unet3D, LDLESPCN, LDLSR
from miscellaneous.metrics import dice
from miscellaneous.logger import Logger
from miscellaneous.utils import KLDivLossSeg, save_prediction_niftil
from data.datagenerator import CTData, BratsDataDist, get_brain_region
from skimage.transform import resize, downscale_local_mean
import os


# on line evaluate the result
def evaluation(model, dataloader, config):
    # 该函数实现对于体数据的指标评价（dice系数）
    xstep = config.step_size[0]
    ystep = config.step_size[1]
    zstep = config.step_size[2]
    num = 0
    with torch.no_grad():
        dice_new_list = []
        dice_new_list1 = []
        for data_val in dataloader:
            images_val, targets_val, _, _ = data_val
            model.eval()
            _, _, W, H, C = images_val.size()
            whole_pred = np.zeros((1,) + (config.num_classes,W, H, C))
            count_used = np.zeros((W, H, C)) + 1e-5

            c_slice = np.arange(0, C - config.crop_size[2] + zstep, zstep)
            w_slice = np.arange(0, W - config.crop_size[0] + xstep, xstep)
            h_slice = np.arange(0, H - config.crop_size[1] + ystep, ystep)
            for i in range(len(c_slice)):
                for j in range(len(w_slice)):
                    for k in range(len(h_slice)):
                        depth = c_slice[i]
                        width = w_slice[j]
                        height = h_slice[k]

                        # 超出边界处理
                        if width+config.crop_size[0]>W:
                            width = W-config.crop_size[0]
                        if height+config.crop_size[1]>H:
                            height = H-config.crop_size[1]
                        if depth+config.crop_size[2]>C:
                            depth = C-config.crop_size[2]

                        image_input = images_val[:, :, width:width + config.crop_size[0],
                                      height:height + config.crop_size[1], depth:depth+config.crop_size[2]].to(
                            device)
                        # Downsampling the original data
                        image_array = image_input.data.cpu().numpy()
                        img_downsample = downscale_local_mean(image_array, factors=((1,)*2+(config.down_scale,) * 3 ))
                        img_downsample = torch.from_numpy(img_downsample).cuda()
                        out_dist, output_patch = model(img_downsample, img_downsample)
                        whole_pred[:, :, width:width + config.crop_size[0],height:height + config.crop_size[1],
                        depth:depth + config.crop_size[2]] += output_patch.data.cpu().numpy()

                        count_used[width:width + config.crop_size[0],
                        height:height + config.crop_size[1],depth:depth+config.crop_size[2]] += 1

            whole_pred = whole_pred / count_used
            whole_pred = whole_pred[0, :, :, :, :]
            whole_pred = np.argmax(whole_pred, axis=0)
            whole_target = targets_val[0, ...].data.cpu().numpy()

            dsc_list1 = []
            for i in range(0, config.num_classes):
                dsc_i = dice(whole_pred, whole_target, i)
                dsc_list1.append(dsc_i)
            dice_new_list1.append(dsc_list1)

            # calculate clinical-specific metrics
            # whole tumor
            WT_target = whole_target>0
            WT_predict = whole_pred>0

            dsc_bg = dice(WT_predict, WT_target,0)
            dsc_wt = dice(WT_predict, WT_target,1)

            # tumor core
            whole_target[whole_target==2] = 0
            whole_pred[whole_pred==2] = 0
            TC_target = whole_target > 0
            TC_predict = whole_pred > 0
            dsc_tc = dice(TC_predict, TC_target, 1)

            # Enhancing tumor
            whole_target[whole_target == 1] = 0
            whole_pred[whole_pred == 1] = 0
            ET_target = whole_target>0
            ET_predict = whole_pred>0
            dsc_et = dice(ET_predict, ET_target, 1)

            # print("bg:{:3f} WH:{:3f} TC:{:3f} ET:{:3f}".format(dsc_list1[0],
            #                                                    dsc_list1[1],dsc_list1[2],dsc_list1[3]))
            # txtlog.write("ID{:30} {:3f}{:3f} {:3f} {:3f} \n".format(num, dsc_list1[0],
            #                                                         dsc_list1[1],dsc_list1[2],dsc_list1[3]))
            dice_new_list.append([dsc_bg, dsc_wt, dsc_tc, dsc_et])
            num += 1

        dice_array = np.array(dice_new_list)
        dice_mean = np.mean(dice_array, axis=0)

        # txtlog.write("Dice_mean bg|WH|TC|ET|: {:3f}{:3f} {:3f} {:3f}\n".format(
        #     dice_mean[0], dice_mean[1],dice_mean[2], dice_mean[3]))

    return dice_mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)

if __name__ == '__main__':
    # write and display config information
    log = Logger("runs", write=False, save_freq=4)
    config = Config()
    log.log(config.get_str_config())
    data_path = config.data_path
    best_dice_train = 0
    best_dice_val = 0

    # define the main model
    model_S = LDLESPCN(input_channel=config.modalities, scale_factor=config.down_scale).to(device)
    # model_S = LDLSR(input_channel=config.modalities, scale_factor=config.down_scale).to(device)
    # load models
    print("start load checkpoint: {}".format(config.note))
    checkpoint = torch.load(config.checkpoint)
    model_S.load_state_dict(checkpoint)
    print("load checkpoint succeed!")

    ct_data = CTData(data_path, mode="validate", modalities=4)
    data_list = ct_data.data_list

    for i, file_path_dict in enumerate(data_list):
        file_path = file_path_dict["path"]
        img_array, label_data, volume = ct_data.load_volumes_label(file_path, True)
        img_shape = img_array.shape
        ref_affine = volume.affine
        # ref_affine = np.eye(4)
        regions = get_brain_region(img_array[:, :, :, 2])
        img_data = img_array[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
        data1_norm = ct_data.Normalization(img_data, axis=(0, 1, 2))
        images_val = torch.from_numpy(data1_norm).float().permute(3, 0,1,2)
        images_val = torch.unsqueeze(images_val, dim=0)
        xstep = config.step_size[0]
        ystep = config.step_size[1]
        zstep = config.step_size[2]
        num = 0
        with torch.no_grad():
            model_S.eval()
            _, _, W, H, C = images_val.size()
            whole_pred = np.zeros((1,) + (config.num_classes, W, H, C))
            count_used = np.zeros((W, H, C)) + 1e-5
            c_slice = np.arange(0, C - config.crop_size[2] + zstep, zstep)
            w_slice = np.arange(0, W - config.crop_size[0] + xstep, xstep)
            h_slice = np.arange(0, H - config.crop_size[1] + ystep, ystep)
            for i in range(len(c_slice)):
                for j in range(len(w_slice)):
                    for k in range(len(h_slice)):
                        depth = c_slice[i]
                        width = w_slice[j]
                        height = h_slice[k]
                        # 超出边界处理
                        if width + config.crop_size[0] > W:
                            width = W - config.crop_size[0]
                        if height + config.crop_size[1] > H:
                            height = H - config.crop_size[1]
                        if depth + config.crop_size[2] > C:
                            depth = C - config.crop_size[2]

                        image_input = images_val[:, :, width:width + config.crop_size[0],
                                      height:height + config.crop_size[1], depth:depth + config.crop_size[2]].to(
                            device)
                        # Downsampling the original data
                        image_array = image_input.data.cpu().numpy()
                        img_downsample = downscale_local_mean(image_array,
                                                              factors=((1,) * 2 + (config.down_scale,) * 3))
                        img_downsample = torch.from_numpy(img_downsample).cuda()
                        out_dist, output_patch = model_S(img_downsample, img_downsample)
                        whole_pred[:, :, width:width + config.crop_size[0], height:height + config.crop_size[1],
                        depth:depth + config.crop_size[2]] += output_patch.data.cpu().numpy()

                        count_used[width:width + config.crop_size[0],
                        height:height + config.crop_size[1], depth:depth + config.crop_size[2]] += 1

            whole_pred = whole_pred / count_used
            whole_pred = whole_pred[0, :, :, :, :]
            whole_pred = np.argmax(whole_pred, axis=0)
            whole_pred[whole_pred == 3] = 4

            # add background(non_brain area)
            whole_pred_new = np.zeros(shape=img_shape[:-1], dtype="int16")
            whole_pred_new[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]] = whole_pred
            name = os.path.basename(file_path)
            out_path = "submit"
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            save_prediction_niftil(out_path, name, ref_affine, whole_pred_new)
            num += 1



