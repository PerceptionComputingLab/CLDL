import torch
import torch.backends.cudnn as cudnn

class Config(object):
    write_log = True
    drop_rate = 0
    alpha = [0.00168, 0.521, 0.268, 1.0]
    # crop_size = [32, 32, 32]
    crop_size = [128, 128, 128]
    step_size = [120,120,120]
    down_scale = 2
    modalities = 4
    training_data_ratio = 0.8
    # data_path = "/home/lixiangyu/Dataset/test_temp/brats"
    # data_path = "/home/lixiangyu/Dataset/Brats2019/BraTS2019_train"
    data_path = "/home/lixiangyu/Dataset/Brats2018/Training"
    # data_path = "/public1/home/acuxepn2sa/lxy/Dataset/Brats2018/Training"
    # data_path = "/home/lixiangyu/Dataset/Brats2018/Validation"
    # data_path = "/home/lixiangyu/Dataset/Brats2019/validation"
    batch_train = 1
    num_batches_per_epoch = 250


    # Network setting
    pre_trained = False
    load_old_model = False
    # Optimization
    num_epoch = 250
    lr_S = 1e-4
    lr_range = [1e-5, 0.005]
    step_size_up = 1500
    step_size_S = 100
    ratio = 0.2
    num_classes= 4
    # Note
    note= "Model"
    # pretrain
    # pre_trained_model_stage1 = "Pretrained_model/ldl.pt"
    # pre_trained_model_stage2 = "Pretrained_model/espcn.pt"
    pre_trained_model_stage1 = "runs/attention_nonew_net_stage1/best_checkpoint_120.pt"
    pre_trained_model_stage2 = "Pretrained_model/patch_128/espcn_128.pt"
    # load old model
    # checkpoint='/home/lixiangyu/myprojects/boundary/LDL2/src/runs/train MultiScaleAttentionNoNewNet/latest_checkpoint.pt'
    # checkpoint = '/home/lixiangyu/myprojects/boundary/LDL/src/runs/NonewNet_dice_loss+cross_entropy/best_checkpoint_215.pt'
    checkpoint = '/home/lixiangyu/myprojects/boundary/LDL1/src/runs/(没用)测试Jesen-shannon loss UNetLDL/best_checkpoint_49.pt'
    # checkpoint = '/home/lixiangyu/myprojects/boundary/LDL1/src/runs/V-Net/best_checkpoint_187.pt'
    # checkpoint = '/home/lixiangyu/myprojects/boundary/LDL1/src/runs/ldl_nonewnet/best_checkpoint_166.pt'
    lamda_sr = 1
    lamda_ldl = 1
    lamda_boundary = 10
    # Testing
    ckp_test = ''
    remarks = " Brats2018 测试LDL_Gaussian "


    def __init__(self):
        pass

    def display(self):
        '''
        display configurations
        :return:
        '''
        print("configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def get_str_config(self):
        '''
        get string of the configurations for displaying and storing
        :return: a string of the configuration
        '''
        str_config = "configurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                temp = "{:30} {}\n".format(a, getattr(self, a))
                str_config+=temp
        return str_config

    def write_config(self, log):
        log.write("configurations: \n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                log.write("{:30} {} \n".format(a, getattr(self, a)))


class ConfigMMWHS(object):
    write_log = False
    crop_size = [96, 96, 96]
    # crop_size = [144, 144, 144]
    data_path = "/home/lixiangyu/Dataset/MMWHS/Train/image"
    test_path = "/home/lixiangyu/Dataset/MMWHS/Test/image"
    num_batches_per_epoch = 200
    batch_train = 1
    num_classes = 8
    num_epoch = 500
    lr_S = 1e-3
    lr_D = 2e-5
    step_size_S = 100
    training_data_ratio = 0.5
    resize_r = 0.6
    modalities = 1
    WINDOW_WIDTH = 800
    WINDOW_CENTER = 300
    checkpoint = "/home/lixiangyu/myprojects/boundary/LDL2/src/runs/iseg_ldl_prior/best_checkpoint.pt"
    remarks = "train UNet mmwhseg data "


    def __init__(self):
        pass

    def display(self):
        '''
        display configurations
        :return:
        '''
        print("configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def get_str_config(self):
        '''
        get string of the configurations for displaying and storing
        :return: a string of the configuration
        '''
        str_config = "configurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                temp = "{:30} {}\n".format(a, getattr(self, a))
                str_config+=temp
        return str_config

    def write_config(self, log):
        log.write("configurations: \n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                log.write("{:30} {} \n".format(a, getattr(self, a)))