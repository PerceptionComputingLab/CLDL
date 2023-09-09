import torch.utils.data as data
from glob import glob
import os
import numpy as np
from skimage.transform import resize, downscale_local_mean
import re
import nibabel as nib
import copy
from miscellaneous.utils import label_discrete2distribution, downsample_image, entropy
import cv2 as cv
import random
import torch


class CTData(data.Dataset):

    def __init__(self, root_path, mode="train", patch_dim = 96, augmentation = None, modalities=3, resize = None):
        self.augmentation = augmentation
        self.volume_path = root_path
        file_list = self._get_img_info()

        # self.train_list = file_list[:split]
        # self.validate_list = file_list[split:]

        self.train_list = file_list
        self.validate_list = file_list
        if mode=="train":
            self.data_list = self.train_list
        elif mode=="validate":
            self.data_list = self.validate_list
        else:
            # choose 10 training set for validation
            self.data_list = self.train_list[:10]
        self.modalities = modalities
        self.resize = resize
        self.patch_dim = patch_dim
        self.mode = mode
        self.rename_map = [0,1,2,4]

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]["path"]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data, label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0] + 1))
            pad_1 = (0, max(1, gap[1] + 1))
            pad_2 = (0, max(1, gap[2] + 1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)

            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]
            img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            label_temp = label_data
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]
            img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            label_temp = label_data
        img_temp = img_temp.transpose(3, 0, 1, 2)

        return (torch.from_numpy(img_temp).float(),
                torch.from_numpy(label_temp).long())

    def __len__(self):
        return len(self.data_list)

    def _get_img_info(self):
        '''
        this function read all files of specific directory, get the path list
        :return:path list of all the volume files
        '''
        file_list = []
        categories = os.listdir(self.volume_path)
        for category in categories:
            category_path = os.path.join(self.volume_path, category)
            dir_list = os.listdir(category_path)
            for dire in dir_list:
                dire_lower = dire.lower()
                if not dire_lower.startswith('brats'):
                    raise Exception("volume file exception!")
                file_abs_path = os.path.join(category_path, dire)
                single_file = {"path": file_abs_path, "category": category}
                file_list.append(single_file)

        return file_list

    # load volumes and the GT
    def load_volumes_label(self, src_path, rename_map_flag):
        '''
        this function get the volume data and gt from the giving path
        :param src_path: directory path of a patient
        :return: GT and the volume data（width,height, slice, modality）
        '''
        # rename_map = [0, 1, 2, 4]
        volume_list, seg_dict = self.data_dict_construct(src_path)
        # assert len(volume_list) == 4
        # assert seg_dict["mod"] == "seg"
        if seg_dict["mod"] == "seg":
            label_nib_data = nib.load(seg_dict["path"])
            label = label_nib_data.get_data().copy()
            # label = nib.load(seg_dict["path"]).get_data().copy()

            # resolve the issue from resizing label, we first undertake binarization and then resize
            label_data = np.zeros(label.shape, dtype='int32')

            if rename_map_flag:
                for i in range(len(self.rename_map)):
                    if i > 0:
                        label_data[label == self.rename_map[i]] = i
                    else:
                        continue
            else:
                label_data = copy.deepcopy(label).astype('int16')

        else:
            label_data = []

        img_all_modality = []
        # order of the sequences [flair, T1, T1ce, T2]
        for i in range(len(volume_list)):
            volume = nib.load(volume_list[i]["path"])
            img = volume.get_data().copy()
            # resized_img = resize(img, resize_dim, order=1, preserve_range=True)
            img_all_modality.append(img)
        # choose different modalities for the network
        if self.modalities == 4:
            # all the modalities
            img_data = img_all_modality
        elif self.modalities == 3:
            # select T1ce T2 Flair modalities
            img_data = [img_all_modality[0], img_all_modality[2], img_all_modality[3]]
        elif self.modalities == 2:
            # two modalities
            # choose T2 and Flair
            img_data = [img_all_modality[0], img_all_modality[3]]
        else:
            # one modality
            img_data = img_all_modality[0]
            img_data = np.expand_dims(img_data, axis=0)

        # list to ndarray
        img_array = np.array(img_data, "float32").transpose((1, 2, 3, 0))
        return img_array, label_data, volume

    # construct data dict
    def data_dict_construct(self, path):
        '''
        this function get the list of dictionary of the patients
        :param path: path of the patient data
        :return: list of dictionary including the path and the modality
        '''
        # list the image volumes and GT
        files = os.listdir(path)
        nii_list = sorted(glob('{}/*.nii.gz'.format(path)))
        re_style = r'[\-\_\.]+'
        volumn_list = []
        seg_dict = {"mod": "None"}
        for count, nii in enumerate(nii_list):
            # modality mapping [seg, flair, T1, T1ce, T2]
            mapping = [0, 1, 2, 3, 4]
            file = os.path.basename(nii)
            split_text = re.split(re_style, file)
            modality = split_text[-3]
            assert modality in ["flair", "seg", "t1", "t2", "t1ce"]
            if modality == "seg":
                data_dict = {"mod": modality, "path": nii, "count": mapping[0]}
            elif modality == "flair":
                data_dict = {"mod": modality, "path": nii, "count": mapping[1]}
            elif modality == "t1":
                data_dict = {"mod": modality, "path": nii, "count": mapping[2]}
            elif modality == "t1ce":
                data_dict = {"mod": modality, "path": nii, "count": mapping[3]}
            else:
                data_dict = {"mod": modality, "path": nii, "count": mapping[4]}

            if data_dict["mod"] != "seg":
                volumn_list.append(data_dict)
            else:
                seg_dict = {"mod": modality, "path": nii, "count": mapping[0]}
        # sort the modalites in the list
        volumn_list.sort(key=lambda x: x["count"])
        return volumn_list, seg_dict

    @staticmethod
    def Normalization(volume, axis=None):
        '''
        Volume shape ： W*H*D*C
        :param volume:
        :param axis:
        :return:
        '''
        _, _, _, C = volume.shape
        bg_mask = volume == 0
        mean_arr = np.zeros(C, dtype="float32")
        std_arr = np.zeros(C, dtype="float32")
        for i in range(C):
            data = volume[..., i]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std

        norm_volume = (volume - mean_arr) / std_arr
        norm_volume[bg_mask] = 0

        return norm_volume


class BratsDataValidate(data.Dataset):
    def __init__(self, root_path, mode="train", train_ratio=0.9, seed=3,  patch_dim = 96, augmentation = None, modalities=4, resize = None):
        self.augmentation = augmentation
        self.volume_path = root_path
        self.train_ratio = train_ratio
        self.seed = seed
        file_list = self._get_img_info()
        self.train_list, self.validate_list = train_validate_split(file_list, train_ratio, seed)
        if mode=="train":
            self.data_list = self.train_list
        elif mode=="validate":
            self.data_list = self.validate_list
        else:
            # training set for validation
            self.data_list = self.train_list
        self.modalities = modalities
        self.resize = resize
        self.patch_dim = patch_dim
        self.mode = mode
        self.rename_map = [0,1,2,4]

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data, label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)


            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)

            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)


            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

        img_temp = img_temp.transpose(3, 0, 1, 2)

        return (torch.from_numpy(img_temp).float(),
                torch.from_numpy(label_temp).long())

    def __len__(self):
        return len(self.data_list)

    def _get_img_info(self):
        '''
        this function read all files of specific directory, get the path list
        :return:path list of all the volume files
        '''
        file_list = []
        categories = os.listdir(self.volume_path)
        for category in categories:
            category_path = os.path.join(self.volume_path, category)
            dir_list = os.listdir(category_path)
            for dire in dir_list:
                dire_lower = dire.lower()
                if not dire_lower.startswith('brats'):
                    raise Exception("volume file exception!")
                file_abs_path = os.path.join(category_path, dire)
                # single_file = {"path": file_abs_path, "category": category}
                file_list.append(file_abs_path)

        return file_list

    # load volumes and the GT
    def load_volumes_label(self, src_path, rename_map_flag):
        '''
        this function get the volume data and gt from the giving path
        :param src_path: directory path of a patient
        :return: GT and the volume data（width,height, slice, modality）
        '''
        # rename_map = [0, 1, 2, 4]
        volume_list, seg_dict = self.data_dict_construct(src_path)
        # assert len(volume_list) == 4
        # assert seg_dict["mod"] == "seg"
        if seg_dict["mod"] == "seg":
            label_nib_data = nib.load(seg_dict["path"])
            label = label_nib_data.get_data().copy()
            # label = nib.load(seg_dict["path"]).get_data().copy()

            # resolve the issue from resizing label, we first undertake binarization and then resize
            label_data = np.zeros(label.shape, dtype='int32')

            if rename_map_flag:
                for i in range(len(self.rename_map)):
                    if i > 0:
                        label_data[label == self.rename_map[i]] = i
                    else:
                        continue
            else:
                label_data = copy.deepcopy(label).astype('int16')

        else:
            label_data = []

        img_all_modality = []
        # order of the sequences [flair, T1, T1ce, T2]
        for i in range(len(volume_list)):
            volume = nib.load(volume_list[i]["path"])
            img = volume.get_data().copy()
            # resized_img = resize(img, resize_dim, order=1, preserve_range=True)
            img_all_modality.append(img)

        # choose different modalities for the network
        if self.modalities == 4:
            # all the modalities
            img_data = img_all_modality
        elif self.modalities == 3:
            # select T1ce T2 Flair modalities
            img_data = [img_all_modality[0], img_all_modality[2], img_all_modality[3]]
        elif self.modalities == 2:
            # two modalities
            # choose T2 and Flair
            img_data = [img_all_modality[0], img_all_modality[3]]
        else:
            # one modality
            img_data = img_all_modality[0]
            img_data = np.expand_dims(img_data, axis=0)

        # list to ndarray
        img_array = np.array(img_data, "float32").transpose((1, 2, 3, 0))
        return img_array, label_data, volume

    # construct data dict
    def data_dict_construct(self, path):
        '''
        this function get the list of dictionary of the patients
        :param path: path of the patient data
        :return: list of dictionary including the path and the modality
        '''
        # list the image volumes and GT
        files = os.listdir(path)
        nii_list = sorted(glob('{}/*.nii.gz'.format(path)))
        re_style = r'[\-\_\.]+'
        volumn_list = []
        seg_dict = {"mod": "None"}
        for count, nii in enumerate(nii_list):
            # modality mapping [seg, flair, T1, T1ce, T2]
            mapping = [0, 1, 2, 3, 4]
            file = os.path.basename(nii)
            split_text = re.split(re_style, file)
            modality = split_text[-3]
            assert modality in ["flair", "seg", "t1", "t2", "t1ce"]
            if modality == "seg":
                data_dict = {"mod": modality, "path": nii, "count": mapping[0]}
            elif modality == "flair":
                data_dict = {"mod": modality, "path": nii, "count": mapping[1]}
            elif modality == "t1":
                data_dict = {"mod": modality, "path": nii, "count": mapping[2]}
            elif modality == "t1ce":
                data_dict = {"mod": modality, "path": nii, "count": mapping[3]}
            else:
                data_dict = {"mod": modality, "path": nii, "count": mapping[4]}

            if data_dict["mod"] != "seg":
                volumn_list.append(data_dict)
            else:
                seg_dict = {"mod": modality, "path": nii, "count": mapping[0]}
        # sort the modalites in the list
        volumn_list.sort(key=lambda x: x["count"])
        return volumn_list, seg_dict

    @staticmethod
    def Normalization(volume, axis=None):
        '''
        Volume shape ： W*H*D*C
        :param volume:
        :param axis:
        :return:
        '''
        _, _, _, C = volume.shape
        bg_mask = volume == 0
        mean_arr = np.zeros(C, dtype="float32")
        std_arr = np.zeros(C, dtype="float32")
        for i in range(C):
            data = volume[..., i]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std

        norm_volume = (volume - mean_arr) / std_arr
        norm_volume[bg_mask] = 0

        return norm_volume


class BratsData(data.Dataset):

    def __init__(self, root_path, mode="train", train_ratio=0.9, seed=3,  patch_dim = 96, augmentation = None, modalities=4, resize = None):
        self.augmentation = augmentation
        self.volume_path = root_path
        self.train_ratio = train_ratio
        self.seed = seed
        file_list = self._get_img_info()
        self.train_list, self.validate_list = train_validate_split(file_list, train_ratio, seed)
        if mode=="train":
            # TODO More data in an epoch  when augmentation
            if augmentation is not None:
                self.data_list = self.train_list+self.train_list+self.train_list+self.train_list
            else:
                self.data_list = self.train_list
        elif mode=="validate":
            self.data_list = self.validate_list
        else:
            # training set for validation
            self.data_list = self.train_list[:len(self.validate_list)]
        self.modalities = modalities
        self.resize = resize
        self.patch_dim = patch_dim
        self.mode = mode
        self.rename_map = [0,1,2,4]

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data, label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)


            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)

            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)


            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(0, gap[0]))
            pad_1 = (0, max(0, gap[1]))
            pad_2 = (0, max(0, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

        img_temp = img_temp.transpose(3, 0, 1, 2)

        return (torch.from_numpy(img_temp).float(),
                torch.from_numpy(label_temp).long())

    def __len__(self):
        return len(self.data_list)

    def _get_img_info(self):
        '''
        this function read all files of specific directory, get the path list
        :return:path list of all the volume files
        '''
        file_list = []
        categories = os.listdir(self.volume_path)
        for category in categories:
            category_path = os.path.join(self.volume_path, category)
            dir_list = os.listdir(category_path)
            for dire in dir_list:
                dire_lower = dire.lower()
                if not dire_lower.startswith('brats'):
                    raise Exception("volume file exception!")
                file_abs_path = os.path.join(category_path, dire)
                # single_file = {"path": file_abs_path, "category": category}
                file_list.append(file_abs_path)

        return file_list

    # load volumes and the GT
    def load_volumes_label(self, src_path, rename_map_flag):
        '''
        this function get the volume data and gt from the giving path
        :param src_path: directory path of a patient
        :return: GT and the volume data（width,height, slice, modality）
        '''
        # rename_map = [0, 1, 2, 4]
        volume_list, seg_dict = self.data_dict_construct(src_path)
        # assert len(volume_list) == 4
        # assert seg_dict["mod"] == "seg"
        if seg_dict["mod"] == "seg":
            label_nib_data = nib.load(seg_dict["path"])
            label = label_nib_data.get_data().copy()
            # label = nib.load(seg_dict["path"]).get_data().copy()

            # resolve the issue from resizing label, we first undertake binarization and then resize
            label_data = np.zeros(label.shape, dtype='int32')

            if rename_map_flag:
                for i in range(len(self.rename_map)):
                    if i > 0:
                        label_data[label == self.rename_map[i]] = i
                    else:
                        continue
            else:
                label_data = copy.deepcopy(label).astype('int16')

        else:
            label_data = []

        img_all_modality = []
        # order of the sequences [flair, T1, T1ce, T2]
        for i in range(len(volume_list)):
            volume = nib.load(volume_list[i]["path"])
            img = volume.get_data().copy()
            # resized_img = resize(img, resize_dim, order=1, preserve_range=True)
            img_all_modality.append(img)

        # choose different modalities for the network
        if self.modalities == 4:
            # all the modalities
            img_data = img_all_modality
        elif self.modalities == 3:
            # select T1ce T2 Flair modalities
            img_data = [img_all_modality[0], img_all_modality[2], img_all_modality[3]]
        elif self.modalities == 2:
            # two modalities
            # choose T2 and Flair
            img_data = [img_all_modality[0], img_all_modality[3]]
        else:
            # one modality
            img_data = img_all_modality[0]
            img_data = np.expand_dims(img_data, axis=0)

        # list to ndarray
        img_array = np.array(img_data, "float32").transpose((1, 2, 3, 0))
        return img_array, label_data, volume

    # construct data dict
    def data_dict_construct(self, path):
        '''
        this function get the list of dictionary of the patients
        :param path: path of the patient data
        :return: list of dictionary including the path and the modality
        '''
        # list the image volumes and GT
        files = os.listdir(path)
        nii_list = sorted(glob('{}/*.nii.gz'.format(path)))
        re_style = r'[\-\_\.]+'
        volumn_list = []
        seg_dict = {"mod": "None"}
        for count, nii in enumerate(nii_list):
            # modality mapping [seg, flair, T1, T1ce, T2]
            mapping = [0, 1, 2, 3, 4]
            file = os.path.basename(nii)
            split_text = re.split(re_style, file)
            modality = split_text[-3]
            assert modality in ["flair", "seg", "t1", "t2", "t1ce"]
            if modality == "seg":
                data_dict = {"mod": modality, "path": nii, "count": mapping[0]}
            elif modality == "flair":
                data_dict = {"mod": modality, "path": nii, "count": mapping[1]}
            elif modality == "t1":
                data_dict = {"mod": modality, "path": nii, "count": mapping[2]}
            elif modality == "t1ce":
                data_dict = {"mod": modality, "path": nii, "count": mapping[3]}
            else:
                data_dict = {"mod": modality, "path": nii, "count": mapping[4]}

            if data_dict["mod"] != "seg":
                volumn_list.append(data_dict)
            else:
                seg_dict = {"mod": modality, "path": nii, "count": mapping[0]}
        # sort the modalites in the list
        volumn_list.sort(key=lambda x: x["count"])
        return volumn_list, seg_dict

    @staticmethod
    def Normalization_old(volume, axis=None):
        mean = np.mean(volume, axis=axis)
        std = np.std(volume, axis=axis)
        norm_volume = (volume - mean) / std
        return norm_volume

    @staticmethod
    def Normalization(volume, axis = None):
        '''
        Volume shape ： W*H*D*C
        :param volume:
        :param axis:
        :return:
        '''
        _,_,_,C = volume.shape
        bg_mask = volume == 0
        mean_arr = np.zeros(C, dtype="float32")
        std_arr = np.zeros(C, dtype="float32")
        for i in range(C):
            data = volume[..., i]
            selected_data = data[data > 0]
            mean = np.mean(selected_data)
            std = np.std(selected_data)
            mean_arr[i] = mean
            std_arr[i] = std

        norm_volume = (volume - mean_arr) / std_arr
        norm_volume[bg_mask] = 0

        return norm_volume


class CTDataDist(CTData):
    def __init__(self, scale, stride, padding, n_class, **kwargs):
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.n_class = n_class
        super(CTDataDist, self).__init__(**kwargs)

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]["path"]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data, label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            if self.resize!=None:
                self.data = resize(img_data, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data
                self.label = label_data

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)

            pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])

            # transfer to label distribution and corresponding downsampled volume
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample = downsample_image(img_temp, scale=self.scale)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            # The downsampled images and labels should be identical in shapes except for the channel dimension.
            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    torch.from_numpy(label_dist).float())
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]
            img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            label_temp = label_data

            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,)*3+(1,))
            img_downsample = downsample_image(img_temp, scale=self.scale)
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)

            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    torch.from_numpy(label_dist).float()
                    )
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]
            img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            label_temp = label_data
            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,) * 3 + (1,))
            img_downsample = downsample_image(img_temp, scale=self.scale)
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)

            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    torch.from_numpy(label_dist).float()
                    )


class BratsDataDist(BratsData):
    def __init__(self, scale, stride, padding, n_class, **kwargs):
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.n_class = n_class
        super(BratsDataDist, self).__init__(**kwargs)

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data[...,0],img_data[...,1],img_data[...,2], img_data[...,3],
                                                      label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region(240*240*155---->approximately 130*170*140)
            regions = get_brain_region(data_aug[:,:,:,0])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)
            try:
                pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            except:
                a = 1
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])

            # transfer to label distribution and corresponding downsampled volume
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample = downsample_image(img_temp, scale=self.scale)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            # The downsampled images and labels should be identical in shapes except for the channel dimension.
            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    label_dist.float())
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 0])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,)*3+(1,))
            img_downsample = downsample_image(img_temp, scale=self.scale)
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)

            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    label_dist.float()
                    )
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]))
            pad_1 = (0, max(1, gap[1]))
            pad_2 = (0, max(1, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            # label_temp = label_data
            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,) * 3 + (1,))
            img_downsample = downsample_image(img_temp, scale=self.scale)
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)

            assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    torch.from_numpy(img_downsample_temp).float(),
                    torch.from_numpy(label_dist).float()
                    )


class BratsDataDistEntropy(BratsData):
    def __init__(self, scale, stride, padding, n_class, **kwargs):
        super(BratsDataDistEntropy, self).__init__(**kwargs)
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.n_class = n_class

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data[...,0],img_data[...,1],img_data[...,2], img_data[...,3],
                                                      label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region(240*240*155---->approximately 130*170*140)
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)
            try:
                pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            except:
                a = 1
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])

            # transfer to label distribution and corresponding downsampled volume
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            # The downsampled images and labels should be identical in shapes except for the channel dimension.
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            label_entropy = entropy(torch.unsqueeze(label_dist, dim=0))
            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    label_dist, label_entropy
                    )
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,)*3+(1,))
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            # label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            #
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    )
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]))
            pad_1 = (0, max(1, gap[1]))
            pad_2 = (0, max(1, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            # label_temp = label_data
            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,) * 3 + (1,))
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            # label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            #
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    )


class BratsDataDistEntropyValidate(BratsDataValidate):
    def __init__(self, scale, stride, padding, n_class, **kwargs):
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.n_class = n_class
        super(BratsDataDistEntropyValidate, self).__init__(**kwargs)

    def __getitem__(self, index):
        # Get MR files
        # data directory of a patient
        single_dir_path = self.data_list[index]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)

        # data augmentation
        if self.augmentation != None:
            data_aug, mask_aug = data_augment_volume([img_data[...,0],img_data[...,1],img_data[...,2], img_data[...,3],
                                                      label_data], self.augmentation)
        else:
            data_aug = img_data
            mask_aug = label_data
        if self.mode == "train":
            # reduce background region(240*240*155---->approximately 130*170*140)
            regions = get_brain_region(data_aug[:,:,:,2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            if self.resize!=None:
                self.data = resize(img_data_pad, self.resize, order=3, cval=0, preserve_range=True)
                self.label = resize(label_data_pad, self.resize, order=0, cval=0, preserve_range=True)
            else:
                self.data = img_data_pad
                self.label = label_data_pad

            data1_norm = self.Normalization(self.data, axis=(0, 1, 2))

            # randomly select a box anchor
            l, w, h = self.label.shape
            l_rand = np.arange(l - self.patch_dim)  # get a start point
            w_rand = np.arange(w - self.patch_dim)
            h_rand = np.arange(h - self.patch_dim)
            np.random.shuffle(l_rand)  # shuffle the start point series
            np.random.shuffle(w_rand)
            np.random.shuffle(h_rand)
            try:
                pos = np.array([l_rand[0], w_rand[0], h_rand[0]])  # get the start point
            except:
                a = 1
            # crop the volume to get the same size for the network
            img_temp = copy.deepcopy(data1_norm[pos[0]:pos[0] +self.patch_dim, pos[1]:pos[1] +
                                                            self.patch_dim, pos[2]:pos[2] +self.patch_dim, :])

            # crop the label just like the volume data
            label_temp = copy.deepcopy(self.label[pos[0]:pos[0] + self.patch_dim,
                                       pos[1]:pos[1] + self.patch_dim,pos[2]:pos[2] + self.patch_dim])

            # transfer to label distribution and corresponding downsampled volume
            label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            # The downsampled images and labels should be identical in shapes except for the channel dimension.
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            label_entropy = entropy(torch.unsqueeze(label_dist, dim=0))
            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    label_dist, label_entropy
                    )
        elif self.mode=="validate":
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]+1))
            pad_1 = (0, max(1, gap[1]+1))
            pad_2 = (0, max(1, gap[2]+1))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,)*3+(1,))
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            # label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            #
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    )
        else:
            # reduce background region
            regions = get_brain_region(data_aug[:, :, :, 2])
            img_data = data_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5], :]
            label_data = mask_aug[regions[0]:regions[1], regions[2]:regions[3], regions[4]:regions[5]]

            # pad image and label in case the shape is smaller than patch_size
            shape = img_data.shape[:-1]
            gap = np.array(self.patch_dim, dtype="int16") - np.array(shape, dtype="int16")
            pad_0 = (0, max(1, gap[0]))
            pad_1 = (0, max(1, gap[1]))
            pad_2 = (0, max(1, gap[2]))
            pad_3 = (0, 0)
            pad_dim = (pad_0, pad_1, pad_2, pad_3)
            img_data_pad = np.pad(img_data, pad_width=pad_dim, mode="constant", constant_values=0)
            label_data_pad = np.pad(label_data, pad_width=(pad_0, pad_1, pad_2), mode="constant", constant_values=0)

            img_temp = self.Normalization(img_data_pad, axis=(0, 1, 2))
            label_temp = label_data_pad

            # img_temp = self.Normalization(img_data, axis=(0, 1, 2))
            # label_temp = label_data
            # Get the downsampled version
            # img_downsample = downscale_local_mean(img_temp, factors=(self.scale,) * 3 + (1,))
            # img_downsample = downsample_image(img_temp, scale=self.scale)
            # label_dist = label_discrete2distribution(label_temp, self.scale, self.stride, self.padding, self.n_class)
            # img_downsample_temp = img_downsample.transpose(3, 0, 1, 2)
            img_temp = img_temp.transpose(3, 0, 1, 2)
            #
            # assert img_downsample_temp.shape[1:] == label_dist.shape[1:]

            return (torch.from_numpy(img_temp).float(),
                    torch.from_numpy(label_temp).long(),
                    )


def train_validate_split(filenames, train_ratio, seed=1):
    '''
    Split the dataset to training and validation set
    :param filenames: list of paths  eg.["1.png", "2.png"....]
    :param train_ratio: Float, the ratio of the training set
    :param seed: random seed, to make sure the spliting is reproducible
    :return: training and validation set
    '''
    # First sort the data set to make sure the filenames have a fixed order
    filenames.sort()
    random.seed(seed)
    random.shuffle(filenames)
    split = int(train_ratio * len(filenames))
    train_filenames = filenames[:split]
    test_filenames = filenames[split:]
    return train_filenames, test_filenames


# data augmentation
def data_augment_volume(datalist, augmentation):
    # first get the volume data from the data list
    image1, image2, image3, image4, mask1 = datalist
    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image1_shape = image1.shape
        mask1_shape = mask1.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        # image should be uint8!!
        image1 = det.augment_image(image1)
        image2 = det.augment_image(image2)
        image3 = det.augment_image(image3)
        image4 = det.augment_image(image4)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask1 = det.augment_image(mask1.astype(np.uint8),
                                  hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image1.shape == image1_shape, "Augmentation shouldn't change image size"
        assert mask1.shape == mask1_shape, "Augmentation shouldn't change mask size"

        # Change mask back to bool
        # masks = masks.astype(np.bool)
        augmented_image = np.array([image1, image2, image3, image4])
    else:
        augmented_image = np.array([image1, image2, image3, image4])
    return augmented_image.transpose(1,2,3,0),  mask1


def data_augment(image, mask, augmentation):
    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        # image should be uint8!!
        images = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        masks = det.augment_image(mask.astype(np.uint8),
                                   hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert images.shape == image_shape, "Augmentation shouldn't change image size"
        assert masks.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        # masks = masks.astype(np.bool)
    return images, masks


def get_brain_region(volume_data):
    # volume = nib.load(volume_path)
    # volume_data = volume.get_data()
    # get the brain region
    indice_list = np.where(volume_data > 0)
    # calculate the min and max of the indice,  here volume have 3 channels
    channel_0_min = min(indice_list[0])
    channel_0_max = max(indice_list[0])

    channel_1_min = min(indice_list[1])
    channel_1_max = max(indice_list[1])

    channel_2_min = min(indice_list[2])
    channel_2_max = max(indice_list[2])

    brain_volume = volume_data[channel_0_min:channel_0_max, channel_1_min:channel_1_max,channel_2_min:channel_2_max]

    return (channel_0_min, channel_0_max, channel_1_min, channel_1_max, channel_2_min, channel_2_max)


def get_brain_region_new(volume_data, patch_size):
    shape = volume_data.shape
    # volume = nib.load(volume_path)
    # volume_data = volume.get_data()
    # get the brain region
    indice_list = np.where(volume_data > 0)
    # calculate the min and max of the indice,  here volume have 3 channels
    channel_0_min = min(indice_list[0])
    channel_0_max = max(indice_list[0])

    channel_1_min = min(indice_list[1])
    channel_1_max = max(indice_list[1])

    channel_2_min = min(indice_list[2])
    channel_2_max = max(indice_list[2])

    len_0 = channel_0_max - channel_0_min
    len_1 = channel_1_max - channel_1_min
    len_2 = channel_2_max - channel_2_min

    gap_0 = patch_size[0] - len_0
    gap_1 = patch_size[1] - len_1
    gap_2 = patch_size[2] - len_2

    if gap_0 > 0:
        pad_0 = gap_0//2
        # boudary problem
        chn_0_min = channel_0_min-pad_0
        chn_0_max = channel_0_max + (gap_0 - pad_0)
        if chn_0_min<0:
            chn_0_min = 0
            chn_0_max = patch_size[0]
        if chn_0_max>shape[0]:
            chn_0_max = shape[0]
            chn_0_min = shape[0] - patch_size[0]
    else:
        chn_0_min = channel_0_min
        chn_0_max = channel_0_max
    if gap_1 > 0:
        pad_1 = gap_1//2
        chn_1_min = channel_1_min - pad_1
        chn_1_max = channel_1_max +(gap_1 - pad_1)
        if chn_1_min<0:
            chn_1_min = 0
            chn_1_max = patch_size[1]
        if chn_1_max>shape[1]:
            chn_1_max = shape[1]
            chn_1_min = shape[1] - patch_size[1]
    else:
        chn_1_min = channel_0_min
        chn_1_max = channel_1_max
    if gap_2 > 0:
        pad_2 = gap_2 //2
        chn_2_min = channel_2_min - pad_2
        chn_2_max = channel_2_max +(gap_2 - pad_2)
        if chn_2_min<0:
            chn_2_min = 0
            chn_2_max = patch_size[2]
        if chn_2_max>shape[2]:
            chn_2_max = shape[2]
            chn_2_min = shape[2] - patch_size[2]
    else:
        chn_2_min = channel_2_min
        chn_2_max = channel_2_max

    brain_volume = volume_data[channel_0_min:channel_0_max, channel_1_min:channel_1_max,channel_2_min:channel_2_max]

    return (chn_0_min, chn_0_max, chn_1_min, chn_1_max, chn_2_min, chn_2_max)


class TestClass(CTData):
    def __getitem__(self, index):
        single_dir_path = self.data_list[index]["path"]
        img_data, label_data, _ = self.load_volumes_label(single_dir_path, True)
        return label_data


if __name__ == "__main__":

    # a = np.random.randint(3,5,(78, 129, 89),)
    # b = np.pad(a, ((10,10), (10,10), (10,10)))
    # dim = get_brain_region_new(b, (96,96,96))
    # data_path = "/home/lixiangyu/Dataset/Brats2019/BraTS2019_train"
    # data = BratsData(data_path, modalities=4)
    # c = np.stack((b,b), axis=-1)
    # fdfd = data.NormalizationFG(c)
    # for i, j in enumerate(data):
    #     a = i
    #     b = j

    path = "/home/lixiangyu/Dataset/Brats2018/Training"
    data_generator = TestClass(root_path=path)
    trainloader = data.DataLoader(data_generator, batch_size=1, shuffle=False)
    HGG_count_list = []
    LGG_count_list = []
    for i, data in enumerate(trainloader):
        data = torch.squeeze(data.view(1, -1), dim=0)
        data_arr = data.numpy()
        x = np.bincount(data_arr)
        if len(x)==4:
            HGG_count_list.append(x)
        else:
            LGG_count_list.append(x)
    HGG = np.array(HGG_count_list)
    LGG = np.array(LGG_count_list)
    HGG_mean = np.mean(HGG, axis=0)
    LGG_mean = np.mean(LGG, axis=0)
    a = 1


