import re
from glob import glob
import nibabel as nib
import copy
import torch
from time import time
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import random
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from miscellaneous.utils import label_discrete2distribution, downsample_image, entropy
from skimage.transform import resize


class BratsDataloader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, mode="train", seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True):
        """
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.modalities = 4
        self.indices = list(range(len(data)))
        self.mode = mode
        self.rename_map = [0, 1, 2, 4]

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        if self.mode == "train":
            # initialize empty array for data and seg
            data = np.zeros((self.batch_size, self.modalities, *self.patch_size), dtype=np.float32)
            seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
            patient_names = []
            # iter0
            for i, j in enumerate(patients_for_batch):
                img_data, label_data, _ = self.load_volumes_label(j, True)
                img_data = img_data.transpose(3,0,1,2)
                label_data = np.expand_dims(label_data, axis=0)
                # patient_data = np.concatenate((img_data, label_data))
                # this will only pad patient_data if its shape is smaller than self.patch_size
                patient_data = pad_nd_image(img_data, self.patch_size)
                patient_seg = pad_nd_image(label_data, self.patch_size)
                # now random crop to self.patch_size
                # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
                # dummy dimension in order for it to work (@Todo, could be improved)
                patient_data, patient_seg = crop(patient_data[None],
                                                 patient_seg[None], self.patch_size, crop_type="random")
                data[i] = patient_data[0]
                seg[i] = patient_seg[0]
                # patient_names.append(os.path.basename(j))
        elif self.mode == "validate":
            img_list = []
            label_list = []
            for i, j in enumerate(patients_for_batch):
                img_data, label_data, _ = self.load_volumes_label(j, True)
                img_data = img_data.transpose(3,0,1,2)
                label_data = np.expand_dims(label_data, axis=0)
                img_list.append(img_data)
                label_list.append(label_data)
            data = np.array(img_list)
            seg = np.array(label_list)
        else:
            pass

        return {'data': data, 'seg':seg}

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


class BratsloaderDisplay(object):
    def __init__(self, data_path):
        """
        """
        super().__init__()
        self.modalities = 4
        self.rename_map = [0, 1, 2, 4]
        self.datapth = data_path

    def generate_display_data(self):
        img_data, label_data, _ = self.load_volumes_label(self.datapth, True)
        img_data = img_data.transpose(3,0,1,2)
        label_data = np.expand_dims(label_data, axis=0)
       
        return {'data': img_data, 'seg':label_data}

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


class BratsDataloaderNnunet(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, n_class=4, modalities=4, rename_map=(0, 1, 2, 4)):
        """
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.modalities = modalities
        self.indices = list(range(len(data)))
        # self.mode = mode
        self.rename_map = rename_map
        self.n_class = n_class

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        patient_names = []
        # iter0
        for i, j in enumerate(patients_for_batch):
            img_data, label_data, _ = self.load_volumes_label(j, True)
            img_data = img_data.transpose(3,0,1,2)
            label_data = np.expand_dims(label_data, axis=0)
            # patient_data = np.concatenate((img_data, label_data))
            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(img_data, self.patch_size)
            patient_seg = pad_nd_image(label_data, self.patch_size)
            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)

            # normalization
            # patient_data = self.Normalization(patient_data[None])
            patient_data, patient_seg = crop(patient_data[None],
                                             patient_seg[None], self.patch_size, crop_type="random")
            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

        return {'data': data, 'seg':seg}

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
    def Normalization(volume):
        '''
        Volume shape ： W*H*D*C
        :param volume:
        :param axis:
        :return:
        '''
        batch, c, _, _, _, = volume.shape
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

        norm_volume = norm_volume.transpose(0, 4, 1, 2, 3)
        norm_volume[bg_mask] = 0

        return norm_volume


class BratsDataloaderNuunetValidate(BratsDataloaderNnunet):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, mode="validate",
                 return_incomplete=False, shuffle=True, infinite=True, n_class=4, modalities=4, rename_map=(0, 1, 2, 4)):
        """
        """
        super().__init__(data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle,
                         return_incomplete, shuffle, infinite, n_class, modalities, rename_map)
        self.mode = mode

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        if self.mode == "train":
            # initialize empty array for data and seg
            data = np.zeros((self.batch_size, self.modalities, *self.patch_size), dtype=np.float32)
            seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
            patient_names = []
            # iter0
            for i, j in enumerate(patients_for_batch):
                img_data, label_data, _ = self.load_volumes_label(j, True)
                img_data = img_data.transpose(3,0,1,2)
                label_data = np.expand_dims(label_data, axis=0)
                # patient_data = np.concatenate((img_data, label_data))
                # this will only pad patient_data if its shape is smaller than self.patch_size
                patient_data = pad_nd_image(img_data, self.patch_size)
                patient_seg = pad_nd_image(label_data, self.patch_size)
                # now random crop to self.patch_size
                # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
                # dummy dimension in order for it to work (@Todo, could be improved)

                # normalization
                # patient_data = self.Normalization(patient_data[None])
                patient_data, patient_seg = crop(patient_data[None],
                                                 patient_seg[None], self.patch_size, crop_type="random")
                data[i] = patient_data[0]
                seg[i] = patient_seg[0]

            return {'data': data, 'seg':seg}
        else:
            # initialize empty array for data and seg
            data = []
            seg =[]
            # iter0
            for i, j in enumerate(patients_for_batch):
                img_data, label_data, _ = self.load_volumes_label(j, True)
                img_data = img_data.transpose(3, 0, 1, 2)
                label_data = np.expand_dims(label_data, axis=0)
                data.append(img_data)
                seg.append(label_data)
            data2 = np.array(data, dtype='float32')
            seg2 = np.array(seg, dtype="int16")
            normed_data = self.Normalization(data2)
            out_img = torch.from_numpy(normed_data).float()
            out_label = torch.from_numpy(seg2).long()

            return out_img, out_label


class BratsDataloaderLDLNuunet(BratsDataloaderNnunet):

    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, n_class=4, modalities=4, rename_map=(0, 1, 2, 4),
                 down_scale=2, stride=2, padding=0, mode = "train"):
        """
        """
        super().__init__(data, batch_size,patch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite, n_class, modalities, rename_map)
        self.down_scale = down_scale
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        dist_list = []
        entropy_list = []

        # iter0
        for i, j in enumerate(patients_for_batch):
            img_data, label_data, _ = self.load_volumes_label(j, True)
            img_data = img_data.transpose(3,0,1,2)
            label_data = np.expand_dims(label_data, axis=0)
            # patient_data = np.concatenate((img_data, label_data))
            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(img_data, self.patch_size)
            patient_seg = pad_nd_image(label_data, self.patch_size)
            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)

            # normalization
            patient_data = self.Normalization(patient_data[None])
            patient_data, patient_seg = crop(patient_data,
                                             patient_seg[None], self.patch_size, crop_type="random")
            data[i] = patient_data[0]
            seg[i] = patient_seg[0].astype("int16")
            if self.mode == "train":
                dist = label_discrete2distribution(seg[i][0, ...], self.down_scale, self.stride, self.padding, self.n_class)
                entropy_label = entropy(torch.unsqueeze(dist, dim=0))
                dist_list.append(dist)
                entropy_list.append(entropy_label)

        if self.mode == "train":
            dist_arr = torch.stack(dist_list, dim=0)
            entropy_arr = torch.stack(entropy_list, dim=0)
            return torch.from_numpy(data).float(), torch.from_numpy(seg[:,0,...]).long(), dist_arr, entropy_arr
        else:
            return torch.from_numpy(data).float(), torch.from_numpy(seg[:,0,...]).long()


class MMWHSDataLoader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, n_class=8, resize_r=None,
                 rename_map=(0, 205, 420, 500, 550, 600, 820, 850)):
        """
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.indices = list(range(len(data)))
        # self.mode = mode
        self.rename_map = rename_map
        self.n_class = n_class
        self.resize_r = resize_r

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.int16)
        patient_names = []
        # iter0
        for i, j in enumerate(patients_for_batch):
            img_data, label_data, _ = self.load_volumes_label(j, True)
            # resize the image and the label
            resize_dim = (np.array(img_data.shape) * self.resize_r).astype('int')
            img_data_resize = resize(img_data, resize_dim, order=1, preserve_range=True)
            label_data_resize = resize(label_data, resize_dim, order=0, preserve_range=True)
            label_data_resize = label_data_resize.astype("int16")
            # lab_r_data = np.zeros(label_data_resize.shape, dtype='int32')
            # for k in range(len(self.rename_map)):
            #     lab_r_data[label_data_resize == self.rename_map[i]] = k

            img_data_resize = np.expand_dims(img_data_resize, axis=0)
            label_data_resize = np.expand_dims(label_data_resize, axis=0)
            # patient_data = np.concatenate((img_data, label_data))
            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(img_data_resize, self.patch_size)
            patient_seg = pad_nd_image(label_data_resize, self.patch_size)
            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)

            patient_data, patient_seg = crop(patient_data[None],
                                             patient_seg[None], self.patch_size, crop_type="random")
            data[i] = patient_data[0]
            seg[i] = patient_seg[0]


        return {'data': data, 'seg':seg}

    # load volumes and the GT
    def load_volumes_label(self, path, rename_map_flag):
        '''
        this function get the volume data and gt from the giving path
        :param src_path: directory path of a patient
        :return: GT and the volume data（width,height, slice, modality）
        '''
        # rename_map = (0, 205, 420, 500, 550, 600, 820, 850)
        image_path = path
        # subject_name = os.path.basename(path)
        # label_path = os.path.join(os.path.dirname(os.path.dirname(path)), "label")
        label_path = path.replace("image", "label", 2)

        label_nib_data = nib.load(label_path)
        label = label_nib_data.get_data().copy()
        label_data = np.zeros(label.shape, dtype='int32')
        if rename_map_flag:
            for i in range(len(self.rename_map)):
                if i > 0:
                    label_data[label == self.rename_map[i]] = i
                else:
                    continue
        else:
            label_data = copy.deepcopy(label).astype('int16')

        volume = nib.load(image_path)
        img = volume.get_data().copy()

        return img, label_data, volume

    @staticmethod
    def Normalization(volume):
        '''
        Volume shape ： W*H*D*C
        :param volume:
        :param axis:
        :return:
        '''
        batch, c, _, _, _, = volume.shape
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

        norm_volume = norm_volume.transpose(0, 4, 1, 2, 3)
        norm_volume[bg_mask] = 0

        return norm_volume


def get_train_transform_old(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    # tr_transforms.append(
    #     SpatialTransform_2(patch_size=None, do_rotation=True,
    #         angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
    #         border_mode_data='constant', border_cval_data=0,
    #         border_mode_seg='constant', border_cval_seg=0,
    #         order_seg=1, order_data=3)
    # )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.9, 1.1), per_channel=True, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_train_transform_mmwh(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(patch_size=None, do_rotation=True,
            angle_x=(- 25 / 360. * 2 * np.pi, 25 / 360. * 2 * np.pi),
            angle_y=(- 25 / 360. * 2 * np.pi, 25 / 360. * 2 * np.pi),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3)
    )

    # now we mirror along all axes
    # tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # # brightness transform for 15% of samples
    # tr_transforms.append(BrightnessMultiplicativeTransform((0.9, 1.1), per_channel=True, p_per_sample=0.15))

    # # now we compose these transforms together
    # tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_img_info(volume_path):
    '''
    this function read all files of specific directory, get the path list
    :return:path list of all the volume files
    '''
    file_list = []
    categories = os.listdir(volume_path)
    for category in categories:
        category_path = os.path.join(volume_path, category)
        dir_list = os.listdir(category_path)
        for dire in dir_list:
            dire_lower = dire.lower()
            if not dire_lower.startswith('brats'):
                raise Exception("volume file exception!")
            file_abs_path = os.path.join(category_path, dire)
            # single_file = {"path": file_abs_path, "category": category}
            file_list.append(file_abs_path)

    return file_list


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


def get_img_info_mmwhseg(volume_path):
    file_list = glob(os.path.join(volume_path, "*.nii.gz"))
    return file_list


if __name__ == "__main__":
    patch_size = (128, 128, 128)
    batch_size = 2
    num_threads_for_brats_example = 4
    data_path = "/home/lixiangyu/Dataset/Selected_Brats/val1"
    file_list = get_img_info(data_path)
    train_list, validate_list = train_validate_split(file_list, 0.9, seed=3)

    # we create a new instance of DataLoader. This one will return batches of shape max_shape. Cropping/padding is
    # now done by SpatialTransform. If we do it this way we avoid border artifacts (the entire brain of all cases will
    # be in the batch and SpatialTransform will use zeros which is exactly what we have outside the brain)
    # this is viable here but not viable if you work with different data. If you work for example with CT scans that
    # can be up to 500x500x500 voxels large then you should do this differently. There, instead of using max_shape you
    # should estimate what shape you need to extract so that subsequent SpatialTransform does not introduce border
    # artifacts
    dataloader_train = BratsDataloader(train_list, batch_size, patch_size, num_threads_for_brats_example)
    # during training I like to run a validation from time to time to see where I am standing. This is not a correct
    # validation because just like training this is patch-based but it's good enough. We don't do augmentation for the
    # validation, so patch_size is used as shape target here
    dataloader_validation = BratsDataloader(validate_list, batch_size, patch_size,
                                            max(1, num_threads_for_brats_example // 2))

    tr_transforms = get_train_transform(patch_size)

    tr_gen = SingleThreadedAugmenter(dataloader_train, transform=tr_transforms)
    val_gen = SingleThreadedAugmenter(dataloader_validation, transform=None)
    # finally we can create multithreaded transforms that we can actually use for training
    # we don't pin memory here because this is pytorch specific.
    # tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=num_threads_for_brats_example,
    #                                 num_cached_per_queue=3,
    #                                 seeds=None, pin_memory=False)
    # # we need less processes for vlaidation because we dont apply transformations
    # val_gen = MultiThreadedAugmenter(dataloader_validation, None,
    #                                  num_processes=max(1, num_threads_for_brats_example // 2), num_cached_per_queue=1,
    #                                  seeds=None,
    #                                  pin_memory=False)
    #
    # # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # # batches while other things run in the main thread
    # tr_gen.restart()
    # val_gen.restart()

    # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
    # inifinite examples! Don't do "for batch in tr_gen:"!!!):
    num_batches_per_epoch = 5
    num_validation_batches_per_epoch = 1
    num_epochs = 2
    # let's run this to get a time on how long it takes
    time_per_epoch = []
    start = time()
    for epoch in range(num_epochs):
        start_epoch = time()
        for b in range(num_batches_per_epoch):
            batch = next(tr_gen)
            # do network training here with this batch

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)
            # run validation here
        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)
    end = time()
    total_time = end - start
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (num_epochs, total_time, str(time_per_epoch)))

    # if you notice that you have CPU usage issues, reduce the probability with which the spatial transformations are
    # applied in get_train_transform (down to 0.1 for example). SpatialTransform is the most expensive transform

    # if you wish to visualize some augmented examples, install batchviewer and uncomment this
    # if view_batch is not None:
    #     for _ in range(4):
    #         batch = next(tr_gen)
    #         view_batch(batch['data'][0], batch['seg'][0])
    # else:
    #     print("Cannot visualize batches, install batchviewer first. It's a nice and handy tool. You can get it here: "
    #           "https://github.com/FabianIsensee/BatchViewer")