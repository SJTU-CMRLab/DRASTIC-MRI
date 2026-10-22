import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import h5py
from main import readfile2numpyv2,savenumpy2mat,readmaskfile2numpy
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat,savemat
import random
import numpy as np
from fastmri.data import transforms as T
import fastmri
import torch.nn as nn
from torchvision.transforms import Resize


ROOT_DIR_TRAIN = "/home/ubuntu/shared/Zhihao/M4Raw/multicoil_train/"
ROOT_DIR_VAL = "/home/ubuntu/shared/Zhihao/M4Raw/multicoil_val/"

ROOT_DIR_TRAIN_FASTMRI = "/home/ubuntu/shared/Zhihao/fastMRI/Brain_2/multicoil_train_CC/"
ROOT_DIR_VAL_FASTMRI = "/home/ubuntu/shared/Zhihao/fastMRI/Brain/Validation/"

class M4RawDataset(Dataset):
    def __init__(self, contrasts=['FLAIR','T1','T2'],root_dir=ROOT_DIR_TRAIN,pattern='Uniform4',mode='train'):
        #将所有数据添加到patient level表中
        self.mode = mode
        self.file_list = []
        self.root_dir = root_dir
        h5file_list = os.listdir(root_dir)
        h5file_list.sort()
        self.idxS = 0 #num of sample    #maximum: 128
        self.idxz = 0 #num of slice     #maximum: 18
        self.idxc = 0 #num of contrast  #maximum: 8
        self.current_scan = []
        self.current_data = [None for _ in range(8)]
        self.current_gt = [None for _ in range(8)]
        self.scalars = [None for _ in range(8)]
        if mode == 'validation':
            h5file_list = h5file_list[0:16]
        #建表
        for idx_scan in range(len(h5file_list)//8):
            scan_dict = {}
            scan_dict['FLAIR_1'] = h5file_list[int(idx_scan*8+0)]
            scan_dict['FLAIR_2'] = h5file_list[int(idx_scan*8+1)]
            scan_dict['T1_1'] = h5file_list[int(idx_scan*8+2)]
            scan_dict['T1_2'] = h5file_list[int(idx_scan*8+3)]
            scan_dict['T1_3'] = h5file_list[int(idx_scan*8+4)]
            scan_dict['T2_1'] = h5file_list[int(idx_scan*8+5)]
            scan_dict['T2_2'] = h5file_list[int(idx_scan*8+6)]
            scan_dict['T2_3'] = h5file_list[int(idx_scan*8+7)]
            self.file_list.append(scan_dict)
        if pattern == 'Uniform4':#注意欠采的维度要和cardiac保持一致,是倒数第二个维度欠采  
            self.mask = torch.zeros((256,256))
            self.mask[0::4,:] = 1.0
            self.mask[128-8:128+8,:] = 1.0
            self.pe_lines = 256
        elif pattern == 'Uniform8':
            self.mask = torch.zeros((256,256))
            self.mask[0::8,:] = 1.0
            self.mask[128-8:128+8,:] = 1.0
            self.pe_lines = 256

    def __len__(self)->int:
        return len(self.file_list)*8*18  #8 repetitions, 18 slices per scan
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def __getitem__(self, index:int):
        if self.idxz == 0 and self.idxc == 0:
            self.current_scan = []
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['FLAIR_1']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['FLAIR_2']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T1_1']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T1_2']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T1_3']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T2_1']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T2_2']))
            self.current_scan.append(os.path.join(self.root_dir,self.file_list[self.idxS]['T2_3']))
            if self.mode == 'train':
                random.shuffle(self.current_scan)
            #读取数据
            for idx,filename in enumerate(self.current_scan):
                with h5py.File(filename,'r') as hf:
                    self.scalars[idx] = hf.attrs['max']
                    self.current_data[idx] = hf['kspace'][()]
                    self.current_gt[idx] = hf['reconstruction_rss'][()]
        Nsample = self.idxS
        Nslice = self.idxz
        Nrep = self.idxc
        kspace = torch.from_numpy(self.current_data[self.idxc][self.idxz,:,:,:]).permute(0,2,1)  #num of coil, num of Ny, num of Nx
        target = torch.from_numpy(self.current_gt[self.idxc][self.idxz,:,:]).permute(1,0)  #产生target图像
        if kspace.shape[0] != 10: #处理通道数
            zero_array = torch.zeros((10-kspace.shape[0],kspace.shape[1],kspace.shape[2]),dtype=kspace.dtype)+1e-10 #用0拼接会得到nan
            kspace = torch.concat([kspace,zero_array],dim=0)
        #产生欠采样kspace
        kspace = torch.view_as_real(kspace) #* self.mask
    #    image_u = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_u)))
        kspace /= self.scalars[self.idxc]
        target /= self.scalars[self.idxc]
        #产生class_label view_label
        view_label = 8 # brain axial (transversal)
        if self.idxc <= 1:
            class_label = 9  #FLAIR
        elif self.idxc <=4:
            class_label = 10  #T1W
        else:
            class_label = 11  #T2W
        class_label = np.array([class_label])
        #更新index
        self.idxc += 1
        if self.idxc >= 8:
            self.idxc = 0
            self.idxz += 1
        if self.idxz >= 18:
            self.idxz = 0
            self.idxS += 1
        if self.idxS >= len(self.file_list):
            self.idxS = 0
    
        #返回数据
        return kspace, target, self.mask, Nsample,Nslice,class_label,Nrep,self.pe_lines,view_label


class FastMRIDataset(Dataset):
    def __init__(self,contrasts=['FLAIR','T1','T2'],root_dir=ROOT_DIR_TRAIN_FASTMRI,pattern='Uniform4',mode='train') -> None:
        super().__init__()
        #TODO
        self.file_list = []
        self.root_dir = root_dir
        matfile_list = os.listdir(root_dir)
        matfile_list.sort()
        self.idxS = 0 #num of sample    
        self.idxz = 0 #num of slice     
        self.current_scan = []
        self.current_data = []
        self.current_gt = []
        self.scalars = []
        self.num_slices = 0
        self.class_label = 0
        for idx, filename in enumerate(matfile_list):            
            with h5py.File(os.path.join(root_dir,filename),'r') as hf:
                Nsl = hf['Nsl'][()]
                self.num_slices += int(Nsl)
                self.file_list.append(os.path.join(root_dir,filename))
        #产生欠采样mask
        if pattern == 'Uniform4':#注意欠采的维度要和cardiac保持一致,是倒数第二个维度欠采  
            self.mask = torch.zeros((320,384))
            self.mask[0::4,:] = 1.0
            self.mask[160-8:160+8,:] = 1.0
        elif pattern == 'Uniform8':
            self.mask = torch.zeros((320,384))
            self.mask[0::8,:] = 1.0
            self.mask[160-8:160+8,:] = 1.0
        #增加一个决定是否要采用random mask的机制
        if pattern == 'Random8':
            self.mask = torch.zeros((320,384))
            self.mask[0::8,:] = 1.0
            self.mask[160-8:160+8,:] = 1.0
            #change the mask
            n_sampled_points = self.mask.sum()
            mask_rand = torch.zeros_like(self.mask)
            random_indices = torch.randperm(int(n_sampled_points))
            mask_rand[random_indices] = 1.0
            self.mask = mask_rand

    def shuffle(self):
        random.shuffle(self.file_list)
        return
        
    def __len__(self):
        return self.num_slices

    def __getitem__(self, index):
        if self.idxz == 0:
            #读取数据
            self.filename = self.file_list[self.idxS]
            with h5py.File(self.filename,'r') as hf:
                data = hf['kspace'][()]
                self.current_data = data['real'] + 1j*data['imag']
                self.scalars = hf['max'][()].astype(np.float32)
                if 'FLAIR' in self.filename:
                    self.class_label = 9
                elif 'T1' in self.filename:
                    self.class_label = 10
                elif 'T2' in self.filename:
                    self.class_label = 11 
        Nsample = self.idxS
        Nslice = self.idxz

        kspace = torch.from_numpy(self.current_data[self.idxz,:,:,:]).permute(0,2,1)  #num of coil, num of Ny, num of Nx
        
        if kspace.shape[0] != 10: #处理通道数
            zero_array = torch.zeros((10-kspace.shape[0],kspace.shape[1],kspace.shape[2]),dtype=kspace.dtype)+1e-10 #用0拼接会得到nan
            kspace = torch.concat([kspace,zero_array],dim=0)
        kspace = torch.view_as_real(kspace)  #Nc,Ny,Nx,2
        self.pe_lines = kspace.shape[-3]
        #  处理RO方向shape
        Nro = kspace.shape[-2]
        if Nro > 384:
            padx = (Nro - 384) // 2
            image = fastmri.ifft2c(kspace)  #Nc, Ny, Nx,2
            if self.pe_lines > 320:
                pady = (self.pe_lines - 320) // 2
                image = image[:,pady:-pady,padx:-padx,:]
            elif self.pe_lines < 320:  #有可能是奇数
                pady = (320 - self.pe_lines) // 2
                image = image[:,:,padx:-padx,:]
                if self.pe_lines % 2 == 1:
                    image = torch.view_as_real(torch.nn.functional.pad(torch.view_as_complex(image),[0,0,pady+1,pady],mode='constant',value=1e-15))
                else:
                    image = torch.view_as_real(torch.nn.functional.pad(torch.view_as_complex(image),[0,0,pady,pady],mode='constant',value=1e-15))
            else:
                image = image[:,:,padx:-padx,:]
            kspace = fastmri.fft2c(image)
            target = fastmri.rss(fastmri.complex_abs(image))
        else:
            if self.pe_lines > 320:
                image = fastmri.ifft2c(kspace)
                pady = (self.pe_lines - 320) // 2
                image = image[:,pady:-pady,:,:]
                kspace = fastmri.fft2c(image)
            elif self.pe_lines < 320:
                image = fastmri.ifft2c(kspace)
                pady = (320 - self.pe_lines) // 2
                image = torch.view_as_real(torch.nn.functional.pad(torch.view_as_complex(image),[0,0,pady,pady],mode='constant',value=1e-15))
                kspace = fastmri.fft2c(image)
            target = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)))

        kspace /= self.scalars
        target /= self.scalars
        #产生class_label view_label
        view_label = 8 # brain axial (transversal)
        
        class_label = self.class_label
        class_label = np.array([class_label])
        #更新index
        self.idxz += 1
        if self.idxz >= self.current_data.shape[0]:
            self.idxz = 0
            self.idxS += 1
        if self.idxS >= len(self.file_list):
            self.idxS = 0
        Npe = self.pe_lines if self.pe_lines <= 320 else 320
        return kspace, target, self.mask, Nsample,Nslice,class_label,self.filename,Npe,view_label


#将两个数据集拼成一个数据集
#其实可以用torch的ConcatDataset，但是最开始我还不知道可以用torch的
class ConcatDataset(Dataset):
    def __init__(self,dataset_list:list) -> None:
        super().__init__()
        self.dataset_list = dataset_list
        num_set = len(self.dataset_list)
        self.length = 0
        for dataset in self.dataset_list:
            self.length += len(dataset) 
        if num_set == 2:
            self.index_table_pre = [0 for _ in range(self.length//2)]
            self.index_table_pre[0:len(self.dataset_list[1])//2] = [1 for _ in range(len(self.dataset_list[1])//2)]
            random.shuffle(self.index_table_pre)
            self.index_table = [None for _ in range(self.length)]
            self.index_table[0::2] = self.index_table_pre
            self.index_table[1::2] = self.index_table_pre
        else:
            self.index_table_pre = [0 for _ in range(self.length//2)]
            begin = 0
            for idxSet in range(num_set):
                self.index_table_pre[begin:begin+len(self.dataset_list[idxSet])//2] = [idxSet for _ in range(len(self.dataset_list[idxSet])//2)]
                begin = len(self.dataset_list[idxSet])//2
            random.shuffle(self.index_table_pre)
            self.index_table = [None for _ in range(self.length)]
        
            self.index_table[0::2] = self.index_table_pre
            self.index_table[1::2] = self.index_table_pre
            
        return
    def shuffle(self):
        for dataset in self.dataset_list:
            dataset.shuffle()
        random.shuffle(self.index_table_pre)
        self.index_table = [None for _ in range(self.length)]
        self.index_table[0::2] = self.index_table_pre
        self.index_table[1::2] = self.index_table_pre
        return 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.dataset_list[self.index_table[index]].__getitem__(index)



