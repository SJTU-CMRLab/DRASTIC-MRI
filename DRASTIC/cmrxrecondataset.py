import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from main import readfile2numpyv2,savenumpy2mat,readmaskfile2numpy
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat,savemat
import random
import numpy as np
from fastmri.data import transforms as T
import fastmri
import torch.nn as nn

FILE_DIR = "/home/ubuntu/shared/Zhihao/CMRxRecon/home2/Raw_data/MICCAIChallenge2024/ChallengeData/MultiCoil/"
CONTRASTS = ['Aorta','Cine','Mapping','Tagging']
#CONTRASTS = ['Cine']
NC = 10
# data cleaning
WITHDRAWAL_LIST = ['CineP100','CineP109','AortaP065','AortaP077','AortaP105','AortaP117','TaggingP087','TaggingP105','MappingP065','MappingP064']  #最后一个Mapping065 064是后加上去的，数据并没有问题

class CMRxReconDataset(Dataset):
    def __init__(self, mode:str = 'TrainingSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    maskfilename = os.path.join(mask_dir,dir,file[:-4]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
        #savelist=np.array(self.file_list)
        #np.save('file_list.npy',savelist)
        print('Dataset initialized.')
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]

            self.current_data = readfile2numpyv2(filename)
            self.current_mask = readmaskfile2numpy(maskname)

        kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        #求取rss
        slice_image = fastmri.ifft2c(T.to_tensor(kspace)) 
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace_u = kspace * self.current_mask
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        #更新index
        self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxS += 1
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, self.current_mask

T2mapLabelList = [5,6,7]
T1contrastIndexList = [0,1,2,3,4,5,6,7,8]
class CMRxReconDatasetDisen(Dataset):
    def __init__(self, mode:str = 'TrainingSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        self.filename = ""
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if withdraw == True and os.path.exists('file_listV2_2.npy'):
            self.file_list = np.load('file_listV2_2.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceV2_2.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    #if 'T2map' in filename:
                    #    continue  #为了简便，暂时只做T1map
                    maskfilename = os.path.join(mask_dir,dir,file[:-4]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
        #savelist=np.array(self.file_list)
        #np.save('file_list.npy',savelist)
        print('Dataset initialized.')
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            self.filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]

            self.current_data = readfile2numpyv2(self.filename)
            self.current_mask = readmaskfile2numpy(maskname)
        if 'T1map' in self.filename:
            kspace = self.current_data[T1contrastIndexList[self.idxt],self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        else:
            kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3
        
        kspace = torch.from_numpy(kspace)
        #对kspace 进行padding
        pad_num = (256 - kspace.shape[-2]) // 2
        if pad_num > 0:            
            pad = nn.ZeroPad2d(padding=[0,0,pad_num,pad_num])
            kspace = pad(kspace)
            mask = pad(torch.from_numpy(self.current_mask))
        else:
            mask = self.current_mask
        #求取rss
        slice_image = fastmri.ifft2c(torch.view_as_real(kspace))
        if slice_image.shape[-2] > 384:  #10,256,384,2
            #中心裁剪
            pad_x = (slice_image.shape[-2] - 384) // 2 
            slice_image = slice_image[:,:,pad_x:pad_x+384,:]
            mask = mask[:,pad_x:pad_x+384]
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace = torch.view_as_complex(fastmri.fft2c(slice_image))  #变换回k空间
        kspace_u = kspace * mask
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        Nslice = np.array([self.idxz])
        Nsample = np.array([self.idxS])
        #判断图像的contrast
        class_label = 0
        if 'Cine' in self.filename:
            class_label = 1
        elif 'T1map' in self.filename:
            if T1contrastIndexList[self.idxt] <= 2:
                class_label = 2
            elif T1contrastIndexList[self.idxt] <= 5:
                class_label = 3
            else:
                class_label = 4
        elif 'T2map' in self.filename:
            class_label = T2mapLabelList[self.idxt]
        elif 'Tagging' in self.filename:
            class_label = 8
        elif 'BlackBlood' in self.filename:
            class_label = 9
        elif 'Flow' in self.filename:
            class_label = 10
        class_label = np.array([class_label])
        
        #更新index
        self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxS += 1
            random.shuffle(T1contrastIndexList)   #换到新样本了则重新洗牌
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, mask, Nsample,Nslice,class_label

class CMRxReconDatasetDisenV4(Dataset):
    def __init__(self, mode:str = 'TrainingSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None,dataset_type='train'):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        self.filename = ""
        self.pattern = pattern
        self.dataset_type = dataset_type
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if withdraw == True and dataset_type == 'train' and os.path.exists('file_listV4.npy'):
            self.file_list = np.load('file_listV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if dataset_type == 'validation' and os.path.exists('file_listValV4.npy'):
            self.file_list = np.load('file_listValV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceValV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if dataset_type == 'test' and os.path.exists('file_listTestV4.npy'):
            self.file_list = np.load('file_listTestV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceTestV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                if dataset_type == 'train': 
                    if (contrast == 'Mapping' and dir[1:] > '050') or (contrast != 'Mapping' and dir[1:] > '025'):
                        continue#训练数据取前50个
                elif dataset_type == 'validation':
                    if (dir[1:] <= '050' or dir[1:] > '052'):
                        continue#验证数据取50-60
                elif dataset_type == 'test':
                    if (dir[1:] < '060' or dir[1:] > '100'):
                        continue
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                #elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                #    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    #if 'T2map' in filename:
                    #    continue  #为了简便，暂时只做T1map
                    maskfilename = os.path.join(mask_dir,dir,file[:-4]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
 
        print('Dataset initialized.')
        #np.save('file_listTestV4.npy',self.file_list)
        #np.save('num_sliceTestV4.npy',np.array(self.num_slices))
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            self.filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]
            if self.pattern != 'Uniform4':
                maskname = maskname.replace('Uniform4',self.pattern)
            self.current_data = readfile2numpyv2(self.filename)
            self.current_mask = readmaskfile2numpy(maskname)
        if 'T1map' in self.filename:
            kspace = self.current_data[T1contrastIndexList[self.idxt],self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        else:
            kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3
        
        kspace = torch.from_numpy(kspace)
        #对kspace 进行padding
        pad_num = (256 - kspace.shape[-2]) // 2
        if pad_num > 0:            
            pad = nn.ZeroPad2d(padding=[0,0,pad_num,pad_num])
            kspace = pad(kspace)
            mask = pad(torch.from_numpy(self.current_mask))
        else:
            mask = self.current_mask
        #求取rss
        slice_image = fastmri.ifft2c(torch.view_as_real(kspace))
        if slice_image.shape[-2] > 384:  #10,256,384,2
            #中心裁剪
            pad_x = (slice_image.shape[-2] - 384) // 2 
            slice_image = slice_image[:,:,pad_x:pad_x+384,:]
            mask = mask[:,pad_x:pad_x+384]
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace = torch.view_as_complex(fastmri.fft2c(slice_image))  #变换回k空间
        kspace_u = kspace * mask  #欠采样
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        Nslice = np.array([self.idxz])
        Nsample = np.array([self.idxS])
        Ntime = np.array([self.idxt])
        #判断图像的contrast
        class_label = 0
        view_label = 0
        if 'Aorta' in self.filename:
            if 'tra' in self.filename:
                view_label = 1 #1:transversal
        if 'Cine' in self.filename:
            class_label = 1
            if 'lvot' in self.filename:
                view_label = 2 #2: lvot
            elif 'lax' in self.filename:
                if Nslice >2:
                    print("Exception")
                    pass
                view_label_list = [3,4,5]  #3:2ch, 4:4ch, 5:3ch
                view_label = view_label_list[self.idxz]
            else:
                view_label = 6  #6: sax
        elif 'T1map' in self.filename:
            view_label = 6
            if T1contrastIndexList[self.idxt] <= 2:
                class_label = 2
            elif T1contrastIndexList[self.idxt] <= 5:
                class_label = 3
            else:
                class_label = 4
        elif 'T2map' in self.filename:
            view_label = 6
            class_label = T2mapLabelList[self.idxt]
        elif 'Tagging' in self.filename:
            class_label = 8
        elif 'BlackBlood' in self.filename:
            class_label = 9
        elif 'Flow' in self.filename:
            class_label = 10
        class_label = np.array([class_label])
        
        #更新index
        self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxS += 1
            if self.dataset_type == 'train':
                random.shuffle(T1contrastIndexList)   #换到新样本了则重新洗牌
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, mask, Nsample,Nslice,class_label,Ntime,view_label


class CMRxReconDatasetDisenV4Traj(Dataset):
    def __init__(self, mode:str = 'TrainingSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None,dataset_type='train'):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        self.filename = ""
        self.pattern = pattern
        self.dataset_type = dataset_type
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if withdraw == True and dataset_type == 'train' and os.path.exists('file_listV4.npy'):
            self.file_list = np.load('file_listV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if dataset_type == 'validation' and os.path.exists('file_listValV4.npy'):
            self.file_list = np.load('file_listValV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceValV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return
        if dataset_type == 'test' and os.path.exists('file_listTestV4.npy'):
            self.file_list = np.load('file_listTestV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceTestV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                if dataset_type == 'train':
                    if (contrast == 'Mapping' and dir[1:] > '050') or (contrast != 'Mapping' and dir[1:] > '025'):
                        continue#训练数据取前50个
                elif dataset_type == 'validation':
                    if (dir[1:] <= '050' or dir[1:] > '052'):
                        continue#验证数据取50-60
                elif dataset_type == 'test':
                    if (dir[1:] < '060' or dir[1:] > '100'):
                        continue
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                #elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                #    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    #if 'T2map' in filename:
                    #    continue  #为了简便，暂时只做T1map
                    maskfilename = os.path.join(mask_dir,dir,file[:-4]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
 
        print('Dataset initialized.')
        #np.save('file_listTestV4.npy',self.file_list)
        #np.save('num_sliceTestV4.npy',np.array(self.num_slices))
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def reset(self)->None:
        self.idxt = 0
        self.idxS = 0
        self.idxt = 0
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            self.filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]
            if self.pattern != 'Uniform4':
                maskname = maskname.replace('Uniform4',self.pattern)
            self.current_data = readfile2numpyv2(self.filename)
            self.current_mask = readmaskfile2numpy(maskname)
        if 'T1map' in self.filename:
            kspace = self.current_data[T1contrastIndexList[self.idxt],self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        else:
            kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3
        
        kspace = torch.from_numpy(kspace)
        pe_lines = kspace.shape[-2]
        #对kspace 进行padding
        pad_num = (256 - kspace.shape[-2]) // 2
        if pad_num > 0:            
            pad = nn.ZeroPad2d(padding=[0,0,pad_num,pad_num])
            kspace = pad(kspace)
            mask = pad(torch.from_numpy(self.current_mask))
        else:
            mask = self.current_mask
        #求取rss
        slice_image = fastmri.ifft2c(torch.view_as_real(kspace))
        
        if slice_image.shape[-2] > 384:  #10,256,384,2
            #中心裁剪
            pad_x = (slice_image.shape[-2] - 384) // 2 
            slice_image = slice_image[:,:,pad_x:pad_x+384,:]
            mask = mask[:,pad_x:pad_x+384]
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace = torch.view_as_complex(fastmri.fft2c(slice_image))  #变换回k空间
        kspace_u = kspace #* mask  #欠采样 这里暂不做欠采
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        Nslice = np.array([self.idxz])
        Nsample = np.array([self.idxS])
        Ntime = np.array([self.idxt])
        #判断图像的contrast和view
        class_label = 0
        view_label = 0 #0: sagittal
        if 'Aorta' in self.filename:
            if 'tra' in self.filename:
                view_label = 1 #1:transversal
        if 'Cine' in self.filename:
            class_label = 1
            if 'lvot' in self.filename:
                view_label = 2 #2: lvot
            elif 'lax' in self.filename:
                if Nslice >2:
                    print("Exception")
                    pass
                view_label_list = [3,4,5]  #3:2ch, 4:4ch, 5:3ch
                view_label = view_label_list[self.idxz]
            else:
                view_label = 6  #6: sax
        elif 'T1map' in self.filename:
            view_label = 6
            if T1contrastIndexList[self.idxt] <= 2:
                class_label = 2
            elif T1contrastIndexList[self.idxt] <= 5:
                class_label = 3
            else:
                class_label = 4
        elif 'T2map' in self.filename:
            view_label = 6
            class_label = T2mapLabelList[self.idxt]
        elif 'Tagging' in self.filename:
            class_label = 8
        elif 'BlackBlood' in self.filename:
            class_label = 9
        elif 'Flow' in self.filename:
            class_label = 10
        class_label = np.array([class_label])

        #更新index
        self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxS += 1
            if self.dataset_type == 'train':
                random.shuffle(T1contrastIndexList)   #换到新样本了则重新洗牌
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, mask, Nsample,Nslice,class_label,Ntime,pe_lines,view_label,self.filename


#Random轨迹支持
class CMRxReconDatasetDisenV4_1Traj(Dataset):
    def __init__(self, mode:str = 'TrainingSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None,dataset_type='train'):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        self.filename = ""
        self.pattern = pattern
        self.dataset_type = dataset_type
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if withdraw == True and dataset_type == 'train' and os.path.exists('file_listV4.npy'):
            self.file_list = np.load('file_listV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return 
        if dataset_type == 'validation' and os.path.exists('file_listValV4.npy'):
            self.file_list = np.load('file_listValV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceValV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return
        if dataset_type == 'test' and os.path.exists('file_listTestV4.npy'):
            self.file_list = np.load('file_listTestV4.npy',allow_pickle=True)
            self.num_slices = np.load('num_sliceTestV4.npy',allow_pickle=True)
            print('Dataset initialized.')
            return
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                if dataset_type == 'train':
                    if (contrast == 'Mapping' and dir[1:] > '050') or (contrast != 'Mapping' and dir[1:] > '025'):
                        continue#训练数据取前50个
                elif dataset_type == 'validation':
                    if (dir[1:] <= '050' or dir[1:] > '052'):
                        continue#验证数据取50-60
                elif dataset_type == 'test':
                    if (dir[1:] < '060' or dir[1:] > '100'):
                        continue
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                #elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                #    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    #if 'T2map' in filename:
                    #    continue  #为了简便，暂时只做T1map
                    maskfilename = os.path.join(mask_dir,dir,file[:-4]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
 
        print('Dataset initialized.')
        #np.save('file_listTestV4.npy',self.file_list)
        #np.save('num_sliceTestV4.npy',np.array(self.num_slices))
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def reset(self)->None:
        self.idxt = 0
        self.idxS = 0
        self.idxt = 0
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            self.filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]
            if self.pattern != 'Uniform4':
                maskname = maskname.replace('Uniform4',self.pattern)
            self.current_data = readfile2numpyv2(self.filename)
            if 'Random' in self.pattern:
                self.current_mask = torch.zeros((256,256)) #待修改
            else:
                self.current_mask = readmaskfile2numpy(maskname)
        if 'T1map' in self.filename:
            kspace = self.current_data[T1contrastIndexList[self.idxt],self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        else:
            kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3
        
        kspace = torch.from_numpy(kspace)
        pe_lines = kspace.shape[-2]
        #对kspace 进行padding
        pad_num = (256 - kspace.shape[-2]) // 2
        if pad_num > 0:            
            pad = nn.ZeroPad2d(padding=[0,0,pad_num,pad_num])
            kspace = pad(kspace)
            mask = pad(torch.from_numpy(self.current_mask))
        else:
            mask = self.current_mask
        #求取rss
        slice_image = fastmri.ifft2c(torch.view_as_real(kspace))
        
        if slice_image.shape[-2] > 384:  #10,256,384,2
            #中心裁剪
            pad_x = (slice_image.shape[-2] - 384) // 2 
            slice_image = slice_image[:,:,pad_x:pad_x+384,:]
            mask = mask[:,pad_x:pad_x+384]
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace = torch.view_as_complex(fastmri.fft2c(slice_image))  #变换回k空间
        kspace_u = kspace #* mask  #欠采样 这里暂不做欠采
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        Nslice = np.array([self.idxz])
        Nsample = np.array([self.idxS])
        Ntime = np.array([self.idxt])
        #判断图像的contrast和view
        class_label = 0
        view_label = 0 #0: sagittal
        if 'Aorta' in self.filename:
            if 'tra' in self.filename:
                view_label = 1 #1:transversal
        if 'Cine' in self.filename:
            class_label = 1
            if 'lvot' in self.filename:
                view_label = 2 #2: lvot
            elif 'lax' in self.filename:
                if Nslice >2:
                    print("Exception")
                    pass
                view_label_list = [3,4,5]  #3:2ch, 4:4ch, 5:3ch
                view_label = view_label_list[self.idxz]
            else:
                view_label = 6  #6: sax
        elif 'T1map' in self.filename:
            view_label = 6
            if T1contrastIndexList[self.idxt] <= 2:
                class_label = 2
            elif T1contrastIndexList[self.idxt] <= 5:
                class_label = 3
            else:
                class_label = 4
        elif 'T2map' in self.filename:
            view_label = 6
            class_label = T2mapLabelList[self.idxt]
        elif 'Tagging' in self.filename:
            class_label = 8
        elif 'BlackBlood' in self.filename:
            class_label = 9
        elif 'Flow' in self.filename:
            class_label = 10
        class_label = np.array([class_label])

        #更新index
        self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxS += 1
            if self.dataset_type == 'train':
                random.shuffle(T1contrastIndexList)   #换到新样本了则重新洗牌
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, mask, Nsample,Nslice,class_label,Ntime,pe_lines,view_label



class CMRxReconDatasetDisenTest(Dataset):
    def __init__(self, mode:str = 'ValidationSet',class_labels:list=CONTRASTS,file_dir:str=FILE_DIR,task:int='Task1',pattern='Uniform4',withdraw=True,size_limit:int = None):
        #mode: TrainingSet, ValidationSet
        #class_labels: Aorta, Cine, BlackBlood, Flow2d, Tagging, Mapping
        #file_dir = 
        #task: Task1, Task2
        self.class_labels = class_labels
        self.file_list = [] #所有文件形成的列表
        self.num_slices = 0
        self.idxS = 0  #index of sample
        self.idxt = 0  #index of time in dynamic MRI or contrast in mapping
        self.idxz = 0  #index of slice
        self.current_data = []
        self.current_mask = []
        self.filename = ""
        #self.file_list = np.load('file_list.npy',allow_pickle=True)
        #self.num_slices = np.load('num_slice.npy')
        #print('Dataset initialized.')
        #return
        if withdraw == True and os.path.exists('file_listaorta.npy'):
            self.file_list = np.load('file_list.npy',allow_pickle=True)
            self.num_slices = np.load('num_slice.npy',allow_pickle=True)[0]
            print('Dataset initialized.')
            return 
        if task == 'Task1':
            masktype = "Mask_Task1"
        elif task == 'Task2':
            masktype = "Mask_Task2"
        for contrast in self.class_labels:
            #构造文件目录名
            if mode == 'TrainingSet':
                current_dir = os.path.join(file_dir,contrast,mode,'FullSample')
            elif mode == 'ValidationSet':
                current_dir = os.path.join(file_dir,contrast,mode,'UnderSample_'+task)
            mask_dir = os.path.join(file_dir,contrast,mode,masktype)
            #找出下面所有文件
            current_list = os.listdir(current_dir)
            for idx, dir in enumerate(current_list):  #dir:样本编号文件夹，如'P002'    
                files = os.listdir(os.path.join(current_dir,dir))
                #读取下面所有文件
                checkname = contrast + dir
                if withdraw is True and checkname in WITHDRAWAL_LIST:  #如果指定了withdraw，则跳过退回的数据
                    continue
                elif withdraw is False and checkname not in WITHDRAWAL_LIST: #如果是withdraw=False，则使用这些数据作为验证
                    continue
                for file in files:
                    filename = os.path.join(current_dir,dir,file)  #k-space数据文件名
                    if pattern not in file:
                        continue
                    #if 'T2map' in filename:
                    #    continue  #为了简便，暂时只做T1map
                    maskfilename = os.path.join(mask_dir,dir,file[:-17]+'_mask_'+pattern+'.mat')
                   # mask_data = readmaskfile2numpy(maskfilename) #shape: Ny,Nx
                    ksp_data = readfile2numpyv2(filename)
                    Nt, Nz, Nc, Ny, Nx = ksp_data.shape
                    self.num_slices += Nz*Nt
                    
                    sample_dict = {"filename":filename,"maskname":maskfilename,"Nz":Nz,"Nt":Nt}
                    self.file_list.append(sample_dict)
                    print("Sample: "+filename)
                    if size_limit is not None and self.num_slices >=size_limit:
                        return
                    #print(ksp_data.shape)  #Nt, Nz, Nc, Ny, Nx
        #savelist=np.array(self.file_list)
        #np.save('file_list.npy',savelist)
        print('Dataset initialized.')
        return
    def __len__(self)->int:
        return self.num_slices
    def shuffle(self)->None:
        random.shuffle(self.file_list)
        return
    def __getitem__(self, index:int):
        if self.idxt == 0 and self.idxz == 0:  #如果读到了新的一层则载入新数据
            self.filename = self.file_list[self.idxS]["filename"]
            maskname = self.file_list[self.idxS]["maskname"]

            self.current_data = readfile2numpyv2(self.filename)
            self.current_mask = readmaskfile2numpy(maskname)
        if 'T1map' in self.filename:
            kspace = self.current_data[T1contrastIndexList[self.idxt],self.idxz,:,:,:]*1e3  #Nc,Ny,Nx
        else:
            kspace = self.current_data[self.idxt,self.idxz,:,:,:]*1e3
        
        kspace = torch.from_numpy(kspace)
        #对kspace 进行padding
        pad_num = (256 - kspace.shape[-2]) // 2
        if pad_num > 0:            
            pad = nn.ZeroPad2d(padding=[0,0,pad_num,pad_num])
            kspace = pad(kspace)
            mask = pad(torch.from_numpy(self.current_mask))
        else:
            mask = self.current_mask
        #求取rss
        slice_image = fastmri.ifft2c(torch.view_as_real(kspace))
        if slice_image.shape[-2] > 384:  #10,256,384,2
            #中心裁剪
            pad_x = (slice_image.shape[-2] - 384) // 2 
            slice_image = slice_image[:,:,pad_x:pad_x+384,:]
            mask = mask[:,pad_x:pad_x+384]
        slice_image_abs = fastmri.complex_abs(slice_image)
        target_image_rss = fastmri.rss(slice_image_abs, dim=0)
        #产生欠采样图像
        kspace = torch.view_as_complex(fastmri.fft2c(slice_image))  #变换回k空间
        kspace_u = kspace * mask
        kspace_u = T.to_tensor(kspace_u)#Nc,Ny,Nx,2
        Nslice = np.array([self.idxz])
        Nsample = np.array([self.idxS])
        #判断图像的contrast
        class_label = 0
        if 'Cine' in self.filename:
            class_label = 1
        elif 'T1map' in self.filename:
            if T1contrastIndexList[self.idxt] <= 2:
                class_label = 2
            elif T1contrastIndexList[self.idxt] <= 5:
                class_label = 3
            else:
                class_label = 4
        elif 'T2map' in self.filename:
            class_label = T2mapLabelList[self.idxt]
        elif 'Tagging' in self.filename:
            class_label = 8
        elif 'BlackBlood' in self.filename:
            class_label = 9
        elif 'Flow' in self.filename:
            class_label = 10
        class_label = np.array([class_label])
        
        #更新index
        self.idxt += 1
        if self.idxt >= self.current_data.shape[0]:
            self.idxt = 0
            self.idxz += 1
        if self.idxz >= self.current_data.shape[1]:
            self.idxz = 0
            self.idxS += 1
            random.shuffle(T1contrastIndexList)   #换到新样本了则重新洗牌
        if self.idxS >= len(self.file_list):
            self.idxS = 0

        return kspace_u, target_image_rss, mask, Nsample,Nslice,class_label


if __name__ == '__main__':
    pass
    