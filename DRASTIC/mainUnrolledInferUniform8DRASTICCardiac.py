import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from varnet import *
from torch.optim import lr_scheduler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from cmrxrecondataset import *
from unetdisen import SamplingTrajEstimationV4_6
# Multi-contrast实验 V3.6 推理测试代码 copied from V3.4.2
BATCH_SIZE = 1
LR = (2e-4) # default: 1e-4
LR_STEP_SIZE = 100
LR_GAMMA = 0.5
START_EPOCH=10
EPOCH=60
SAVE_PER_EPOCH = 2
UNDERSAMPLE = 'Uniform8'
CONTRASTS = ['Mapping','Cine','Aorta']
modelDir = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",'Uniform8',)  # 模型保存目录
logDir  = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,"Logs")
epoch = '60'
modelname =  'VarNetV3_7_4Uniform8Ep'+epoch+'.pth'
classifiermodelname = 'ClassifierV3_7_4Uniform8Ep'+epoch+'.pth'
#modelTrajname = "ModelTrajV3_7TwoSetUniform4Ep20.pth"
modelTrajname = 'ModelTrajV3_7_4Uniform8Ep'+epoch+'.pth'  #
T1contrastIndexList = [0,1,2,3,4,5,6,7,8]
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cpu")
#构造数据集
#valDataset = CMRxReconDatasetDisenTest(pattern=UNDERSAMPLE,withdraw=True,size_limit=200,class_labels=CONTRASTS)

valDataset = CMRxReconDatasetDisenV4Traj(pattern=UNDERSAMPLE,withdraw=False,size_limit=140,class_labels=CONTRASTS,dataset_type='test')
print("Size: "+str(len(valDataset)))
# subList = []
# for idx,item in enumerate(valDataset.file_list):
#     if 'T1map' in item['filename']:
#         print(item['filename'])
#         subList.append(item)
valDataloader = DataLoader(valDataset,batch_size=BATCH_SIZE,shuffle=False)
#构建模型
model = VarNetDisenV2(num_cascades=12,sens_chans=8,sens_pools=4,chans=18,pools=4,version='V3_6').to(device)
classifier = Classifier(size=16,out_features=12).to(device)
model.load_state_dict(torch.load(os.path.join(modelDir,modelname),map_location=device))
classifier.load_state_dict(torch.load(os.path.join(modelDir,classifiermodelname),map_location=device))
model.train()
modelTraj = SamplingTrajEstimationV4_6(input_n=7).to(device)
modelTraj.load_state_dict(torch.load(os.path.join(modelDir,modelTrajname),map_location=device))
#modelTraj.load_state_dict(torch.load("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/Uniform4/ModelTrajV3_7_1Uniform4Ep15.pth"))
#优化器
optimizer=torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=LR,weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
#输出参数总数量
total = sum([param.nelement() for param in model.parameters()])
print("Model saved directory: "+str(modelDir))
print("Number of parameter: %.2fM" % (total/1e6))
disenloss = DisenLoss(device=device)


saveDir = "/home/ubuntu/shared/Zhihao/Disentanglement/ReconDisenR8V3_7/"

Nsample_pool = []
trajMode = 'estimation' #'estimation'
def validationLoop(dataloader:DataLoader,model:VarNetDisenV2,loss_fn:DisenLoss,optimizer,writer,device,epoch,classifier):
    size = len(dataloader.dataset)
    num_batch = 0
    total_loss = 0
    with torch.no_grad():
        for batch, sampleDict in enumerate(dataloader):
            
  
            ksp = sampleDict[0].to(torch.float32).to(device) #1,4,640,320,2
            #masked_ksp = sampleDict[0].to(torch.float32).to(device) #1,4,640,320,2
            y = sampleDict[1].to(device) #1,640,320
            Nsample = sampleDict[3]  #2,1
            Nslice = sampleDict[4]  #2,1
            class_label = sampleDict[5]  #2,1
            
            class_label = class_label.to(torch.float32).cuda()
            mask_org = (sampleDict[2].to(torch.bool)).to(device) #2,256,384
            num_lines = [int((mask_org[0,...].sum())/384)]  #num_lines条线

            pe_lines = sampleDict[7]
            view_label = sampleDict[8]
            view_label_onehot = torch.nn.functional.one_hot(view_label,num_classes=7).to(torch.float32).to(device)
  
        #    mask = (sampleDict[2].to(torch.bool)).to(device) #1,640,320
        #    if BATCH_SIZE == 1:
        #        mask = mask.unsqueeze(0)
        #    else:
        #        mask = mask.unsqueeze(1)
            if trajMode == 'estimation':
                masked_ksp,mask = modelTraj(ksp,view_label_onehot,num_lines=num_lines,total_lines=pe_lines)
                mask = mask.unsqueeze(-1)
            elif trajMode == 'no':
                mask = mask_org.unsqueeze(0)
                mask = mask.unsqueeze(-1)
                masked_ksp = ksp * mask
            


            Nt = sampleDict[6]  #2,1
            print("batch"+str(batch))
       #     if batch <= 3864:
       #         continue
            if (class_label == 0 or class_label == 1) and Nt >= 2:
                continue
            #class_label_onehot = torch.nn.functional.one_hot((class_label),num_classes=12).to(torch.float32)
            y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label,num_low_frequencies=8)
    
            #random.shuffle(T1contrastIndexList)  #取的index重新洗牌
            # if Nsample[0].item() not in Nsample_pool:
            #     print(Nsample[0].item(),end=":")
            #     print(valDataset.file_list[Nsample[0]]['filename'])
            #     Nsample_pool.append(Nsample[0].item())
            
            # if Nsample[1].item() not in Nsample_pool:
            #     print(Nsample[1].item(),end=":")
            #     print(valDataset.file_list[Nsample[1]]['filename'])
            #     Nsample_pool.append(Nsample[1].item())
            # continue

        #    y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label)
            

            metrics = np.zeros((masked_ksp.shape[0],3))
            for idx in range(masked_ksp.shape[0]):
                metrics[idx,0] = psnr(y[idx,...].cpu().numpy(),y_pred[idx,...].cpu().numpy(),data_range = y[idx,...].cpu().numpy().max())
                metrics[idx,1] = ssim(y[idx,...].cpu().numpy(),y_pred[idx,...].cpu().numpy(),data_range = y[idx,...].cpu().numpy().max())
                metrics[idx,2] = nmse(y[idx,...].cpu().numpy(),y_pred[idx,...].cpu().numpy())
            print(metrics)
            #保存图像
            save_dict = {"pred":y_pred.cpu().numpy(),"y":y.cpu().numpy(),"cont":cont_list[-1].cpu().numpy(),\
                "style":style_list[-1].cpu().numpy(),"Nsample":Nsample,"Nslice":Nslice,"contrast":class_label.cpu().numpy(),\
                    "metrics":metrics,"m":mask[0,0,:,192,0].cpu().numpy(),"view":view_label.cpu().numpy()}                    
            #savemat(os.path.join(saveDir,"reconBatch"+"Disen"+str(batch)+".mat"),save_dict,do_compression=True)
            
            num_batch += 1
            #total_loss += loss.item()
            #if batch % 1 == 0:
            #    loss, current = loss.item(), batch * len(masked_ksp)+1
            #    print(f"Loss: {loss:>6f} Loss rec: {loss_rec.item():>6f}  [{current:>5d}/{size:>5d}]")
           # if batch > 100:
            #    break
            
    #total_loss = total_loss / num_batch

for t in range(1):
    print("Validation: ")
    validationLoop(valDataloader,model,disenloss,optimizer,None,device,t,classifier=classifier)
    

print("Done!")

