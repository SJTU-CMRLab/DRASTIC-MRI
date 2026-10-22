import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from varnet import *
from torch.optim import lr_scheduler
from cmrxrecondataset import *
from unetdisen import SamplingTrajEstimationV4_7
# ACS input for trajectory estimation
# DRASTIC-A at R=8
BATCH_SIZE = 2
LR = (2e-4) # default: 1e-4
LR_TRAJ = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.5
START_EPOCH=5
EPOCH=60
SAVE_PER_EPOCH = 5
UNDERSAMPLE = 'Uniform8'
CONTRASTS = ['Mapping','Cine','Aorta']
modelDir = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,)  # 模型保存目录
logDir  = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,"LogsACSAuto")

modelname =  'VarNetV3_7_4Uniform8Ep60.pth'
classifiermodelname = 'ClassifierV3_7_1Uniform8Ep60.pth'
modelTrajname = 'ModelTrajV3_7ACSUniform4Ep30.pth'
if not os.path.exists(logDir):
    os.mkdir(logDir)
print(UNDERSAMPLE)
writer=SummaryWriter(logDir)
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")
#构造数据集
trainDataset = CMRxReconDatasetDisenV4Traj(pattern=UNDERSAMPLE,withdraw=True,size_limit=None,class_labels=CONTRASTS,dataset_type='train')
print("Size: "+str(len(trainDataset)))
trainDataloader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=False)
valDataset = CMRxReconDatasetDisenV4Traj(pattern=UNDERSAMPLE,withdraw=False,size_limit=None,class_labels=CONTRASTS,dataset_type='validation')
print("Size: "+str(len(valDataset)))
valDataloader = DataLoader(valDataset,batch_size=BATCH_SIZE,shuffle=False)
#构建模型
model = VarNetDisenV2(num_cascades=12,sens_chans=8,sens_pools=4,chans=18,pools=4,version='V3_6').to(device)
classifier = Classifier(size=16,out_features=12).to(device)
#model.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform8") ,modelname)))

modelTraj = SamplingTrajEstimationV4_7(input_n=256,layer_list=[128,64,64,128]).to(device)
#modelTraj.load_state_dict(torch.load(os.path.join(modelDir,modelTrajname)))
modelTraj.train()


#classifier.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform8"),classifiermodelname)))
model.train()
classifier.train()
#优化器
optimizer=torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
optimizer_traj = torch.optim.Adam(modelTraj.parameters(),lr=LR_TRAJ)
scheduler_traj = lr_scheduler.StepLR(optimizer_traj, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
#输出参数总数量
total = sum([param.nelement() for param in model.parameters()])
print("Model saved directory: "+str(modelDir))
print("Number of parameter: %.2fM" % (total/1e6))
disenloss = DisenLoss(device=device)
#训练循环
def trainLoop(dataloader:DataLoader,model,loss_fn:DisenLoss,optimizer:torch.optim.Optimizer,writer,device,epoch,classifier):
    size = len(dataloader.dataset)
    num_batch = 0
    total_loss = 0;accumulation_steps = 4
    for batch, sampleDict in enumerate(dataloader): 
        ksp = sampleDict[0].to(torch.float32).to(device) #2,10,256,384,2
        #extract ACS data from kspace data
        with torch.no_grad():
            nb,nc,npe,nro,_ = ksp.shape
            acs = ksp[:,:,npe//2-8:npe//2+8,nro//2-8:nro//2+8,:]#2,10,16,16,2
            #to image domain
            img_acs = fastmri.ifft2c(acs)
            #coil combination using rss
            img_acs = fastmri.rss_complex(img_acs,dim=1)  #2,16,16
            #归一化
            img_acs = img_acs / 20.0
            img_acs = img_acs.reshape(nb,256)          
        y = sampleDict[1].to(device) #1,640,320

        Nsample = sampleDict[3]  #2,1
        Nslice = sampleDict[4]  #2,1
        class_label = sampleDict[5]  #2,1
        class_label_onehot = torch.nn.functional.one_hot((class_label),num_classes=12).to(torch.float32)
        class_label = class_label.to(torch.float32).cuda()
        mask_org = (sampleDict[2].to(torch.bool)).to(device) #2,256,384
        #num_lines = [int((mask_org[0,...].sum())/384),int((mask_org[1,...].sum())/384)]  #num_lines条线
        num_lines = [int((mask_org[0,...].sum())/384)]  #num_lines条线
               
        pe_lines = sampleDict[7]
       # view_label = sampleDict[8]
       # view_label_onehot = torch.nn.functional.one_hot(view_label,num_classes=7).to(torch.float32).to(device)#2,7
  
        masked_ksp,mask = modelTraj(ksp,img_acs,num_lines=num_lines,total_lines=pe_lines)
        mask = mask.unsqueeze(-1)

        #取的index重新洗牌
        y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label)
        #y_pred: 2x256x384
        #cont_list: length = 12 shape=2,4,256,384
        #style_list: length = 12 shape=2,16
        #计算loss
        contrat_label={"Nsample":Nsample,"Ns":Nslice,"contrast":class_label,"class":class_label_onehot.squeeze(1)}
        loss,loss_rec = loss_fn(y_pred,y,cont_list,style_list,contrat_label,classifier,sens_maps)

        loss = loss/accumulation_steps;num_batch += 1
        loss.backward()  
        #可能出现nan的原因： loss计算不正确；梯度爆炸；（前期CMRxRecon训练已证明剔除异常后的数据无问题）
        #梯度裁剪
        #应当做梯度裁剪 原来的threshold是1.0  a#加2.0梯度裁剪1500step后似乎没有要收敛的迹象
        #下一步计划：1. 将loss的所有权重都减小；2. 改小学习率到1e-4 3. 减小参数正则化
        if (batch+1) % accumulation_steps == 0 or (batch + 1 == len(dataloader)):
            #torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=2.0)
            if epoch < 30:
                torch.nn.utils.clip_grad_value_(parameters=modelTraj.parameters(), clip_value=1.0)
                optimizer_traj.step()
                optimizer_traj.zero_grad()

            optimizer.step()
            optimizer.zero_grad()  #应当在loss反向传播前先清零梯度
        total_loss += loss.item()
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(masked_ksp)+1
            print(f"Loss: {loss*accumulation_steps:>6f} Loss rec: {loss_rec.item():>6f}  [{current:>5d}/{size:>5d}]")
            #if loss > 1:
            #    pass
            #    print('\n')
            writer.add_scalar('Loss/TrainlossPerStep',loss,epoch*size/BATCH_SIZE + batch)

    total_loss = total_loss / num_batch
    writer.add_scalar('Loss/Trainloss',total_loss,epoch)
    

def validationLoop(dataloader:DataLoader,model,loss_fn:DisenLoss,optimizer,writer,device,epoch,classifier):
    size = len(dataloader.dataset)
    num_batch = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch, sampleDict in enumerate(dataloader):
            masked_ksp = sampleDict[0].to(torch.float32).to(device) #1,4,640,320,2
            y = sampleDict[1].to(device) #1,640,320
            mask = (sampleDict[2].to(torch.bool)).to(device) #1,640,320
            if BATCH_SIZE == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(-1)
            Nsample = sampleDict[3]  #2,1
            Nslice = sampleDict[4]  #2,1
            class_label = sampleDict[5]  #2,1
            class_label_onehot = torch.nn.functional.one_hot((class_label),num_classes=12).to(torch.float32)
            class_label = class_label.to(torch.float32).cuda()

            y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label)
        #y_pred: 2x256x384
        #cont_list: length = 12 shape=2,4,256,384
        #style_list: length = 12 shape=2,16
        #计算loss
            contrat_label={"Nsample":Nsample,"Ns":Nslice,"contrast":class_label,"class":class_label_onehot.squeeze(1)}
            loss,loss_rec = loss_fn(y_pred,y,cont_list,style_list,contrat_label,classifier,sens_maps)

            num_batch += 1
            total_loss += loss.item()
            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(masked_ksp)+1
                print(f"Loss: {loss:>6f} Loss rec: {loss_rec.item():>6f}  [{current:>5d}/{size:>5d}]")
            
    total_loss = total_loss / num_batch
    writer.add_scalar('Loss/Valloss',total_loss,epoch)

#把前面的学习率调整补上
for t in range(0,START_EPOCH):
    scheduler.step(t)
    scheduler_traj.step(t)

for t in range(START_EPOCH,EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    trainDataset.shuffle()
    trainLoop(trainDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    print("Validation: ")
    validationLoop(valDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    
    #保存网络参数  v3.7NoStyleLoss 去除Style loss 和auxillary loss，研究其对于style feature 解耦的作用
    if (t+1) %SAVE_PER_EPOCH==0:
        torch.save(model.state_dict(), modelDir+'/VarNetV3_7ACS'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth') 
        torch.save(classifier.state_dict(), modelDir+'/ClassifierV3_7ACS'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')    # 只保存网络中的参数
        torch.save(modelTraj.state_dict(),modelDir+'/ModelTrajV3_7ACS'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')
    scheduler.step(t)
    scheduler_traj.step(t)
writer.close()
print("Done!")

