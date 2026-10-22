import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from varnet import *
from torch.optim import lr_scheduler
from cmrxrecondataset import *
from unetdisen import SamplingTrajEstimationV4_6
# Multi-contrast实验 v3.7 copied form v3.6
#v3.6 DisenRecon + sampling pattern optimization
BATCH_SIZE = 2
LR = (2e-4) # default: 1e-4
LR_TRAJ = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.5
START_EPOCH=4
EPOCH=60
SAVE_PER_EPOCH = 5
UNDERSAMPLE = 'Uniform8'
CONTRASTS = ['Mapping','Cine','Aorta']
modelDir = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,)  # 模型保存目录
logDir  = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,"LogsV3_6")
modelname = 'VarNetV3_7Uniform4Ep60.pth'
classifiermodelname = 'ClassifierV3_7_1Uniform8Ep60.pth'
modelTrajname = "ModelTrajV3_7_1Uniform8Ep60.pth"  #15
if not os.path.exists(logDir):
    os.mkdir(logDir)
print(UNDERSAMPLE)
writer=SummaryWriter(logDir)
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
#model.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4",),modelname)))

classifier = Classifier(size=16,out_features=12).to(device)
#model.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4") ,modelname)))
#classifier.load_state_dict(torch.load(os.path.join(modelDir,classifiermodelname)))
modelTraj = SamplingTrajEstimationV4_6(input_n=7).to(device)
#modelTraj.load_state_dict(torch.load(os.path.join(modelDir,modelTrajname)))
modelTraj.train()

modelnameList = ["/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_21_12_33/ckpt_cont_encoder.pth",\
    "/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_21_12_33/ckpt_style_encoder.pth",\
    "/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_12_9_55/ckpt_sem.pth"]

#model.load_ckpt_V3_6(modelList=modelnameList)
#classifier.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4"),classifiermodelname)))
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
def trainLoop(dataloader:DataLoader,model:VarNetDisenV2,loss_fn:DisenLoss,optimizer:torch.optim.Optimizer,writer,device,epoch,classifier):
    size = len(dataloader.dataset)
    num_batch = 0
    total_loss = 0;accumulation_steps = 4
    for batch, sampleDict in enumerate(dataloader): 
        ksp = sampleDict[0].to(torch.float32).to(device) #1,4,640,320,2
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
        view_label = sampleDict[8]
        view_label_onehot = torch.nn.functional.one_hot(view_label,num_classes=7).to(torch.float32).to(device)
  
        masked_ksp,mask = modelTraj(ksp,view_label_onehot,num_lines=num_lines,total_lines=pe_lines)
        mask = mask.unsqueeze(-1)

        #取的index重新洗牌
        y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label,num_low_frequencies=9)
        #y_pred: 2x256x384
        #cont_list: length = 12 shape=2,4,256,384
        #style_list: length = 12 shape=2,16
        #计算loss
        contrat_label={"Nsample":Nsample,"Ns":Nslice,"contrast":class_label,"class":class_label_onehot.squeeze(1)}
        loss,loss_rec = loss_fn(y_pred,y,cont_list,style_list,contrat_label,classifier,sens_maps)

        loss = loss/accumulation_steps;num_batch += 1
        loss.backward()  
        
        if (batch+1) % accumulation_steps == 0 or (batch + 1 == len(dataloader)):
            #torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=2.0)
            if True: #epoch < 30:
                torch.nn.utils.clip_grad_value_(parameters=modelTraj.parameters(), clip_value=1.0)
                optimizer_traj.step()
            optimizer_traj.zero_grad()

            optimizer.step()
            optimizer.zero_grad()  #应当在loss反向传播前先清零梯度
        total_loss += loss.item()
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(masked_ksp)+1
            print(f"Loss: {loss*accumulation_steps:>6f} Loss rec: {loss_rec.item():>6f}  [{current:>5d}/{size:>5d}]")

            writer.add_scalar('Loss/TrainlossPerStep',loss,epoch*size/BATCH_SIZE + batch)

    total_loss = total_loss / num_batch
    writer.add_scalar('Loss/Trainloss',total_loss,epoch)
    

def validationLoop(dataloader:DataLoader,model:VarNetDisenV2,loss_fn:DisenLoss,optimizer,writer,device,epoch,classifier):
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

for t in range(START_EPOCH,EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    trainDataset.shuffle()
    trainLoop(trainDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    print("Validation: ")
    validationLoop(valDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    
    #保存网络参数  v3.7.1 与v3.7相同，但保存了轨迹模型参数 v3.7.2 仅在前20 epoch进行轨迹优化 v3.7.4 轨迹模型用4.6版本，且重新训
    if (t+1) %SAVE_PER_EPOCH==0:
        torch.save(model.state_dict(), modelDir+'/VarNetV3_7_4'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth') 
        torch.save(classifier.state_dict(), modelDir+'/ClassifierV3_7_4'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')    # 只保存网络中的参数
        torch.save(modelTraj.state_dict(),modelDir+'/ModelTrajV3_7_4'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')
    scheduler.step(t)
    scheduler_traj.step(t)  #记得加上学习率scheme
writer.close()
print("Done!")


#下一步工作：增加Validation代码；重写2.5D MoDL的Dataset