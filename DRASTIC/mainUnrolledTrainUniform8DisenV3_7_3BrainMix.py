import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from varnet import *
from torch.optim import lr_scheduler
from cmrxrecondataset import *
from unetdisen import SamplingTrajEstimationV4_6
from brainmridataset import M4RawDataset,ROOT_DIR_VAL,ROOT_DIR_TRAIN,ConcatDataset,FastMRIDataset
from brainmridataset import ROOT_DIR_TRAIN_FASTMRI
# Multi-contrast实验 v3.7.3 copied from v3.7  加入 fastmri
#v3.6 DisenRecon + sampling pattern optimization
BATCH_SIZE = 2
LR = (2e-4) # default: 1e-4
LR_TRAJ = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.5
START_EPOCH=0
EPOCH=60
SAVE_PER_EPOCH = 5
UNDERSAMPLE = 'Uniform8'
CONTRASTS = ['Mapping','Cine','Aorta']
modelDir = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,)  # 模型保存目录
logDir  = os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/",UNDERSAMPLE,"LogsV3_6")
modelname = 'VarNetV3_7_4TwoSetUniform4Ep60.pth' #'VarNetV3_7Uniform4Ep60.pth'
classifiermodelname = 'ClassifierV3_7_4TwoSetUniform4Ep60.pth'#'ClassifierV3_7Uniform4Ep60.pth'
modelTrajname = "ModelTrajV3_7_4TwoSetUniform4Ep60.pth"#"ModelTrajV3_7_1Uniform4Ep15.pth"  
if not os.path.exists(logDir):
    os.mkdir(logDir)
print(UNDERSAMPLE)
writer=SummaryWriter(logDir)
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")
#构造数据集
trainDataset1 = CMRxReconDatasetDisenV4Traj(pattern=UNDERSAMPLE,withdraw=True,size_limit=None,class_labels=CONTRASTS,dataset_type='train')
print("Size: "+str(len(trainDataset1)))
trainDataset2 = M4RawDataset(mode='train',pattern=UNDERSAMPLE,root_dir=ROOT_DIR_TRAIN)
trainDataset3 = FastMRIDataset(pattern=UNDERSAMPLE,mode='train',root_dir=ROOT_DIR_TRAIN_FASTMRI)
print("Brain Dataset Size: "+str(len(trainDataset2)))
print("Brain Dataset2 Size: "+str(len(trainDataset3)))


trainDataset = ConcatDataset([trainDataset1,trainDataset2,trainDataset3])

#trainDataloader2 = DataLoader(trainDataset2,batch_size=BATCH_SIZE,shuffle=False)
trainDataloader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=False)
#trainDataloader3 = DataLoader(trainDataset3,batch_size=BATCH_SIZE,shuffle=False)
valDataset = CMRxReconDatasetDisenV4Traj(pattern=UNDERSAMPLE,withdraw=False,size_limit=None,class_labels=CONTRASTS,dataset_type='validation')
print("Size: "+str(len(valDataset)))
valDataloader = DataLoader(valDataset,batch_size=BATCH_SIZE,shuffle=False)

valDataset2 = M4RawDataset(mode='validation',pattern=UNDERSAMPLE,root_dir=ROOT_DIR_VAL)
print("Brain Dataset Size: "+str(len(valDataset2)))
valDataloader2 = DataLoader(valDataset2,batch_size=BATCH_SIZE,shuffle=False)
#构建模型
model = VarNetDisenV2(num_cascades=12,sens_chans=8,sens_pools=4,chans=18,pools=4,version='V3_6').to(device)
classifier = Classifier(size=16,out_features=12).to(device)

#model.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4") ,modelname)))

modelTraj = SamplingTrajEstimationV4_6(input_n=10).to(device)
#modelTraj.load_state_dict(torch.load(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/Uniform4",modelTrajname)))
modelTraj.train()
'''
modelnameList = ["/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_21_12_33/ckpt_cont_encoder.pth",\
    "/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_21_12_33/ckpt_style_encoder.pth",\
    "/home/ubuntu/shared/Zhihao/Disentanglement/ckpt/BraTS/MultimodalModel/2024_11_12_9_55/ckpt_sem.pth"]

model.load_ckpt_V3_6(modelList=modelnameList)
'''
model.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4"),modelname)))
classifier.load_state_dict(torch.load(os.path.join(os.path.join("/home/ubuntu/shared/Zhihao/CMRxRecon/VarnetModelDisen/","Uniform4"),classifiermodelname)))
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
    #    T1 = time.time()
        ksp = sampleDict[0].to(torch.float32).to(device) #1,4,640,320,2
        y = sampleDict[1].to(device) #1,640,320
    #    if ksp.shape[2] != 320:
    #        print(sampleDict[6])
        Nsample = sampleDict[3]  #2,1
        Nslice = sampleDict[4]  #2,1
        class_label = sampleDict[5]  #2,1
        class_label_onehot = torch.nn.functional.one_hot((class_label),num_classes=12).to(torch.float32)
        class_label = class_label.to(torch.float32).to(device)
        mask_org = (sampleDict[2].to(torch.bool)).to(device) #2,256,384
        #num_lines = [int((mask_org[0,...].sum())/384),int((mask_org[1,...].sum())/384)]  #num_lines条线
        num_lines = [int((mask_org[0,...].sum())/mask_org.shape[-1])]  #num_lines条线
               
        pe_lines = sampleDict[7]
        view_label = sampleDict[8]
        view_label_onehot = torch.nn.functional.one_hot(view_label,num_classes=10).to(torch.float32).to(device)
    #    T2 = time.time()
        masked_ksp,mask = modelTraj(ksp,view_label_onehot,num_lines=num_lines,total_lines=pe_lines)
        mask = mask.unsqueeze(-1)
    #    T3 = time.time()
        #取的index重新洗牌
        y_pred,cont_list,style_list,sens_maps = model(masked_ksp,mask,inputs_type = class_label)
        #y_pred: 2x256x384
        #cont_list: length = 12 shape=2,4,256,384
        #style_list: length = 12 shape=2,16
    #    T4 = time.time()
        #计算loss
        contrat_label={"Nsample":Nsample,"Ns":Nslice,"contrast":class_label,"class":class_label_onehot.squeeze(1)}
        loss,loss_rec = loss_fn(y_pred,y,cont_list,style_list,contrat_label,classifier,sens_maps)

        loss = loss/accumulation_steps;num_batch += 1
    #    T5 = time.time()
        loss.backward()  
    #    if torch.isnan(loss):
    #        print(sampleDict[6]) #输出文件名
        #可能出现nan的原因： loss计算不正确；梯度爆炸；（前期CMRxRecon训练已证明剔除异常后的数据无问题）
        #梯度裁剪
        #应当做梯度裁剪 原来的threshold是1.0  a#加2.0梯度裁剪1500step后似乎没有要收敛的迹象
        #下一步计划：1. 将loss的所有权重都减小；2. 改小学习率到1e-4 3. 减小参数正则化
    #    T6 = time.time()
        if (batch+1) % accumulation_steps == 0 or (batch + 1 == len(dataloader)):
            #torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=2.0)
            if epoch < 30:  #训练后期不更新
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
      #  print(T2-T1,end=' ')
      #  print(T3-T2,end=' ')
      #  print(T4-T3,end=' ')
      #  print(T5-T4,end=' ')
      #  print(T6-T5)
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
#把前面的学习率调整补上
for t in range(0,START_EPOCH):
    scheduler.step(t)
    scheduler_traj.step(t)

for t in range(START_EPOCH,EPOCH):
    print("LR decay for both models")
    print(f"Epoch {t+1}\n-------------------------------")
  #  if t == START_EPOCH:
 #       for i, param_group in enumerate((optimizer.param_groups)):
   #         param_group['lr'] = LR * (LR_GAMMA)**3
   #     for i, param_group in enumerate((optimizer_traj.param_groups)):
   #         param_group['lr'] = LR_TRAJ * (LR_GAMMA)**3
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    trainDataset.shuffle()
    trainLoop(trainDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    print("Validation: ")
    validationLoop(valDataloader2,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    validationLoop(valDataloader,model,disenloss,optimizer,writer,device,t,classifier=classifier)
    
    #保存网络参数  v3.7.1 与v3.7相同，但保存了轨迹模型参数 v3.7.3 增加fastMRI  v3.7.4 model_traj也引入学习率衰减
    #V3.7.5 traj model后期不更新  v3.7.6 从ep7开始
    if (t+1) %SAVE_PER_EPOCH==0:
        torch.save(model.state_dict(), modelDir+'/VarNetV3_7_5TwoSet'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth') 
        torch.save(classifier.state_dict(), modelDir+'/ClassifierV3_7_5TwoSet'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')    # 只保存网络中的参数
        torch.save(modelTraj.state_dict(),modelDir+'/ModelTrajV3_7_5TwoSet'+UNDERSAMPLE+'Ep'+str(t+1)+'.pth')
    scheduler.step(t)
    scheduler_traj.step(t)  
writer.close()
print("Done!")


#下一步工作：增加Validation代码；重写2.5D MoDL的Dataset