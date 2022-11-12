from dilation_resnet import *
import math
import paddle
from paddle.vision import transforms
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
#添加删除warnings的信息
import warnings
warnings.filterwarnings("ignore")

def training():
    model.train()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_acc=0
    losses = []
    mydata={'myeval_loss':[], 'myeval_acc':[], 'lr':[], 'a':[], 'h':[], 'k':[]}
    epoch = 0
    falsenum = 0
    myloss = []
    a = []
    h = []
    k = []
    while(True):
        # train for one epoch
        start = time.time()
        for (x,y) in train_loader:
            y = paddle.reshape(y, (-1, 1))
            #计算损失函数？
            loss = loss_fn(model(x), y)

           # 进入反向传播
            loss.backward()
            opt.step()
            opt.clear_grad()
            losses.append(np.mean(loss.numpy()))
        print('Time per epoch:%.2f,loss:%.4f'%(time.time()-start,np.mean(losses)))

        # if (epoch+1)%save_every==0 or epoch+1==epochs:
        if epoch%save_every==0:
            # evaluate on validation set
            eval_acc,eval_loss = test()
            myloss.append(eval_loss)
            mydata['myeval_loss'].append(eval_loss)
            mydata['myeval_acc'].append(eval_acc)
            mydata['lr'].append(scheduler.get_lr())
            
            print("Validation accuracy/loss: %.2f%%,%.4f"%(eval_acc, eval_loss))
            model.train()
            if eval_acc > best_acc:
                paddle.save(model.state_dict(),os.path.join(save_dir, 'checkpoint.pdparams'))
                paddle.save(opt.state_dict(),os.path.join(save_dir, 'checkpoint.pdopt')) 
            best_acc = max(eval_acc, best_acc)
        scheduler.step()
        a.append(abs(myloss[epoch]-sum(myloss)/(epoch+1))*math.exp(-scheduler.get_lr()*(epoch+1)))
        if epoch==0:
            h.append(0)
        else:
            h.append(abs(myloss[epoch-1]-myloss[epoch]))
        k.append(a[epoch]>=h[epoch])
        # 
        mydata['a'].append(a[epoch])
        mydata['h'].append(h[epoch])
        mydata['k'].append(k[epoch])
        if k[epoch] == False:
            falsenum = epoch
        if epoch - falsenum > 1/2*int(eval_acc):
            break
        epoch = epoch + 1
    paddle.save(model.state_dict(),os.path.join(save_dir, 'model.pdparams'))
    paddle.save(opt.state_dict(),os.path.join(save_dir, 'model.pdopt'))
    print('Best accuracy on validation dataset: %.2f%%'%(best_acc))
    # print(myeval_loss)
    # plt.plot(myeval_loss)
    # df = pd.DataFrame(mydata)
    # df.to_excel('/home/aistudio/myeval_loss32.xlsx')
    return mydata

def test():
    model.eval()
    accuracies = []
    losses = []
    for (x,y) in val_loader:
        with paddle.no_grad():
            logits = model(x)
            y = paddle.reshape(y, (-1, 1))
            loss = loss_fn(logits, y)
            acc = acc_fn(logits, y)
            accuracies.append(np.mean(acc.numpy()))
            losses.append(np.mean(loss.numpy()))
    return np.mean(accuracies)*100, np.mean(losses) 
if __name__ == '__main__':
    CUDA = False
    if CUDA:
        paddle.set_device('gpu')
    place = paddle.CUDAPlace(0) if CUDA else paddle.CPUPlace()

    mean,std = ([0.4914, 0.4822, 0.4465],[0.2471, 0.2435,0.2616])
    mean = list(map(lambda x:x*255,mean))
    std = list(map(lambda x:x*255,std))

    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.Cifar10(mode='train', transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Transpose(order=(2,0,1)),
            transforms.Normalize(mean=mean,std=std),
        ]), download=True),
        places=place,batch_size=128, shuffle=True,
        num_workers=4, use_shared_memory=True)

    val_loader = paddle.io.DataLoader(
        paddle.vision.datasets.Cifar10(mode='test', transform=transforms.Compose([
            transforms.Transpose(order=(2,0,1)),
            transforms.Normalize(mean=mean,std=std),
        ])), places=place,
        batch_size=256, shuffle=False,
        num_workers=4, use_shared_memory=True)

    # ---------------------------------------------Dilation_myresnet20----------------------------------------------------
    model = myresnet20()



    epochs = 200 # 当train为固定迭代次数时，这里就是迭代次数。若不是固定迭代次数，则此参数无效
    save_every = 1
    #https://www.bookstack.cn/read/paddlepaddle-2.0-zh/61e889f1daed0459.md 解释paddle.nn.CrossEntropyLoss()
    #学习率设置：https://www.zhihu.com/question/36113643/answer/1868497020
    loss_fn = paddle.nn.CrossEntropyLoss()
    acc_fn = paddle.metric.accuracy
    scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[80,120],values=[0.1,0.01,0.001],verbose=True)
    opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)
    # opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler)
    save_dir = 'models/cifar10/myresnet20'
    time_start = time.time()
    mydata = training()
    time_end = time.time()
    print('Time cost = %fs' % (time_end - time_start))
    # 路径
    writer = pd.ExcelWriter('./models/data/C1output.xlsx')
    df1 = pd.DataFrame(mydata)
    df1.to_excel(writer,sheet_name='myresnet20')
    # writer.save()
    # ---------------------------------------------Dilation_myresnet20----------------------------------------------------
    
    # ---------------------------------------------Dilation_myresnet32----------------------------------------------------
    # model = myresnet32()
    # epochs = 200
    # save_every = 1
    # loss_fn = paddle.nn.CrossEntropyLoss()
    # acc_fn = paddle.metric.accuracy
    # scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[80,120],values=[0.1,0.01,0.001],verbose=True)
    # opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)
    # save_dir = 'models/cifar10/myresnet32'
    # mydata = training()
    # df2 = pd.DataFrame(mydata)
    # df2.to_excel(writer,sheet_name='myresnet32')
    # ---------------------------------------------Dilation_myresnet32----------------------------------------------------
    # ---------------------------------------------Dilation_myresnet44----------------------------------------------------
    # model = myresnet44()
    # epochs = 200
    # save_every = 1
    # loss_fn = paddle.nn.CrossEntropyLoss()
    # acc_fn = paddle.metric.accuracy
    # scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[80,120],values=[0.1,0.01,0.001],verbose=True)
    # opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)
    # save_dir = 'models/cifar10/myresnet44'
    # mydata = training()
    # df3 = pd.DataFrame(mydata)
    # df3.to_excel(writer,sheet_name='myresnet44')
    # ---------------------------------------------Dilation_myresnet44----------------------------------------------------
    # ---------------------------------------------Dilation_myresnet56----------------------------------------------------
    # model = myresnet56()
    # epochs = 200
    # save_every = 1
    # loss_fn = paddle.nn.CrossEntropyLoss()
    # acc_fn = paddle.metric.accuracy
    # scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[80,120],values=[0.1,0.01,0.001],verbose=True)
    # opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)
    # save_dir = 'models/cifar10/myresnet56'
    # mydata = training()
    # df3 = pd.DataFrame(mydata)
    # df3.to_excel(writer,sheet_name='myresnet56')
    # ---------------------------------------------Dilation_myresnet56----------------------------------------------------
    # ---------------------------------------------Dilation_myresnet110----------------------------------------------------
    # model = myresnet110()
    # epochs = 200
    # save_every = 1
    # loss_fn = paddle.nn.CrossEntropyLoss()
    # acc_fn = paddle.metric.accuracy
    # scheduler=paddle.optimizer.lr.PiecewiseDecay(boundaries=[80,120],values=[0.1,0.01,0.001],verbose=True)
    # opt = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)
    # save_dir = 'models/cifar10/myresnet110'
    # mydata = training()
    # df3 = pd.DataFrame(mydata)
    # df3.to_excel(writer,sheet_name='myresnet110')
    # ---------------------------------------------Dilation_myresnet110----------------------------------------------------
    writer.save()
    # ---------------------------------------------save---------------------------------------------------------------