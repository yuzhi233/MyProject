from matplotlib import pyplot as plt
from IPython import display
import torch


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize



def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend =None,figsize=(3.5,2.5)):
    ''' # 画y轴为对数坐标的对数图 主要是来画训练集和测试集epoch-loss 曲线
        x_vals:训练集x轴坐标,y_vals:训练集y轴坐标,x_label:训练集x轴标签文本,y_label:训练集y轴标签文本,
        x2_vals=None:如果画测试集 测试集的x轴坐标,y2_vals=None:如果画测试集 测试集的y轴坐标,
        legend =None :给图像加上图例,figsize=(3.5,2.5):默认窗口大小'''
    set_figsize(figsize)#设置图像尺寸
    plt.xlabel(x_label)#这里的label是
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:#意思都不同时为0  当需要画测试集 epoch与loss关系时候 （传入的有这俩参数的 ）
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)#只花一个图的时候没必要打开legend


def evaluate_test_accuracy(data_iter, net,device=None):#用于评估测试集准确率 目的是要实现评估的时候要自动关闭dropout
'''用测试集来评估模型的准确率(分类问题)
    data_iter 数据集迭代器, net:要评估的模型,device=None :传入的设备 '''


    if device is None and isinstance(net,torch.nn.Module):#如果设备是none且net是由nn.module生成的实例。则：
          # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0

    with torch.no_grad():#不追踪操作 因为这个是测试集评估的 默认模型训练好了也就不需要记录梯度
        for X, y in data_iter:#从data_iter取出一个batch的X,y
          #先判断你这个net是怎么产生的是你自己手写的还是利用pytorch快速生成的
            if isinstance(net, torch.nn.Module):#判断net是不是用torch.nn.Module创建的实例(判断net是不是利用module模块搭建的)
                net.eval() # #如果是上面方法创建的 那么开启评估模式 dropout层全部关闭(因为我们要是通过module模块创建一个模型有可能添加了dropout层)
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()#判断正确的个数
                net.train() # 改回训练模式
            else: # 如果是我们自定义的模型    else下面的这段主要是用于3.13节我们自定义带dropout的模型，计算准确率的以后不会用到 不考虑GPU
                print('这个模型没有继承Module')
            n += y.shape[0]#其实就是算了以下一个批次有多少样本 每次循环累加一下参加计算的样本数
    return acc_sum / n#在所有批次循环后  计算准确率 拿 准确的个数/总个数


def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)#将模型添加到设备上 可能是GPU可能是CPU
    print("training on ", device)#显示是在哪训练的
    loss = torch.nn.CrossEntropyLoss()#使用交叉熵损失函数

    train_ls=[]
    test_ls=[]
    for epoch in range(num_epochs):#几个迭代周期 开始迭代
        #定义 训练集损失和,训练集准确总数，总样本数n,几个batch,开始时间
        train_l_batch_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:#一个批次中(比如一个batch_size是256)
            X = X.to(device)#先将需要计算的添加到 设备
            y = y.to(device)#同上
            y_hat = net(X)#计算模型预测值y_hat---->
            l = loss(y_hat, y.long())#计算损失（利用前面定义的交叉熵损失函数）
            optimizer.zero_grad()#优化器梯度清0
            l.backward()#误差反向传播
            optimizer.step()#梯度更新
            train_l_batch_sum += l.cpu().item()# train_l_sum计算的是一个batch上的总误差，最后累加的train_l_sum需要除以总的batch数     计算得到的误差可能再GPU上先移动到CPU转成pyton数字
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()#一个batch上训练集正确个数  一个batch一个batch不断累加得到最后的总个数 最后要除以训练集样本总数n
            n += y.shape[0]
            batch_count += 1


        #一个epoch后 整个训练集/测试集的 loss   ----------画图用 不画图要注释掉
        with torch.no_grad():

            # train_ls.append(loss(net(train_features.cuda()), train_labels.cuda().long()).cpu().item())
            train_ls.append(train_l_batch_sum/n).cpu().item())
            test_ls.append( loss(net(test_features.cuda()), test_labels.cuda().long()).cpu().item() )#计算整个测试集上的误差

        #一个epoch后 对测试集准确率进行模型评估
        test_acc = evaluate_accuracy_2(test_iter, net)
        #打印 一个epoch上 测试集的平均loss  测试集的准确率  训练集的准确率 计算所用时间
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_batch_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    # print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])#与train_l_batch_sum / batch_count等价

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',range(1, num_epochs + 1), test_ls, ['train', 'test'])#--------画图用 不画图可以注释








