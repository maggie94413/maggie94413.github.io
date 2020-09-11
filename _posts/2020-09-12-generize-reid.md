>torch.cuda.set_device(gpu_ids[0])#设置当前设备
>if opt.color_jitter:#随机更改图片的亮度，对比度，饱和度
transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

>transforms.Resize((256,128), interpolation=3),
transforms.Pad(10),#给定颜色填充图片
transforms.RandomCrop((256,128)),#在随机位置裁剪给定的PIL图像
transforms.RandomHorizontalFlip(),#以给定概率水平翻转该图片
transforms.ToTensor(),#将图片转换为张量
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#用均值和标准差归一化张量图像
torch.cuda.is_available()#返回bool值，指示当前cuda是否可用
torch.no_grad():#禁用梯度计算
loss = criterion(outputs, labels)#计算loss
optimizer_ft = optim.SGD([#随机梯度下降
{'params': base_params, 'lr': 0.1*opt.lr},
{'params': model.classifier.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()#logSoftMAX和NLLloss整合到一起

# Save to Matlab for check：类似excel格式的文件（mat
>result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

>score = np.dot(gf,query)#矩阵乘法
good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)#在query中而不在后者的已经排序的唯一值
cmc[rows_good[0]:] = 1#精妙：第一个good之后的所有cmc都设置为1
