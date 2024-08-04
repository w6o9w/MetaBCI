import numpy as np

def SlideWindow(data,mark,WindowLength,slide): ###window:时间窗长  slide:滑动步长
    trials=data.shape[0]            ###trials:试次数   time:数据长度   channel：电极数目
    time=data.shape[1]
    channel=data.shape[2]
    EachtrialNum=int((time-WindowLength)/slide+1)       ###每trial时间窗数量
    trial_points=EachtrialNum*WindowLength                         ###每trial增强后总窗长
    CombineLength = np.zeros((int(trial_points*trials), channel))   ###拼接后总窗长
    CropData=np.zeros((EachtrialNum*trials,WindowLength,channel))    ###数据增强后数据形状
    CropMark=[]
    for i in range(trials):
        for k in range(EachtrialNum):
            CombineLength[trial_points * i + WindowLength * k:trial_points * i + WindowLength * (k + 1), :] = data[i,k * slide:WindowLength + k * slide,:]
    for i in range(EachtrialNum * trials):
        CropData[i] = CombineLength[WindowLength * i:WindowLength * (i + 1)]
    for i in mark:
        for j in range(EachtrialNum):
            CropMark.append(i)
    return CropData, CropMark

