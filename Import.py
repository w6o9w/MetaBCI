import scipy.io as sio
def Import(type,subnum,folder):   ###type:实验类型(1、2、3、4、5)   subnum：被试序号  floder：处理方法
    if subnum>=10:
        Datapath='E:\\BCI数据\\yhdata\\'+str(folder)+'\Sub0'+str(subnum)+'\\'+str(type)+'\\data.mat'
        Markpath='E:\\BCI数据\\yhdata\\'+str(folder)+'\\Sub0'+str(subnum)+'\\'+str(type)+'\\mark.mat'
    else:
        Datapath = 'E:\\BCI数据\\yhdata\\' + str(folder) + '\Sub00' + str(subnum) + '\\' + str(type) + '\\data.mat'
        Markpath = 'E:\\BCI数据\\yhdata\\' + str(folder) + '\\Sub00' + str(subnum) + '\\' + str(type) + '\\mark.mat'
    Data=sio.loadmat(Datapath)['data']
    Mark=sio.loadmat(Markpath)['label']
    return Data,Mark

