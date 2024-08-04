import numpy as np
def Shuffle(data,mark):
    index=[i for i in range(data.shape[0])]
    np.random.shuffle(index)
    ShuffleData=data[index]
    ShuffleMark=mark[index]
    return ShuffleData,ShuffleMark
