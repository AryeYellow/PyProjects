"""配置"""

corpus_path = 'poem5.txt'
len_chr = 1000  # 字库大小
window = 24  # 滑窗大小
filters = 25  # 卷积录波器数量
kernel_size = 5  # 卷积核大小
times = 40  # 训练总次数
batch_size = 512
epochs = 25
filepath = 'model.hdf5'
