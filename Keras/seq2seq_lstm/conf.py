"""配置"""

# 数据预处理参数
corpus_path = 'en2cn.txt'
num_classes_input = 32 + 1  # 英文低频字过滤
num_classes_output = 900 + 3  # 中文低频字过滤
chr_pad = ''  # 填充字符
chr_start = '['  # 起始字符
chr_end = ']'  # 结束字符
id_pad = 0  # 填充字ID
id_start = 1  # 起点ID
id_end = 2  # 终点ID
maxlen_input = 24  # 输入序列最大长度
maxlen_output = 16  # 输出序列最大长度

# 模型参数
units = 300  # LSTM神经元数量
batchsize = 512
epochs = 500

# 网络权重存储路径
prefix_hdf5 = 'model/'
path_hdf5 = prefix_hdf5 + 'model.hdf5'
path_hdf5_encoder = prefix_hdf5 + 'encoder.hdf5'
path_hdf5_decoder = prefix_hdf5 + 'decoder.hdf5'

# 模型可视化存储路径
prefix_png = 'model_png/'
path_png = prefix_png + 'model.png'
path_png_encoder = prefix_png + 'encoder.png'
path_png_decoder = prefix_png + 'decoder.png'
