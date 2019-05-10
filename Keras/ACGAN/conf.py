"""配置"""

# 生成图像路径
dir_imgs = 'images/'
path_imgs = dir_imgs + '%02d.png'

# 模型可视化路径
dir_model_imgs = 'model_images/'
path_cnnt_imgs = dir_model_imgs + 'cnnt.png'
path_generator_imgs = dir_model_imgs + 'generator.png'
path_cnn_imgs = dir_model_imgs + 'cnn.png'
path_judge_imgs = dir_model_imgs + 'judge.png'
path_tricker_imgs = dir_model_imgs + 'tricker.png'

# 网络参数
shape = (28, 28, 1)  # 图像维度
num_classes = 10  # 手写数字10分类
noise_dim = 100  # 【噪音】输入层维度

# adam优化器参数
lr = 2e-4
beta_1 = .5

# 训练参数
n_samples = 40960  # 样本总量
batch_size = 512  # 批量
times = int(n_samples / batch_size)  # 训练次数
epochs = 40  # 训练轮数
