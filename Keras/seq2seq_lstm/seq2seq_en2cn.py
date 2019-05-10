import numpy as np, os
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM
from .conf import *


def preprocess_data():
    """语料加载"""
    with open(corpus_path, encoding='utf-8') as f:
        seqs = f.read().lower().split('\n')

    """构建序列和字库"""
    seqs_input, seqs_output = [], []  # 输入、输出序列
    counter_input, counter_output = Counter(), Counter()  # 字库
    for seq in seqs:
        inputs, outputs = seq.split('\t')
        counter_input += Counter(list(inputs))
        counter_output += Counter(list(outputs))
        outputs = chr_start + outputs + chr_end  # 加入起终点
        seqs_input.append(inputs)
        seqs_output.append(outputs)

    # 过滤低频词
    counter_input = counter_input.most_common(num_classes_input - 1)
    counter_output = counter_output.most_common(num_classes_output - 3)

    # 加入字符（填充、起点、终点）到字库
    counter_input = [chr_pad] + [i[0] for i in counter_input]
    counter_output = [chr_pad, chr_start, chr_end] + [i[0] for i in counter_output]

    """字符和索引间的映射"""
    chr2id_input = {c: i for i, c in enumerate(counter_input)}
    chr2id_output = {c: i for i, c in enumerate(counter_output)}
    c2i_input = lambda c: chr2id_input.get(c, 0)
    c2i_output = lambda c: chr2id_output.get(c, 0)
    id2chr_output = {i: c for c, i in chr2id_output.items()}
    yield c2i_input, c2i_output, id2chr_output

    """输入层和输出层"""
    # 输入序列
    x_encoder = [[c2i_input(c) for c in chrs if c2i_input(c)] for chrs in seqs_input]
    # 起点 + 输出序列
    x_decoder = [[c2i_output(c) for c in chrs[:-1] if c2i_output(c)] for chrs in seqs_output]
    # 输出序列 + 终点
    y = [[c2i_output(c) for c in chrs[1:] if c2i_output(c)] for chrs in seqs_output]

    # 序列截断或补齐为等长
    x_encoder = pad_sequences(x_encoder, maxlen_input, padding='post', truncating='post')
    x_decoder = pad_sequences(x_decoder, maxlen_output, padding='post', truncating='post')
    y = pad_sequences(y, maxlen_output, padding='post', truncating='post')

    # 独热码
    x_encoder = to_categorical(x_encoder, num_classes=num_classes_input)
    x_decoder = to_categorical(x_decoder, num_classes=num_classes_output)
    y = to_categorical(y, num_classes=num_classes_output)
    print('输入维度', x_encoder.shape, x_decoder.shape, '输出维度', y.shape)
    yield x_encoder, x_decoder, y


[(c2i_input, c2i_output, id2chr_output),
 (x_encoder, x_decoder, y)] = list(preprocess_data())


if os.listdir(prefix_hdf5):
    """加载已训练模型"""
    model = load_model(path_hdf5)
    model_encoder = load_model(path_hdf5_encoder)
    model_decoder = load_model(path_hdf5_decoder)
else:
    """编码模型"""
    encoder_input = Input(shape=(None, num_classes_input))  # 编码器输入层
    encoder_lstm = LSTM(units, return_state=True)  # 编码器LSTM层
    _, encoder_h, encoder_c = encoder_lstm(encoder_input)  # 编码器LSTM输出
    model_encoder = Model(encoder_input, [encoder_h, encoder_c])  # 【编码模型】

    # 解码器
    decoder_input = Input(shape=(None, num_classes_output))  # 解码器输入层
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)  # 解码器LSTM层
    decoder_output, _, _ = decoder_lstm(
        decoder_input, initial_state=[encoder_h, encoder_c])  # 解码器LSTM输出
    decoder_dense = Dense(num_classes_output, activation='softmax')  # 解码器softmax层
    decoder_output = decoder_dense(decoder_output)  # 解码器输出

    """训练模型"""
    model = Model([encoder_input, decoder_input], decoder_output)  # 【训练模型】
    model.compile('adam', 'categorical_crossentropy')
    model.fit([x_encoder, x_decoder], y, batchsize, epochs, verbose=2)

    """解码模型"""
    decoder_h_input = Input(shape=(units,))  # 解码器状态输入层h
    decoder_c_input = Input(shape=(units,))  # 解码器状态输入层c
    decoder_output, decoder_h, decoder_c = decoder_lstm(
        decoder_input, initial_state=[decoder_h_input, decoder_c_input])  # 解码器LSTM输出
    decoder_output = decoder_dense(decoder_output)  # 解码器输出
    model_decoder = Model([decoder_input, decoder_h_input, decoder_c_input],
                          [decoder_output, decoder_h, decoder_c])  # 【解码模型】

    # 模型保存
    plot_model(model, path_png, show_shapes=True, show_layer_names=False)
    plot_model(model_encoder, path_png_encoder, show_shapes=True, show_layer_names=False)
    plot_model(model_decoder, path_png_decoder, show_shapes=True, show_layer_names=False)
    model.save(path_hdf5)
    model_encoder.save(path_hdf5_encoder)
    model_decoder.save(path_hdf5_decoder)


def seq2seq(x_encoder_pred):
    """序列生成序列"""
    h, c = model_encoder.predict(x_encoder_pred)
    id_pred = id_start
    seq = ''
    for _ in range(maxlen_output):
        y_pred = to_categorical([[[id_pred]]], num_classes=num_classes_output)
        output, h, c = model_decoder.predict([y_pred, h, c])
        id_pred = np.argmax(output[0])
        seq += id2chr_output[id_pred]
        if id_pred == id_end:
            break
    return seq[:-1]


if __name__ == '__main__':
    while True:
        chrs = input('输入：').strip().lower()
        x_encoder_pred = [[c2i_input(c) for c in chrs]]
        x_encoder_pred = pad_sequences(x_encoder_pred, maxlen_input, padding='post', truncating='post')
        x_encoder_pred = to_categorical(x_encoder_pred, num_classes_input)
        seq = seq2seq(x_encoder_pred)
        print('输出：%s\n' % seq)
