from os.path import exists
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF  # https://github.com/keras-team/keras-contrib
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from process_data import load_data, chunk_tags, load_vocab  # 自写模块导入

batch_size = 512
epochs = 20
verbose = 1
filepath = 'model/crf.h5'
min_delta = 1e-9
patience = 2
callbacks = [EarlyStopping('val_crf_viterbi_accuracy', min_delta, patience)]
validation_split = .1

EMBED_DIM = 200
BiRNN_UNITS = 200


def modeling(train=True):
    model = Sequential()
    model.add(Embedding(len(load_vocab()) + 1, EMBED_DIM, mask_zero=True))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    plot_model(model, to_file='model/model.png', show_shapes=True)
    model.compile(Adam(), loss=crf.loss_function, metrics=[crf.accuracy])
    if exists(filepath):
        model.load_weights(filepath)
    if train:
        x, y = load_data()
        model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split)
        model.save_weights(filepath)
    return model


if __name__ == '__main__':
    modeling()
