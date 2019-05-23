from bilsm_crf_model import create_model

n = 30000
batch_size = 256
epochs = 1

model, (train_x, train_y) = create_model()
model.fit(train_x[:n], train_y[:n], batch_size, epochs)
model.save('model/crf.h5')
