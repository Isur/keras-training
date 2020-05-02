from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pathlib

ABS_PATH = pathlib.Path(__file__).parent.absolute()
DATASET_PATH = f"{ABS_PATH}/dataset/pima-indians-diabetes.csv"


def Keras():
    dataset = loadtxt(DATASET_PATH, delimiter=',')
    x = dataset[:, 0:8]
    y = dataset[:, 8]
    model = Sequential()
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=250, batch_size=15)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    predictions = model.predict_classes(x)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    for i in range(15):
        print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
