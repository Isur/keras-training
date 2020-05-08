from numpy import loadtxt
from keras.models import Sequential, load_model
from keras.layers import Dense
import pathlib

ABS_PATH = pathlib.Path(__file__).parent.absolute()
DATASET_PATH = f"{ABS_PATH}/dataset/pima-indians-diabetes.csv"
SAVE_PATH = f"{ABS_PATH}/saved"


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
    model.fit(x, y, epochs=100, batch_size=15)
    loss, accuracy = model.evaluate(x, y, verbose=0)
    predictions = model.predict_classes(x)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    for i in range(15):
        print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
    # saving to file
    # model with architecture, weights, config, state
    model.save(SAVE_PATH + "/savedmodel.h5")
    # saving to json
    model_json = model.to_json()
    with open(SAVE_PATH + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # loading model
    loaded = load_model(SAVE_PATH + "/savedmodel.h5")
    new_dataset = loadtxt(DATASET_PATH, delimiter=",")
    i = dataset[:, 0:8]
    j = dataset[:, 8]
    score = loaded.evaluate(i, j, verbose=0)
    print(f"Accuracy: {score[1]}")
    predicts = loaded.predict_classes(i)
    for k in range(15):
        print('%s => %d (expected %d)' % (i[k].tolist(), predicts[k], j[k]))
