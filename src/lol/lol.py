from .dataset import Dataset, r_dictionary
from keras.models import Sequential, load_model
from keras.layers import Dense
from pathlib import Path


ABS_PATH = Path(__file__).parent.absolute()
MODEL_PATH = f"{ABS_PATH}/dataset/model.h5"


class Lol(object):
    def __init__(self):
        self.dataset = None

    def load_dataset(self):
        if self.dataset is not None:
            print("dataset already loaded!")
            return
        db = Dataset()
        db.get_data()
        print("loaded dataset:")
        db.show_dataset()
        self.dataset = db

    def victory_model(self):
        m = Path(MODEL_PATH)
        self.dataset.prepare_data()
        x, y = self.dataset.get_training_data()
        if m.is_file():
            print("MODEL LOADING")
            model = load_model(MODEL_PATH)
        else:
            print("Model training")
            model = Sequential()
            model.add(Dense(10, input_dim=10, activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x, y, epochs=100, batch_size=10)
            model.save(MODEL_PATH)
        loss, accuracy = model.evaluate(x, y)
        predictions = model.predict_classes(x)
        print(f"Accuracy: {accuracy}")
        print(f"Loss: {loss}")
        correct = 0
        predictions_num = len(predictions)
        self.dataset.human_data()
        x, y = self.dataset.get_training_data() 
        for i in range(predictions_num):
            # print('%s => %d (expected %d)' % (x[i:i+1], predictions[i], y[i:i + 1]))
            if predictions[i] == y[i:i + 1].values[0]:
                correct += 1
        print(f"Correct: {correct} / {predictions_num}")


def main():
    lol = Lol()
    lol.load_dataset()
    lol.victory_model()
