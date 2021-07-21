from tensorflow.keras.utils import to_categorical
from SensodataLoader import  Loader
import numpy as np
from sklearn.model_selection import train_test_split
from MyVGG import MyVGG

datasets = Loader().load_data()
# datasets = np.delete(datasets, 2, axis=1)
datasets = np.array(datasets)
# print(datasets[0])
def locv_train(datasets):
    for idx in range(len(datasets)):
        x_test, y_test = datasets[idx][0], datasets[idx][1]
        test_index = np.ones(len(datasets), dtype=bool)
        test_index[idx] = False  
        # print(datasets)
        other_dataset = datasets[test_index]
        x = [x[0] for x in other_dataset]
        y = [y[1] for y in other_dataset]
        x = np.concatenate(x).transpose(0,2,1)
        y = np.concatenate(y)
        # print(x.shape)
        x_train, x_val, y_train, y_val = train_test_split(x, y) 
        # print(y_train)
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        model = MyVGG().load_model()
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_val,y_val))
        score = model.evaluate(x_test, y_test, verbose=0)
        print(score)


locv_train(datasets)