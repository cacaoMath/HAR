from tensorflow.keras.utils import to_categorical
from SensodataLoader import  Loader
import numpy as np
from sklearn.model_selection import train_test_split
from MyVGG import MyVGG
import tensorflow as tf

def train_locv(datasets):
    for idx in range(len(datasets)):
        x_test, y_test = datasets[idx][0].transpose(0,2,1), to_categorical(datasets[idx][1], num_classes=5)
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
        y_train = to_categorical(y_train, num_classes=5)
        y_val = to_categorical(y_val, num_classes=5)

        model = MyVGG().load_model()
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_val,y_val))
        model.save("my_model")
        score = model.predict(x_test, verbose=0)
        print(score)
        if idx == 0:
            break

def model_converter_tflite(model_path):
    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, 256, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    open("tflite_model/converted_model.tflite", "wb").write(tflite_model)

def main():
    datasets = Loader().load_data()
    train_locv(datasets)
    # model_converter_tflite("my_model")

if __name__ == '__main__':
    main()