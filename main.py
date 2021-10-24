from tensorflow.keras.utils import to_categorical
from SensodataLoader import  Loader
import numpy as np
from sklearn.model_selection import train_test_split
from MyVGG import MyVGG
import tensorflow as tf

def train_locv(datasets):
    """
    加速度センサデータのデータセットを用いて深層学習で学習する．
    モデルの評価にはLeave One Subject Outを用いることで個人差の影響を考慮する．
    学習後は評価におけて最大の精度を出したモデルを/my_modelに保存する．
    datasets : ndarrayの形になっているデータセット

    return : pred
    """
    accuracy = 0
    all_pred = np.empty(0)
    all_y_test = np.empty(0)
    print(len(datasets))
    for idx in range(len(datasets)):
        if idx != 4:
            continue
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
        model.fit(x_train, y_train, batch_size=128, epochs=30, verbose=1, validation_data=(x_val,y_val))

        pred = model.predict(x_test, verbose=0)
        pred = np.argmax(pred,axis=1)
        y_test = np.argmax(y_test, axis=1)
        score = np.count_nonzero(pred==y_test)/y_test.size
        print("Accuracy : {}".format(score))

        if score > accuracy:
            model.save("my_model")
            accuracy = score

        all_pred = np.append(all_pred, pred)
        all_y_test = np.append(all_y_test, y_test)

    
    return all_pred, all_y_test


def model_converter_tflite(model_path):
    """
    モデルをtensorflowLiteで使える形に変換する．
    model_path : tensorflowのModel()のsave()で出力されたモデルフォルダを指定
    """
    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, 256, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    open("tflite_model/converted_model.tflite", "wb").write(tflite_model)

def conf_mat(true, pred):
    from sklearn.metrics import confusion_matrix
    print("Confusion Matrix")
    print(confusion_matrix(pred, true))
    
def main():
    datasets = Loader().load_data()
    pred, grand_true = train_locv(datasets)
    conf_mat(grand_true.tolist(), pred.tolist())
    model_converter_tflite("my_model")

if __name__ == '__main__':
    main()