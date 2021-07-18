import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm

class Loader:
    def __init__(self, sensordata_dir="SensorData", sensordata_labelFile="./SensorData/DataLabel.csv"):
        self.sensordata_dir=sensordata_dir
        self.sensordata_labelFile = sensordata_labelFile

    def __call__(self, sensordata_type="accelerate"):
        self.sensordata_type = sensordata_type
        if self.sensordata_type == "accelerate":
            print("accelerate")
        elif self.sensordata_type == "gyro":
            print("gyro")
        else:
            print("type is nothing")
    
    def read_data(self, sensordata_type="accelerate"):
        labels_data = pd.read_csv(self.sensordata_labelFile)
        labels_data["SensingDataFileName"] = labels_data["SensingDataFileName"].apply(lambda x: "{}_{}".format(sensordata_type,x))
        
        #フォルダ以下のファイル名を読み込みマージ用にDataFrameにする
        sensordata_list = [filename.name[0:-4] for filename in pathlib.Path(self.sensordata_dir).glob("*.txt") if sensordata_type in filename.name]
        sensordata_list = pd.DataFrame({"SensingDataFileName":sensordata_list})
        
        #dataframeでlabel_dataに存在するラベルに合わせてデータファイル情報を残す
        merged_df = pd.merge(labels_data, sensordata_list)

        #todo:file 読み込みしてndarrayをかえす
        #data = pd.read_csv(self.sensordata_dir+"/"+merged_df.SensingDataFileName[0]+".txt")
        dataset = []
        for idx, item in merged_df.iterrows():
            data = pd.read_csv(self.sensordata_dir+"/"+item.SensingDataFileName+".txt")
            data = data.drop(columns="timestamp").values
            dataset.append((data, item.Label, item.id))

        return dataset

    #データセットの成型（データの分割など）
    def shape_dataset(self, dataset, window_size=256, label_dict = {"lie":0,"sit":1,"stand":2,"walk_treadmill":3,"walk_disturb":4}):
        """
        dataset : taple(sensordata : ndarrayでx,y,z軸で格納されている,
                        label : strでラベルがふられている，
                        id : strでセンサデータの被験者id )のリスト
        window_size : 生データをウィンドウサイズ何で分割するかの値
        label_dict : datasetのlabelに対応した辞書
        """
        print("shape dataset ...")
        shaped_dataset = []
        for elm in tqdm(dataset):
            sensor_data = elm[0]
            label = elm[1]
            id = elm[2]
            batch_size = int(len(sensor_data)/window_size)
            split_data = [sensor_data[i:i+window_size] for i in range(0, batch_size*window_size, window_size)]
            split_data = np.array(split_data).transpose(0,2,1)
            print(split_data)
            label = np.full(batch_size, label_dict[label])
            shaped_dataset.append((split_data, label, id))
        return shaped_dataset

    #成形したデータセットロードする．
    def load_data(self):
        raw_data = self.read_data()
        return self.shape_dataset(raw_data)

if __name__ == '__main__':
    loader = Loader()
    a = loader.read_data()
    #print(a)
    b = loader.shape_dataset(a)
    #print(b)

