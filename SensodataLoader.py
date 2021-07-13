import pandas as pd
import numpy as np
import pathlib

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

        return merged_df




if __name__ == '__main__':
    loader = Loader()
    print(loader.read_data())

