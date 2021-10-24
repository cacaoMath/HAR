# Human Activity Recognition(HAR)
加速度センサデータから行動認識(HAR)を深層学習で行うためのプログラム(加速度センサ以外にジャイロセンサデータでも使用可)  
本プログラムでの想定する行動認識は(寝転ぶ，座る，立つ，歩行（2種類）)の5種となっている．

# 構成
- ./main.py : データセットから学習を行いモデルの出力,及びモデルのtfLiteへの変換を行う．  
- ./MyVGG.py : 深層学習モデルvgg16を今回の行動認識用に変更したモデルを呼び出すためのクラス
- ./SensordataLoader.py : スマートフォン加速度センサデータを用いて，データを成形しデータセットにする処理を行う．
- ./my_model/ : main.pyで学習されたモデルが出力される．
- ./SensorData/ : 学習に使うスマホ等で計測された加速度センサデータを格納する．データは複数人で計測されている想定で複数存在するとする．また，各センサデータのメタデータを含んだDataLabel.csvを含めることで，SensordataLoader.pyによってメタデータに合わせたデータセットが生成できる．
- ./tflite_model/ : main.pyで学習したモデルを.tflite_model形式にしたものを格納する．

# 想定するセンサデータ等の形式
## /SensorDataに格納するセンサデータ 
- ファイルの拡張子  
.txt 
- ファイル名  
accelerate_{timestamp}_{label}  
または  
gyro_{timestamp}_{label}  
例)accelerate_1617237735285_walk.txt

- ファイル内容例(csv形式)  
timestamp : データ取得時のタイムスタンプ  
x : 加速度センサデータx軸  
y : 加速度センサデータy軸  
z : 加速度センサデータz軸  

```
timestamp,x,y,z
1617930819643,0.443042,0.572680,-0.939078
1617930819644,0.443042,0.572680,-0.939078
1617930819644,0.003398,0.001339,0.000716
1617930819645,0.003398,0.001339,0.000716
1617930819646,0.002885,0.003970,-0.001788
1617930819646,0.002885,0.003970,-0.001788
1617930819672,-0.004358,0.004936,0.002693
1617930819673,-0.004358,0.004936,0.002693
1617930819673,0.004732,0.003792,-0.004903
1617930819674,0.004732,0.003792,-0.004903
1617930819674,-0.002113,-0.003786,-0.003217
1617930819675,-0.002113,-0.003786,-0.003217
1617930819710,0.000542,0.003076,-0.001642
1617930819711,0.000542,0.003076,-0.001642
1617930819712,-0.002614,-0.002667,0.000004
1617930819713,-0.002614,-0.002667,0.000004
1617930819713,-0.002841,0.004680,-0.004220
1617930819714,-0.002841,0.004680,-0.004220
1617930819730,0.000981,0.000744,0.004648
1617930819730,0.000981,0.000744,0.004648
...
```

## DataLabel.csv
- ファイル名  
DataLabel.csv
- ファイル内容例
SensingDataFileName : 加速度センサデータを記録したファイル名(拡張子なし)  
Label : 各加速度センサデータファイルに対する，行動のラベル  
id : 加速度センサデータの個人識別のための任意のid
```
SensingDataFileName,Label,id
1618554784868_walk_treadmill,walkwalk_treadmill,subA
1618554553296_stand,stand,subA
1618554322153_sit,sit,subB
1618274964800_stand,stand,subC
1618274777264_sit,sit,sunA
1618274509796_walk_disturb,walk_disturb,subC
...
```

# その他注意
- 本プログラムで用いる加速度センサデータは1ファイルのデータ量の想定として5分以上の計測が前提となっている．そのため，各ファイルにおいて最低でも256サンプルはある前提となっている．

# 使用ライブラリ等
Python3.7.10, Numpy1.19.5, Tensorflow2.4, Pandas1.1.3, tqdm4.62.2




