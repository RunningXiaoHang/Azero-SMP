#**足彩胜负预测 基于Tensorflow机器学习**

---

##Python环境:

> Python >= 3.6.6

> Tensorflow >= 2.11.0

> numpy >= 1.21.0

> pandas >= 0.23.4

> sklearn >= 0.19.2

推荐使用Anaconda

---

##数据集:

Kaggle数据集: all_euro_data.csv

---

##train.py

###导入所需要的Python包

```python
import numpy
import pandas
```

###加载加载数据集

```python
print("Load Dataset...")
dataset = pd.read_csv('./all_euro_data_.csv', low_memory=False)
dataset = pd.DataFrame.dropna(dataset, how='all', axis=0)
```

###获取标签

```python
ftr = dataset['FTAG']
ftr = np.array(ftr)
```

###用均值填充空缺值

```python
dataset["B365H"] = dataset["B365H"].fillna(dataset["B365H"].mean())
dataset["B365D"] = dataset["B365D"].fillna(dataset["B365D"].mean())
dataset["B365A"] = dataset["B365A"].fillna(dataset["B365A"].mean())
dataset["WHH"] = dataset["WHH"].fillna(dataset["WHH"].mean())
dataset["WHD"] = dataset["WHD"].fillna(dataset["WHD"].mean())
dataset["WHA"] = dataset["WHA"].fillna(dataset["WHA"].mean())
dataset["VCH"] = dataset["VCH"].fillna(dataset["VCH"].mean())
dataset["VCD"] = dataset["VCD"].fillna(dataset["VCD"].mean())
dataset["VCA"] = dataset["VCA"].fillna(dataset["VCA"].mean())
```

这里的B365,WH,VC分别对应bet365,威廉希尔,韦德公司的赔率

H, D, A分别对应胜平负的赔率

###加载到X_data

```python
X_data = dataset[["WHH", "WHD", "WHA", "VCH", "VCD", "VCA", "B365H", "B365D", "B365A"]].values
X_data = np.array(X_data)
```

###将ftr里的标签转换为数值

```python
y_data = []
y_data = np.array(y_data)
for i in range(len(X_data)):
    if ftr[i] == 'H':
        y_data = np.append(y_data, 0)
    elif ftr[i] == 'D':
        y_data = np.append(y_data, 1)
    elif ftr[i] == 'A':
        y_data = np.append(y_data, 2)
```

这里ftr的标签为H, D, A转换为数值0, 1, 2

###分离训练集和测试集

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)
```

###导入keras库

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```

###定义EarlyStopping的参数

```python
monitor = EarlyStopping(monitor='val_loss',
                        min_delta=1e-3,
                        patience=5,
                        mode='auto',
                        verbose=1,
                        restore_best_weights=True)
```

以val_loss为标准, 当变化大于0.001才算一次改进, 当五次没有改进则停止训练,并传入参数

restore_best_weights=True, 使模型自动回滚到最佳状态

###构建模型

```python
model = Sequential()
model.add(Flatten(input_shape=[9,1]))
model.add(Dense(9, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
```

###编译模型

```python
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])
```

###训练模型

```python
model.fit(X_train,
          y_train,
          epochs=100,
          batch_size=32,
          callbacks=[monitor],
          validation_data=(X_test, y_test))
```

###保存模型到当前目录

```python
model.save('./model.h5')
```

---

##Run.py

###忽略Tensorflow的日志

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

###加载模型

```python
import numpy as np
print('Load Model...')
from keras.models import load_model
class_names = ['Win', 'Flat', 'Failure']
model = load_model("./model.h5")
print('Ok!')
```

###输入赔率数值

```python
home_team = input("HomeTeam: ")
away_team = input("AwayTeam: ")
WHH = float(input("WHH: "))
WHD = float(input("WHD: "))
WHA = float(input("WHA: "))
VCH = float(input("VCH: "))
VCD = float(input("VCD: "))
VCA = float(input("VCA: "))
B365H = float(input("B365H: "))
B365D = float(input("B365D: "))
B365A = float(input("B365A: "))
```

###整合成一个数组

```python
X_new = [WHH, WHD, WHA, VCH, VCD, VCA, B365H, B365D, B365A]
X_new = np.array(X_new)
X_new = X_new.reshape(1,-1)
```

Tensorflow要求传入参数为二维数组, 即reshape(1, -1)

###预测

```python
proba = model.predict(X_new)
proba = np.array(proba)
result = np.argmax(proba)
```

###输出结果

```python
print("=====================================")
print("Predict:")
print("HomeTeam: {} vs AwayTeam: {}".format(home_team, away_team))
print("WinRate: {}%".format(proba[0,0]*100))
print("FlatRate: {}%".format(proba[0,1]*100))
print("FailureRate: {}%".format(proba[0,2]*100))
print("=====================================")
print("HomeTeam: {} vs AwayTeam: {}".format(home_team, away_team))
print("WHOdds: {} {} {}".format(WHH, WHA, WHD))
print("VCOdds: {} {} {}".format(VCH, VCD, VCA))
print("Bet365Odds: {} {} {}".format(B365H, B365D, B365A))
print("FinallyResult: {} ".format(class_names[result]))
```

---
