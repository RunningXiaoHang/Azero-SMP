import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

print("Load Dataset...")
csv_data_df = pd.read_csv("./Spider/all_data_v1.0.csv", encoding='gbk')
csv_data_df = dataset = pd.DataFrame.dropna(csv_data_df, how='all', axis=0)
hg = csv_data_df['主队进球'].values
ag = csv_data_df['客队进球'].values
hg, ag = np.uint(np.array(hg)), np.uint(np.array(ag))
y_data = []
y_data = np.array(y_data)
for i in range(len(hg)):
    if hg[i] > ag[i]:
        y_data = np.append(y_data, 0)
    elif hg[i] == ag[i]:
        y_data = np.append(y_data, 1)
    else:
        y_data = np.append(y_data, 2)
csv_data_df['亚盘'] = csv_data_df['亚盘'].astype('category').cat.codes
X_data = csv_data_df[['亚盘水位','亚盘','赔率水位','胜','平','负']].values
X_data = np.array(X_data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

from keras import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_dim=6))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))


epochs = 20
model.compile(optimizer='RMSprop',
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test))
model.evaluate(X_test, y_test)

#model.save('./model.h5')



