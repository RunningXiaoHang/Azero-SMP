import pandas as pd
import numpy as np
print("Load Dataset...")
dataset = pd.read_csv('./all_euro_data_.csv', low_memory=False)
dataset = pd.DataFrame.dropna(dataset, how='all', axis=0)
ftr = dataset['FTAG']
ftr = np.array(ftr)

dataset["B365H"] = dataset["B365H"].fillna(dataset["B365H"].mean())
dataset["B365D"] = dataset["B365D"].fillna(dataset["B365D"].mean())
dataset["B365A"] = dataset["B365A"].fillna(dataset["B365A"].mean())
dataset["WHH"] = dataset["WHH"].fillna(dataset["WHH"].mean())
dataset["WHD"] = dataset["WHD"].fillna(dataset["WHD"].mean())
dataset["WHA"] = dataset["WHA"].fillna(dataset["WHA"].mean())
dataset["VCH"] = dataset["VCH"].fillna(dataset["VCH"].mean())
dataset["VCD"] = dataset["VCD"].fillna(dataset["VCD"].mean())
dataset["VCA"] = dataset["VCA"].fillna(dataset["VCA"].mean())
X_data = dataset[["WHH", "WHD", "WHA", "VCH", "VCD", "VCA", "B365H", "B365D", "B365A"]].values
X_data = np.array(X_data)

y_data = []
y_data = np.array(y_data)
for i in range(len(X_data)):
    if ftr[i] == 'H':
        y_data = np.append(y_data, 0)
    elif ftr[i] == 'D':
        y_data = np.append(y_data, 1)
    elif ftr[i] == 'A':
        y_data = np.append(y_data, 2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

monitor = EarlyStopping(monitor='val_loss',
                        min_delta=1e-3,
                        patience=5,
                        mode='auto',
                        verbose=1,
                        restore_best_weights=True)

model = Sequential()
model.add(Flatten(input_shape=[9,1]))
model.add(Dense(9, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])

model.fit(X_train,
          y_train,
          epochs=100,
          batch_size=32,
          callbacks=[monitor],
          validation_data=(X_test, y_test))

model.save('./model.h5')