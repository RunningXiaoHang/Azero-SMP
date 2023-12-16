import numpy as np
import pandas as pd
import joblib

#print("Load Dataset...")
csv_data_df = pd.read_csv("./Spider/all_data_v1.0.csv", encoding='gbk')
csv_data_df = csv_data_df.dropna()
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

#X_data = csv_data_df[['亚盘水位','亚盘','赔率水位','胜','平','负']]#.astype(float)
#X_data = np.array(X_data).reshape(-1,6)
#print(X_data[0:len(X_data)-1][1])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

en1 = LabelEncoder()
tem = np.array(csv_data_df["亚盘"])
tem = tem.reshape(-1, 1)
csv_data_df["亚盘"] = en1.fit_transform(tem)
#joblib.dump(en1, "./models/LabelEncoder.pkl")

X_data = csv_data_df[['亚盘水位','亚盘','赔率水位','胜','平','负']].astype(float)
X_data = np.array(X_data).reshape(-1,6)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)
print("train acc: {}".format(gbc_clf.score(X_train, y_train)), end=' ')
print("test acc: {}".format(gbc_clf.score(X_test, y_test)))
joblib.dump(gbc_clf, './models/gbc_clf.pkl')