import random
import time
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np

from Spider.crawler import get_data_to_pred

def print_color(text, color):
    colors = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }
    if color not in colors:
        print(text)  # 如果颜色不支持，以普通方式输出
    else:
        color_code = colors[color]
        print(f"\033[{color_code}m{text}\033[0m")

print("=====================================")
print("Author: Running_XiaoHang")
print("Github: https://github.com/laodengbulao/Azero-SMP")
print("=====================================")

class_names = ['胜', '平', '负']
expect = input("Input Expect: ")
print("\r正在爬取数据[>...........]0%", end='')
data = get_data_to_pred(expect)
matches = data["赛事"]
home_teams = data["主队"]
away_teams = data["客队"]
en1 = joblib.load("./models/LabelEncoder.pkl")
tem = np.array(data["亚盘"])
tem = tem.reshape(-1, 1)
data["亚盘"] = en1.transform(tem)

print("\r正在爬取数据2[===>........]{}%".format(random.randint(15, 35)), end='')
X_datas = data[['亚盘水位',"亚盘",'赔率水位','胜','平','负']].astype(float)
X_datas = np.array(X_datas)
gbc_clf = joblib.load("./models/gbc_clf.pkl")

print("\r正在预测[======>.....]{}%".format(random.randint(40, 60)), end='')
y_preds = gbc_clf.predict(X_datas.reshape(-1,6))
time.sleep(1)

print("\r正在预测[=========>..]{}%".format(random.randint(70, 90)), end='')
time.sleep(1)

print("\r预测完成[============]100%")
for i in range(14):
    print("{} {}VS{} 预测: {}".format(matches[i], home_teams[i], away_teams[i], class_names[y_preds[i].astype(int)]), flush=True)
