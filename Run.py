#!pip install tensorflow
#!pip install numpy
#!pip install pandas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=====================================")
print("Author: Running_XiaoHang")
print("Github: https://github.com/laodengbulao")
print("=====================================")

import numpy as np
print('Load Model...')
from keras.models import load_model
class_names = ['Win', 'Flat', 'Failure']
model = load_model("./model.h5")
print('Ok!')

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
X_new = [WHH, WHD, WHA, VCH, VCD, VCA, B365H, B365D, B365A]
X_new = np.array(X_new)
X_new = X_new.reshape(1,-1)
proba = model.predict(X_new)
proba = np.array(proba)
result = np.argmax(proba)


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