from selenium import webdriver              #导包 无需多言
from selenium.webdriver.common.by import By
import numpy as np
import pandas as pd
import time
import sys
import datetime
#设置浏览器启动项
option = webdriver.ChromeOptions()
option.page_load_strategy = 'eager'#仅加载DOM树和HTML静态资源并解析 提高爬取速度
option.add_argument("--headless") #隐藏浏览器界面 也可以删除该行，观察selenuim如何爬取数据
driver = webdriver.Chrome(options=option) #启动浏览器

def get_dates(year):
    #
    start_date = datetime.date(year=year, month=1, day=1)
    end_date = start_date + datetime.timedelta(days=364)  # 判断闰年或非闰年
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(str(current_date))
        current_date += datetime.timedelta(days=1)
    print(np.array(dates[0][8:10]))
    return np.array(dates)

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
#获取数据
#expects = np.load("./expect.npy", encoding='utf-8')
#定义获取数据预测的函数
def get_data_to_pred(expect):
    #url字段
    url = "https://odds.500.com/index_sfc_{}.shtml".format(expect)
    #尝试连接
    try:
        driver.get(url)
    except:#连接出错的措施
        print_color("连接出错,正在重试", 'yellow')#打印信息
        time.sleep(3)#延时3秒 缓解网络问题和避免网站反爬机制
        driver.get(url)#重新连接
    #定义储存地址
    sum_home_team = []
    sum_home_team = np.array(sum_home_team)
    sum_away_team = []
    sum_away_team = np.array(sum_away_team)
    sum_match = []
    sum_match = np.array(sum_match)
    sum_home_goal, sum_away_goal = [], []
    sum_home_goal, sum_away_goal = np.array(sum_home_goal), np.array(sum_away_goal)
    all_water1 = []
    all_water1 = np.array(all_water1)
    all_asia = []
    all_asia = np.array(all_asia)
    all_water2 = []
    all_water2 = np.array(all_water2)
    all_odd1 = []
    all_odd1 = np.array(all_odd1)
    all_odd2 = []
    all_odd2 = np.array(all_odd2)
    all_odd3 = []
    all_odd3 = np.array(all_odd3)
    all_rate = []
    all_rate = np.array(all_rate)
    #尝试爬取
    try:
        #爬取页面内比赛
        matchs = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[2]/a')
        for match in matchs:
            sum_match = np.append(sum_match, match.text).reshape(-1)
        #获取主客队名
        home_teams = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[5]/a')
        away_teams = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[7]/a')
        for ht in home_teams:
            sum_home_team = np.append(sum_home_team, ht.text)
        for at in away_teams:
            sum_away_team = np.append(sum_away_team, at.text)
        #获取比分
        scores = driver.find_elements(By.XPATH, '//*[@id="main-tbody"]//td[6]/span')
        for score in scores:
            sum_home_goal = np.append(sum_home_goal, score.text[0]).reshape(-1)
            sum_away_goal = np.append(sum_away_goal, score.text[2]).reshape(-1)
        #获取亚盘水位
        water1 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[9]')
        for w1 in water1:
            all_water1 = np.append(all_water1, w1.text).reshape(-1)
        #获取亚盘
        asia = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[10]')
        for a in asia:
            all_asia = np.append(all_asia, a.text).reshape(-1)
        #获取赔率水位
        water2 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[11]')
        for w2 in water2:
            all_water2 = np.append(all_water2, w2.text).reshape(-1)
        #获取赔率
        odds1 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[12]')
        for odd1 in odds1:
            all_odd1 = np.append(all_odd1, odd1.text).reshape(-1)
        odds2 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[13]')
        for odd2 in odds2:
            all_odd2 = np.append(all_odd2, odd2.text).reshape(-1)
        odds3 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[14]')  #
        for odd3 in odds3:
            all_odd3 = np.append(all_odd3, odd3.text).reshape(-1)
        rates = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[15]')
        #获取返还率 #用于强化学习(还未实现)
        for r in rates:
            all_rate = np.append(all_rate, r.text).reshape(-1)

    except:  #爬取该页面失败-->
        print_color("未匹配到数据或浏览器出错", 'red')
    #写入字典
    data = {"赛事": sum_match, "主队": sum_home_team, "客队": sum_away_team,
            "亚盘水位": all_water1, "亚盘": all_asia, "赔率水位": all_water2,
            "胜": all_odd1, "平": all_odd2, "负": all_odd3}
    #转化为DF格式
    data = pd.DataFrame(data)
    #返回
    return data
#获取数据用于训练
def get_data_to_train(expects):
    i = 1
    #获取比赛信息
    for expect in expects:
        #加载页面
        sum_home_team = []
        sum_home_team = np.array(sum_home_team)
        sum_away_team = []
        sum_away_team = np.array(sum_away_team)
        sum_match = []
        sum_match = np.array(sum_match)
        sum_home_goal, sum_away_goal = [], []
        sum_home_goal, sum_away_goal = np.array(sum_home_goal), np.array(sum_away_goal)
        all_water1 = []
        all_water1 = np.array(all_water1)
        all_asia = []
        all_asia = np.array(all_asia)
        all_water2 = []
        all_water2 = np.array(all_water2)
        all_odd1 = []
        all_odd1 = np.array(all_odd1)
        all_odd2 = []
        all_odd2 = np.array(all_odd2)
        all_odd3 = []
        all_odd3 = np.array(all_odd3)
        all_rate = []
        all_rate = np.array(all_rate)
        #获取开始时间
        st = time.time()
        url = "https://odds.500.com/index_sfc_{}.shtml".format(expect)
        try:
            driver.get(url)
        except:
            print_color("连接出错,正在重试", 'yellow')#同上
            time.sleep(3)
            driver.get(url)
        try:
            matchs = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[2]/a')#XPath定位元素
            for match in matchs:
                sum_match = np.append(sum_match, match.text).reshape(-1)
            home_teams = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[5]/a')
            away_teams = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[7]/a')
            for ht in home_teams:
                sum_home_team = np.append(sum_home_team, ht.text)
            for at in away_teams:
                sum_away_team = np.append(sum_away_team, at.text)
            scores = driver.find_elements(By.XPATH, '//*[@id="main-tbody"]//td[6]/span')
            for score in scores:
                sum_home_goal = np.append(sum_home_goal, score.text[0]).reshape(-1)
                sum_away_goal = np.append(sum_away_goal, score.text[2]).reshape(-1)
            water1 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[9]')
            for w1 in water1:
                all_water1 = np.append(all_water1, w1.text).reshape(-1)
            asia = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[10]')
            for a in asia:
                all_asia = np.append(all_asia, a.text).reshape(-1)
            water2 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[11]')
            for w2 in water2:
                all_water2 = np.append(all_water2, w2.text).reshape(-1)
            odds1 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[12]')
            for odd1 in odds1:
                all_odd1 = np.append(all_odd1, odd1.text).reshape(-1)
            odds2 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[13]')
            for odd2 in odds2:
                all_odd2 = np.append(all_odd2, odd2.text).reshape(-1)
            odds3 = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[14]')#
            for odd3 in odds3:
                all_odd3 = np.append(all_odd3, odd3.text).reshape(-1)
            rates = driver.find_elements(By.XPATH, '/html/body/div[7]/table/tbody//td[15]')
            for r in rates:
                all_rate = np.append(all_rate, r.text).reshape(-1)
        except:
            print_color("未匹配到数据或浏览器出错", 'red')
        #尝试保存
        try:
            #写入字典
            data = {"赛事": sum_match, "主队": sum_home_team, "客队": sum_away_team, "主队进球": sum_home_goal,
                    "客队进球": sum_away_goal, "亚盘水位": all_water1,
                    "亚盘": all_asia, "赔率水位": all_water2, "胜": all_odd1, "平": all_odd2, "负": all_odd3,
                        "返还率": all_rate}
            data_df = pd.DataFrame(data)#转换DF格式
            #写入csv文件
            if i == 1: #如果是第一次写入就先创建文件再写入
                data_df.to_csv("all_data_v1.0.csv", index=False, header=True, encoding='gbk')#gbk编码，防止中文乱码
            else: #如果不是第一次写入就定义mode='a'再追加写入
                data_df.to_csv("all_data_v1.0.csv", mode='a', index=False, header=False, encoding='gbk')#同上
        #失败
        except ValueError:
            print_color("数据长度不一致", 'red')#输出+跳过
        except PermissionError:
            print_color("打开文件,进程中断", 'yellow')#输出
            sys.exit()#推出
        #结尾时间
        et = time.time()
        #计算剩余时间 公式：(结尾时间-开始时间)*剩余数据/60 单位：min
        rt = int((len(expects) - i) * (et - st))/60
        print_color("进度:{}/{} 完成率:{}% 剩余时间:{}min".format(i, len(expects), round(i/len(expects)*100, 2), round(rt, 1)), 'blue')#输出
        i += 1                                         #format方法用于显示数据     #round函数定义输出2位小数        同理

#启动爬虫
if __name__ == '__main__':
    print_color("爬取开始", 'green')
    #加载期数(已爬取好)
    expects = np.load("./expect.npy")
    get_data_to_train(expects)
    #get_dates(2023)
    print_color("爬取结束", 'green')
    #关闭浏览器
    driver.quit()