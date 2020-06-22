from flask import Flask, request, jsonify

app = Flask(__name__)

from sklearn import linear_model
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import MeanShift
import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
from statsmodels.tsa.arima_model import ARIMA
import re
from statsmodels.tsa.stattools import adfuller
# importing packages for the prediction of time-series data
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

modelList = list()
cropList = list()
markets = list()
marketList = list()
model_list = []
out_second = pd.DataFrame() 

kmeans = None
df_train_mod_two = pd.read_csv('Datasets/Dataset(Module-2).csv')
df_tr_mod_two = df_train_mod_two
df_train_mod_three = pd.read_csv('Datasets/Dataset(Module-3).csv')
df_tr_mod_three = df_train_mod_three

def train_mod_one():
    spreadsheet = pd.ExcelFile('Datasets/Dataset(Module-1).xlsx')
    df2 = spreadsheet.parse('Sheet1')
        # df2.head()
    crops_combo = []
    crop_season = []
    for row in df2.itertuples():
            # use row[5] strip
        crop_season.append(str(row[5]).strip())
            # use row[4] strip
        crop_season.append(str(row[4]).strip())
        crops_combo.append(crop_season)
        crop_season = []
    a = set(tuple(i) for i in crops_combo)
        # print('length of set :',len(a))
        # for l in sorted(a):
        #     print(l[0]," ",l[1])
    ar = []
    ar2d = []
    x_list = []
    z_list = []
    i = 0
    y_list = []
    for l in sorted(a):
        for row in df2.itertuples():
            if str(row[5]).strip() == l[0] and str(row[4]).strip() == l[1]:
                ar.append(row[8])
                x_list.append(row[8])
                ar.append(row[9])
                z_list.append(row[9])
                y_list.append(row[10])
                ar2d.append(ar)
                ar = []
        if len(ar2d) < 10:
            continue
        nar2d = np.array(ar2d)
        nar2d = np.nan_to_num(nar2d)
        ny = np.array(y_list)
        ny = np.nan_to_num(ny)
        X_train = nar2d[:-4]
        X_test = nar2d[-4:]
        y_train = ny[:-4]
        y_test = ny[-4:]
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X_train, y_train)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_y_ransac = ransac.predict(X_test)
        model_list.append([ransac, l[0], l[1]])


def yieldPredict(test_input):
    y_predicted = []
    for m in model_list:
        ys = 10 * m[0].predict(test_input)
        if ys[0] > 0:
            y_predicted.append([ys[0], m[1], m[2]])
        # print(m[1], m[2])
            # print(ys)
    y_predicted.sort(reverse=True)
    df = pd.DataFrame(np.array(y_predicted), columns=['Current Yeild(q/ha)', 'Crop', 'Season'])
        # for i in y_predicted:
        #     print(i[0], " ", i[1], " ", i[2])
    return df


def train_mod_two():
    clmns = ['Available-N(kg/ha)', 'Available-P(kg/ha)', 'Available-K(kg/ha)']
    tr_clmns = ['Available-N(kg/ha)', 'Available-P(kg/ha)', 'Available-K(kg/ha)']
    i = 70
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df_tr_mod_two[tr_clmns])
    labels = kmeans.labels_
    df_tr_mod_two['clusters'] = labels
    clmns.extend(['clusters'])
    print(clmns)
    df_clust = df_tr_mod_two[clmns]
    df_test = df_tr_mod_two[tr_clmns]
    return kmeans


def varietyPredict(kmeans, values, out_one):
    prediction = kmeans.predict(values)
    out_second = df_tr_mod_two[['Crop', 'Variety', 'Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)', 'Yeild(q/ha)']].loc[df_tr_mod_two['clusters'] == prediction[0]]
    out_second = pd.merge(out_second, out_one, on='Crop', how='inner')
    # print(out_second.to_string())
    return out_second

def printing():
    df_list = out_second.values.tolist()
    # JSONP_data = jsonpify(df_list)
    result=[]
    print(df_list)
    # df_str = String(df_list)
    result=[]
    df_list=out_second.values.tolist()
    for val in df_list:
        temp=[]
        # for val2 in val:
        temp.append(val[0])
        temp.append(val[1])
        result.append(temp)
    final=[]
    for i in result:
        if i in final:
            continue
        else:
            final.append(i)
    print(final)
    ans="CROP NAME \t VARIETY\n"

    for j in final:
        ans+=j[0]+"\t"+j[1]+"\n"
    print(ans)
    return ans
    # return jsonify({"paramkey",})
    # "crop":"","variety":"","reqd_n":"","reqd_p":"","reqd_k":"","yield":"","current_yield":""})
    # print(out_second.head())



@app.route('/yield/<user>',methods=['GET'])
def hello_user(user):
    print("Training module - 1")
    train_mod_one()
    print("Training module - 2")
    kmeans = train_mod_two()
    # print("Training module - 3")
    # meanShift = train_mod_three()
    # print("Training module - 4")
    # train_mod_four()
    # while p != float(0):
    # print("Enter test data")
    #     # p = float(input("Enter precipitaion : "))
        # t = input("Enter temperature : ")
    user1=list(user.split(","))
    out_one = yieldPredict([[float(user1[0]), float(user1[1])]])
        # cropN = input("Enter N value : ")
        # cropP = input("Enter P value : ")
        # cropK = input("Enter K value : ")
    out_second = varietyPredict(kmeans, np.array([[user1[2],user1[3],user1[4]]]), out_one)
    out_second=out_second[['Crop', 'Variety', 'Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)', 'Yeild(q/ha)', 'Current Yeild(q/ha)']]
        # out_second has Columns: [Crop, Variety, Required-N(kg/ha), Required-P(kg/ha), Required-K(kg/ha), Yeild(q/ha), Current Yeild(q/ha), Season]
    print("Output from module-2")
        # print(out_second.to_string())
    # df_str = String(df_list)
    result=[]
    df_list=out_second.values.tolist()
    for val in df_list:
        temp=[]
        # for val2 in val:
        temp.append(val[0])
        temp.append(val[1])
        result.append(temp)
    final=[]
    for i in result:
        if i in final:
            continue
        else:
            final.append(i)
    print(final)
    ans="CROP_NAME \t VARIETY\n"

    for j in final:
        ans+=j[0]+"\t"+j[1]+"\n"
    print(ans)
    return jsonify({"paramkey":ans})


if __name__ == "__main__":
    app.run(host='192.168.10.104', port=5000)