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
    out_second = df_tr_mod_two[
        ['Crop', 'Variety', 'Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)', 'Yeild(q/ha)']].loc[
        df_tr_mod_two['clusters'] == prediction[0]]
    out_second = pd.merge(out_second, out_one, on='Crop', how='inner')
    # print(out_second.to_string())
    return out_second


def train_mod_three():
    clmns = ['n%', 'p%', 'k%']
    tr_clmns = ['n%', 'p%', 'k%']
    meanShift = MeanShift(bandwidth=0.1).fit(df_tr_mod_three[tr_clmns])
    labels = meanShift.labels_
    df_tr_mod_three['clusters'] = labels
    clmns.extend(['clusters'])
    df_clust = df_tr_mod_three[clmns]
    df_test = df_tr_mod_three[tr_clmns]
    clusterCenters = meanShift.cluster_centers_
    n_cluster = len(np.unique(labels))
    return meanShift


def fertilizerPredict(input):
    outputdf = pd.DataFrame(columns=['Suggested_fert', 'Suggested_fert_price/kg', 'Reqd_N', 'Reqd_P', 'Reqd_K',
                                     'Total_price', 'Urea_reqd', 'triple superphosphate_reqd',
                                     'Potassium_chloride_reqd', 'Urea_price', 'triple superphosphate_price',
                                     'Potassium_chloride_price', 'Grand_total'])
    for i in input:
        arr = [float(i[0]), float(i[1]), float(i[2])]
        b = float(arr[0]) + float(arr[1]) + float(arr[2])
        farr = [0, 0, 0]
        for i in range(0, 3):
            farr[i] = (float(arr[i] * 100)) / b
        # print(farr)
        val = np.asarray(farr)
        values = [val]
        prediction = meanShift.predict(values)
        # print("Predicted Cluster " + str(prediction))
        output = df_tr_mod_three[
            ['Fert_name', 'n%', 'p%', 'k%', 'n_ratio', 'p_ratio', 'k_ratio', 'price/kg', 'price(100kg)bag']].loc[
            df_tr_mod_three['clusters'] == prediction[0]]
        tr_clmns = ['n%', 'p%', 'k%']
        # print(output)
        df_test = df_tr_mod_three[tr_clmns]
        # output
        a = [0, 0, 0]
        cp_a = [0, 0, 0]
        b = [0, 0, 0]
        for row in output.itertuples():
            # for i in range(0, 3):
            #     if (row[i + 2] != 0):
            #         a[i] = arr[i] / row[i + 2]
            # a.sort()
            # if (a[0] == 0):
            #     for i in range(1, 3):
            #         a[i] = a[1]
            #         cp_a[i] = a[i] * row[9]
            # else:
            #     for i in range(0, 3):
            #         a[i] = a[0]
            #         cp_a[i] = a[i] * row[9]
            # for i in range(0, 3):
            #     b[i] = a[i] * row[i + 2]
            for i in range(0, 3):
                if (row[i + 2] != 0):
                    a[i] = arr[i] / row[i + 2]
            a1 = [0, 0, 0]
            for i in range(0, 3):
                a1[i] = a[i]
            a1.sort()
            if (a1[0] == 0 and a1[1] == 0):
                for i in range(2, 3):
                    a1[i] = a1[2]
                    cp_a[i] = a1[i] * row[9]
            elif (a1[0] == 0):
                for i in range(1, 3):
                    a1[i] = a1[1]
                    cp_a[i] = a1[i] * row[9]
            else:
                for i in range(0, 3):
                    a1[i] = a1[0]
                    cp_a[i] = a1[i] * row[9]
            x = 0
            for i in range(0, 3):
                if (a1[i] != 0):
                    x = a1[i]
                    break
            for i in range(0, 3):
                if (a[i] != 0):
                    a[i] = x
            for i in range(0, 3):
                b[i] = a[i] * row[i + 2]
        c = [0, 0, 0]
        cp_c = [0, 0, 0]

        c[0] = arr[0] - b[0]
        c[0] = c[0] / 46
        cp_c[0] = c[0] * 536

        c[1] = arr[1] - b[1]
        c[1] = c[1] / 46
        cp_c[1] = c[1] * 837.2

        c[2] = arr[2] - b[2]
        c[2] = c[2] / 60
        cp_c[2] = c[2] * 1570
        tot_cp = [0, 0, 0]
        tot = 0
        cp_a.sort()
        cp = cp_a[2]
        for i in range(0, 3):
            tot_cp[i] = cp_c[i]
            tot = tot + tot_cp[i]
        tot = tot + cp
        new = output[['Fert_name', 'price/kg']].copy()
        # new1=new.values.T.tolist()
        new1 = list(new.values.flatten())
        fin = {'Suggested_fert': [new1[0]], 'Suggested_fert_price/kg': [new1[1]], 'Reqd_N': [b[0]], 'Reqd_P': [b[1]],
               'Reqd_K': [b[2]], 'Total_price': [cp_a[2]], 'Urea_reqd': [c[0]], 'triple superphosphate_reqd': [c[1]],
               'Potassium_chloride_reqd': [c[2]], 'Urea_price': [cp_c[0]], 'triple superphosphate_price': [cp_c[1]],
               'Potassium_chloride_price': [cp_c[2]], 'Grand_total': tot}
        fin1 = pd.DataFrame(fin, columns=['Suggested_fert', 'Suggested_fert_price/kg', 'Reqd_N', 'Reqd_P', 'Reqd_K',
                                          'Total_price', 'Urea_reqd', 'triple superphosphate_reqd',
                                          'Potassium_chloride_reqd', 'Urea_price', 'triple superphosphate_price',
                                          'Potassium_chloride_price', 'Grand_total'])
        # print(fin1)
        outputdf = pd.concat([outputdf, fin1], ignore_index=True)
    return outputdf


def train_mod_four():
    from statsmodels.tsa.arima_model import ARIMA
    trainog = pd.read_excel('Datasets/Dataset(Module-4).xlsx')
    crops = trainog['crop/commodity'].unique()
    for i in crops:
        train = trainog
        train = train[(train['Modal Price(Rs./Quintal)'] != "NR")]
        traintemp = train[(train['crop/commodity'] == i)]
        market = list(train['Market'].unique())
        # print(market)
        # print("For crop " + i)
        # print(market)
        for j in market:
            train = traintemp[(traintemp['Market'] == j)]
            # print(i + " " + j)
            data = train['Modal Price(Rs./Quintal)']
            Date1 = train['Date']
            train1 = train[['Date', 'Modal Price(Rs./Quintal)']]
            train2 = train1.set_index('Date')
            train2.dropna(inplace=True)
            train2.sort_index(inplace=True)
            ts = train2['Modal Price(Rs./Quintal)']
            #             if(test_stationarity(ts)==0):
            # print(ts.size)
            if (ts.size > 50):
                ts_list = ts.tolist()
                ts_log = dragon.log(ts_list)
                # plot.plot(ts_log, color="green")
                # plot.show()
                #             print(type(ts_log))
                test_stationarity(pd.Series(ts_log))
                ts_log_diff = ts_log - pd.Series(ts_log).shift()
                ts_log_diff.dropna(inplace=True)
                test_stationarity(ts_log_diff)
                # follow lag
                # model = ARIMA(ts_log, order=(1, 1, 0))
                # results_ARIMA = model.fit(disp=-1)
                # plot.plot(ts_log_diff)
                # plot.plot(results_ARIMA.fittedvalues, color='red')
                # plot.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues - ts_log_diff) ** 2))
                # plot.show()
                # follow error
                # model = ARIMA(ts_log, order=(0, 1, 1))
                # results_MA = model.fit(disp=-1)
                # plot.plot(ts_log_diff)
                # plot.plot(results_MA.fittedvalues, color='red')
                # plot.title('RSS: %.7f' % sum((results_MA.fittedvalues - ts_log_diff) ** 2))
                # plot.show()
                from statsmodels.tsa.arima_model import ARIMA
                model = ARIMA(ts_log, order=(2, 1, 0))
                results_ARIMA = model.fit(disp=-1)
                modelList.append(results_ARIMA)
                markets.append(j)
                cropList.append(i)


def cropPricePrediction(Crop):
    # print(Crop)
    # print(cropList)
    for j in Crop:
        for i in range(0, len(cropList)):
            if cropList[i] == j:
                print("For crop " + cropList[i] + " market " + markets[i] + " prediction in next month is " + str(int(dragon.exp(modelList[i].forecast()[0])))+" Rs/Quintal")


def test_stationarity(x):
    toreturn = 0
    # Determing rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()
    rolstd = x.rolling(window=12, center=False).std()
    # Plot rolling statistics:
    orig = plot.plot(x, color='blue', label='Original')
    mean = plot.plot(rolmean, color='red', label='Rolling Mean')
    std = plot.plot(rolstd, color='black', label='Rolling Std')
    # plot.legend(loc='best')
    # plot.title('Rolling Mean & Standard Deviation')
    # plot.show(block=False)
    #     Perform Dickey Fuller test
    result = adfuller(x)
    # print('ADF Stastistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            # print("The graph is non stationery")
            break
        else:
            # print("The graph is stationery")
            toreturn = 1
            break;
    # # print('Critical values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f ' % (key, value))
    return toreturn


if __name__ == "__main__":
    p = 10
    print("Training module - 1")
    train_mod_one()
    print("Training module - 2")
    kmeans = train_mod_two()
    print("Training module - 3")
    meanShift = train_mod_three()
    print("Training module - 4")
    train_mod_four()
    while p != float(0):
        print("Enter test data")
        p = float(input("Enter precipitaion : "))
        t = input("Enter temperature : ")
        out_one = yieldPredict([[float(p), float(t)]])
        cropN = input("Enter N value : ")
        cropP = input("Enter P value : ")
        cropK = input("Enter K value : ")
        out_second = varietyPredict(kmeans, np.array([[cropN, cropP, cropK]]), out_one)
        out_second=out_second[['Crop', 'Variety', 'Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)', 'Yeild(q/ha)', 'Current Yeild(q/ha)']]
        # out_second has Columns: [Crop, Variety, Required-N(kg/ha), Required-P(kg/ha), Required-K(kg/ha), Yeild(q/ha), Current Yeild(q/ha), Season]
        print("Output from module-2")
        # print(out_second.to_string())
        out_three = fertilizerPredict(list(out_second[['Required-N(kg/ha)', 'Required-P(kg/ha)', 'Required-K(kg/ha)']].values))
        # print(out_three.to_string())
        completeout = pd.merge(out_second, out_three, left_index=True, right_index=True)
        completeout=completeout.sort_values(by='Grand_total')
        print(completeout.to_string())
        crop=out_second['Crop'].unique()
        cropPricePrediction(list(crop))
