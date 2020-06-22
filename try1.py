import pandas as pd
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#Create a Gaussian Classifier
model = GaussianNB()
df2 = pd.read_csv('G:/be project/mushrooms.csv')
la=[]
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
i=[]
j=[]
k=[]
l=[]
m=[]
n=[]
o=[]
p=[]
q=[]
r=[]
s=[]
t=[]
u=[]
v=[]
w=[]
ing=0
for row in df2.itertuples():
    a.append(row[1])
    b.append(row[2])
    c.append(row[3])
    d.append(row[4])
    e.append(row[5])
    f.append(row[6])
    g.append(row[7])
    h.append(row[8])
    i.append(row[9])
    j.append(row[10])
    k.append(row[11])
    l.append(row[12])
    m.append(row[13])
    n.append(row[14])
    o.append(row[15])
    p.append(row[16])
    q.append(row[17])
    r.append(row[18])
    s.append(row[19])
    t.append(row[20])
    u.append(row[21])
    v.append(row[22])
    w.append(row[23])
    # if ing==0:
    #     print(a)
    #     print(b)
    #     print(c)
    #     print(d)
    #     print(e)
    #     print(f)
    #     print(g)
    #     print(h)
    #     print(i)
    #     print(j)
    #     print(k)
    #     print(l)
    #     print(m)
    #     print(n)
    #     print(o)
    #     print(p)
    #     print(q)
    #     print(r)
    #     print(s)
    #     print(t)
    #     print(u)
    #     print(v)
    #     print(w)
    #     ing=1
asort = list(sorted(set(a)))
bsort = list(sorted(set(b)))
csort = list(sorted(set(c)))
dsort = list(sorted(set(d)))
esort = list(sorted(set(e)))
fsort = list(sorted(set(f)))
gsort = list(sorted(set(g)))
hsort = list(sorted(set(h)))
isort = list(sorted(set(i)))
jsort = list(sorted(set(j)))
ksort = list(sorted(set(k)))
lsort = list(sorted(set(l)))
msort = list(sorted(set(m)))
nsort = list(sorted(set(n)))
osort = list(sorted(set(o)))
psort = list(sorted(set(p)))
qsort = list(sorted(set(q)))
rsort = list(sorted(set(r)))
ssort = list(sorted(set(s)))
tsort = list(sorted(set(t)))
usort = list(sorted(set(u)))
vsort = list(sorted(set(v)))
wsort = list(sorted(set(w)))
# print(bsort)
from sklearn import preprocessing
# # #creating labelEncoder
le = preprocessing.LabelEncoder()
# # # Converting string labels into numbers.
a=le.fit_transform(a)
b=le.fit_transform(b)
c=le.fit_transform(c)
d=le.fit_transform(d)
e=le.fit_transform(e)
f=le.fit_transform(f)
g=le.fit_transform(g)
h=le.fit_transform(h)
i=le.fit_transform(i)
j=le.fit_transform(j)
k=le.fit_transform(k)
l=le.fit_transform(l)
m=le.fit_transform(m)
n=le.fit_transform(n)
o=le.fit_transform(o)
p=le.fit_transform(p)
q=le.fit_transform(q)
r=le.fit_transform(r)
s=le.fit_transform(s)
t=le.fit_transform(t)
u=le.fit_transform(u)
v=le.fit_transform(v)
w=le.fit_transform(w)
features=[b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]
train_features=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
train_labels=[]
test_features=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
test_labels=[]
for trn in range(6001):
    train_features[0].append(features[0][trn])
    train_features[1].append(features[1][trn])
    train_features[2].append(features[2][trn])
    train_features[3].append(features[3][trn])
    train_features[4].append(features[4][trn])
    train_features[5].append(features[5][trn])
    train_features[6].append(features[6][trn])
    train_features[7].append(features[7][trn])
    train_features[8].append(features[8][trn])
    train_features[9].append(features[9][trn])
    train_features[10].append(features[10][trn])
    train_features[11].append(features[11][trn])
    train_features[12].append(features[12][trn])
    train_features[13].append(features[13][trn])
    train_features[14].append(features[14][trn])
    train_features[15].append(features[15][trn])
    train_features[16].append(features[16][trn])
    train_features[17].append(features[17][trn])
    train_features[18].append(features[18][trn])
    train_features[19].append(features[19][trn])
    train_features[20].append(features[20][trn])
    train_features[21].append(features[21][trn])
    train_labels.append(a[trn])

train_features=zip(train_features[0],train_features[1],train_features[2],train_features[3],train_features[4],train_features[5],train_features[6],train_features[7],train_features[8],train_features[9],train_features[10],train_features[11],train_features[12],train_features[13],train_features[14],train_features[15],train_features[16],train_features[17],train_features[18],train_features[19],train_features[20],train_features[21])

for trn in range(6001,8123):
    test_features[0].append(features[0][trn])
    test_features[1].append(features[1][trn])
    test_features[2].append(features[2][trn])
    test_features[3].append(features[3][trn])
    test_features[4].append(features[4][trn])
    test_features[5].append(features[5][trn])
    test_features[6].append(features[6][trn])
    test_features[7].append(features[7][trn])
    test_features[8].append(features[8][trn])
    test_features[9].append(features[9][trn])
    test_features[10].append(features[10][trn])
    test_features[11].append(features[11][trn])
    test_features[12].append(features[12][trn])
    test_features[13].append(features[13][trn])
    test_features[14].append(features[14][trn])
    test_features[15].append(features[15][trn])
    test_features[16].append(features[16][trn])
    test_features[17].append(features[17][trn])
    test_features[18].append(features[18][trn])
    test_features[19].append(features[19][trn])
    test_features[20].append(features[20][trn])
    test_features[21].append(features[21][trn])
    test_labels.append(a[trn])

test_features=zip(test_features[0],test_features[1],test_features[2],test_features[3],test_features[4],test_features[5],test_features[6],test_features[7],test_features[8],test_features[9],test_features[10],test_features[11],test_features[12],test_features[13],test_features[14],test_features[15],test_features[16],test_features[17],test_features[18],test_features[19],test_features[20],test_features[21])
# Train the model using the training sets
model.fit(train_features,train_labels)

# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
# print(h)
# print(i)
# print(j)
# print(k)
# print(l)
# print(m)
# print(n)
# print(o)
# print(p)
# print(q)
# print(r)
# print(s)
# print(t)
# print(u)
# print(v)
# print(w)
#Predict Output
# classify=[]
# for i in range(0,22):
#     classify.append(input())
# ans=[]
# ans.append(bsort.index(classify[0]))
# ans.append(csort.index(classify[1]))
# ans.append(dsort.index(classify[2]))
# ans.append(esort.index(classify[3]))
# ans.append(fsort.index(classify[4]))
# ans.append(gsort.index(classify[5]))
# ans.append(hsort.index(classify[6]))
# ans.append(isort.index(classify[7]))
# ans.append(jsort.index(classify[8]))
# ans.append(ksort.index(classify[9]))
# ans.append(lsort.index(classify[10]))
# ans.append(msort.index(classify[11]))
# ans.append(nsort.index(classify[12]))
# ans.append(osort.index(classify[13]))
# ans.append(psort.index(classify[14]))
# ans.append(qsort.index(classify[15]))
# ans.append(rsort.index(classify[16]))
# ans.append(ssort.index(classify[17]))
# ans.append(tsort.index(classify[18]))
# ans.append(usort.index(classify[19]))
# ans.append(vsort.index(classify[20]))
# ans.append(wsort.index(classify[21]))

predicted= model.predict(test_features)
print("accuracy",accuracy_score(test_labels,predicted))