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
# print(csort)
# print(dsort)
# print(esort)
# print(fsort)
# print(gsort)
# print(hsort)
# print(isort)
# print(jsort)
# print(ksort)
# print(lsort)
# print(msort)
# print(nsort)
# print(osort)
# print(psort)
# print(qsort)
# print(rsort)
# print(ssort)
# print(tsort)
# print(usort)
# print(vsort)
# print(wsort)

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
features=zip(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w)

# Train the model using the training sets
model.fit(features,a)

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
classify=[]
print("Enter Grades")
for i in range(0,22):
    classify.append(input())
ans=[]
ans.append(bsort.index(classify[0]))
ans.append(csort.index(classify[1]))
ans.append(dsort.index(classify[2]))
ans.append(esort.index(classify[3]))
ans.append(fsort.index(classify[4]))
ans.append(gsort.index(classify[5]))
ans.append(hsort.index(classify[6]))
ans.append(isort.index(classify[7]))
ans.append(jsort.index(classify[8]))
ans.append(ksort.index(classify[9]))
ans.append(lsort.index(classify[10]))
ans.append(msort.index(classify[11]))
ans.append(nsort.index(classify[12]))
ans.append(osort.index(classify[13]))
ans.append(psort.index(classify[14]))
ans.append(qsort.index(classify[15]))
ans.append(rsort.index(classify[16]))
ans.append(ssort.index(classify[17]))
ans.append(tsort.index(classify[18]))
ans.append(usort.index(classify[19]))
ans.append(vsort.index(classify[20]))
ans.append(wsort.index(classify[21]))

predicted= model.predict([ans])
print("Predicted Value:",predicted[0])