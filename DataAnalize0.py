import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy
df = pandas.read_csv('Data/titanic.csv', index_col='PassengerId')
#print(df.loc[df["Sex"]=="male"])#first task
#print(df.loc[df["Sex"]=="female"])
#2
#print(df['Survived'].value_counts())
#3
#print(df["Pclass"].value_counts())
#4
#print(df["Age"].mean())
#print(df["Age"].median())
#5
#print(df["SibSp"].corr(df["Parch"]))
#6
#b=[]
#a=(df.loc[df["Sex"]=="female"])
#my_seria=pandas.Series([])
#my_seria=a["Name"].str.split(",").str[1]
#print(my_seria.mode())
#II
a=(df.loc[:,["Pclass","Fare","Age","Sex","Survived"]])#ВЫБИРАЕМ ИЗ ДАТАСЕТА ВСЕ НЕОХОДИМЫЕ КОЛОНКИ
a=a.dropna()#удаляем все записи с неполной информацией
b=(a.loc[:,["Survived"]])#выделяем целевую переменную
a=(a.loc[:,["Pclass","Fare","Age","Sex"]])#выделяем обучающую выборку
a.Sex.replace(["male","female"] , [1,0],inplace=True)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(a,b)
importances = clf.feature_importances_
print(importances)