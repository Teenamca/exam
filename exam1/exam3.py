
import pandas as pd
from sklearn import tree,metrics,model_selection
data=pd.read_csv('car.csv',names=['buying','mains','indoor','presence','lug_boot','safety','class'])
data['class'],class_names=pd.factorize(data['class'])
print(class_names)
print(data['class'].unique())
data['buying'],_=pd.factorize(data['buying'])
data['mains'],_=pd.factorize(data['mains'])
data['indoor'],_=pd.factorize(data['indoor'])
data['presence'],_=pd.factorize(data['presence'])
data['lug_boot'],_=pd.factorize(data['lug_boot'])
data['safety'],_=pd.factorize(data['safety'])
data.head()
data.info()
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)
dtree=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)

accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:{:.2f}".format(accuracy))
count_misclassified=(y_test!=y_pred)
print("Misclassified samples.{}".format(count_misclassified))
