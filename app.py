#  Deployment
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# load the dataset
url = "iris.csv"

# column names to use
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# read the dataset from the URL
dataset = pd.read_csv("iris.csv") 

# check the first few rows of iris-classification data
dataset.head()
'-----------------------------------'
# separate the independent and dependent features
X = dataset.iloc[:,:-1]
y = dataset.iloc[:, 4]

# Split dataset into random training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
# feature standardization
scaler = StandardScaler ()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
'-----------------------------------'
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train) 

# make predictions on the testing data
y_predict = model.predict(X_test)

# check results
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 
'-----------------------------------'
import pickle

# save the iris classification model as a pickle file
model_pkl_file = "iris_classifier_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)

'-----------------------------------'
# load model from pickle file
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

# evaluate model 
y_predict = model.predict(X_test)

# check results
print(classification_report(y_test, y_predict)) 
'-----------------------------------'

'''################################################################'''
from flask import Flask, render_template, request
import pickle
import pandas as pd 

app = Flask(__name__)   

#model = pickle.load(open('C:\inetpub\wwwroot\models\iris_classifier_model.pkl', 'rb'))
model = pickle.load(open('iris_classifier_model.pkl', 'rb'))

@app.route("/" ) 
def home(): 
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    prediction = model.predict()
    output = prediction
    
    return output
    return render_template("index.html" ,  prediction)

if __name__ == "__main__": 
    app.run(debug=True) 
