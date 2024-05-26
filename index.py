from flask import Flask, render_template, request,flash
import pandas as pd
from flask import Response
import csv

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import json
from sklearn.neighbors import KNeighborsClassifier
from flask import session
from werkzeug.utils import secure_filename
import sys
from sklearn.preprocessing import LabelEncoder
import os
import io
from RFC import rfc_evaluation
from DTC import dt_evaluation
from SVM import svm_evaluation
from ANN import ann_evaluation

import base64
from DBConfig import DBConnection
import shutil
from random import randint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import numpy as np
from sklearn.model_selection import train_test_split
from DBConfig import DBConnection


app = Flask(__name__)
app.secret_key = "abc"

dict={}
accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/ahome")
def ahome():
    return render_template("admin_home.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/newuser")
def newuser():
    return render_template("register.html")

@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    name = request.form.get('name')
    uid = request.form.get('uid')
    pwd = request.form.get('pwd')
    email = request.form.get('email')
    mno = request.form.get('mno')
    db = DBConnection.getConnection()
    cursor = db.cursor()
    sql = "select count(*) from register where userid='" + uid + "'"
    cursor.execute(sql)
    res = cursor.fetchone()[0]
    if res > 0:
        return render_template("register.html", msg="User Id already exists..!")
    else:
        sql = "insert into register values(%s,%s,%s,%s,%s)"
        values = (name, uid, pwd, email, mno)
        cursor.execute(sql,values)
        db.commit()
        return render_template("user.html", msg="Registered Successfully..! Login Here.")
    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():
        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        db = DBConnection.getConnection()
        cursor = db.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid
            qry = "select * from register where userid= '" + uid + " ' "
            cursor.execute(qry)
            val = cursor.fetchall()
            for values in val:
                name = values[0]
                print(name)

            return render_template("user_home.html",name=name)
        else:

            return render_template("user.html",msg2="Invalid Credentials")
        return ""

@app.route("/uhome")
def uhome():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    uid = session['uid']
    qry = "select * from register where userid= '" + uid + " ' "
    cursor.execute(qry)
    vals = cursor.fetchall()
    for values in vals:
        name = values[0]
        print(name)

    return render_template("user_home.html",name=name)


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/admin_home")
def admin_home():
    return render_template("admin_home.html")

@app.route("/perevaluations")
def perevaluations():
    accuracy_graph()
    precision_graph()
    recall_graph()
    f1score_graph()
    return render_template("metrics.html")


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")

        return ""

@app.route("/evaluations" )
def evaluations():

    rf_list=[]
    dt_list = []
    svm_list = []
    ann_list = []
    metrics=[]

    data_train = pd.read_csv('../HAR/dataset/aw_fb_data.csv')

    data_train = data_train.drop(["unnamed"], axis=1)
    data_train = data_train.drop(["X1"], axis=1)
    data_train = data_train.drop(['device'], axis=1)

    labels = {'Lying': 0, 'Running 3 METs': 1, 'Running 5 METs': 2, 'Running 7 METs': 3, 'Self Pace walk': 4,
              'Sitting': 5}

    data_train['target'] = data_train['activity'].map(labels)
    data_train = data_train.drop(['activity'], axis=1)

    y = data_train["target"]
    del data_train["target"]
    print("maindataset:", data_train)

    #print("dataset",data_train)

    #y = data_train['activity']

    #del data_train['activity']

    X = data_train

    # Split train test: 80 % - 20 %
    print("x=",X)
    print("y=", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

    #print("X_train:", X_train)
    #print("X_test:", X_test)
    #print("Y_train:", y_train)
    #print("Y_train:", y_test)

    # Convert target variable 'activity' to numerical labels using label encoder
    #label_encoder = LabelEncoder()
    #y_train = label_encoder.fit_transform(y_train)
    #y_test = label_encoder.transform(y_test)



    accuracy_dt, precision_dt, recall_dt, fscore_dt = dt_evaluation(X_train, X_test, y_train, y_test)
    dt_list.append("DTC")
    dt_list.append(accuracy_dt)
    dt_list.append(precision_dt)
    dt_list.append(recall_dt)
    dt_list.append(fscore_dt)


    accuracy_svm, precision_svm, recall_svm, fscore_svm  = svm_evaluation(X_train, X_test, y_train, y_test)
    svm_list.append("SVM")
    svm_list.append(accuracy_svm)
    svm_list.append(precision_svm)
    svm_list.append(recall_svm)
    svm_list.append(fscore_svm)

    accuracy_rf, precision_rf, recall_rf, fscore_rf=rfc_evaluation(X_train, X_test, y_train, y_test)
    rf_list.append("RFC")
    rf_list.append(accuracy_rf)
    rf_list.append(precision_rf)
    rf_list.append(recall_rf)
    rf_list.append(fscore_rf)



    accuracy_ann, precision_ann, recall_ann,fscore_ann = ann_evaluation(X_train, X_test, y_train, y_test)
    ann_list.append("ANN")
    ann_list.append(accuracy_ann)
    ann_list.append(precision_ann)
    ann_list.append(recall_ann)
    ann_list.append(fscore_ann)


    metrics.clear()

    metrics.append(dt_list)
    metrics.append(svm_list)
    metrics.append(rf_list)
    metrics.append(ann_list)


    return render_template("evaluations.html", evaluations=metrics)


def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from evaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list
    print("height=",height)
    bars = ('DTC','SVM','RFC','ANN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red', 'green', 'blue', 'orange'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis on ML Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()



def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    precision_list.clear()

    cursor.execute("select precesion from evaluations")
    pdata = cursor.fetchall()

    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('DTC', 'SVM', 'RFC', 'ANN')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet', 'blue'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis on ML Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()



def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    recall_list.clear()

    cursor.execute("select recall from evaluations")
    recdata = cursor.fetchall()

    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list
    #print("height=",height)
    bars = ('DTC', 'SVM', 'RFC', 'ANN')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'gray', 'violet'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis on ML Recall')
    plt3.savefig('static/recall.png')
    plt3.clf()


def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    f1score_list.clear()

    cursor.execute("select f1score from evaluations")
    fsdata = cursor.fetchall()

    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list
    print("fheight=",height)
    bars = ('DTC', 'SVM', 'RFC', 'ANN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['gray', 'green', 'orange', 'brown'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis on ML F1-Score')
    plt4.savefig('static/f1score.png')
    plt4.clf()



@app.route("/detection_HAR", methods =["GET", "POST"])
def detection_HAR():

    df = pd.read_csv("../HAR/dataset/aw_fb_data.csv")

    df = df.drop(["unnamed"], axis=1)
    df = df.drop(["X1"], axis=1)
    df = df.drop(['device'], axis=1)

    labels = {'Lying': 0, 'Running 3 METs': 1, 'Running 5 METs': 2, 'Running 7 METs': 3, 'Self Pace walk': 4,
              'Sitting': 5}

    df['target'] = df['activity'].map(labels)
    df = df.drop(['activity'], axis=1)
    print("maindataset:", df)
    y_train = df["target"]
    del df["target"]
    x_train = df

    testframe = pd.read_csv("../HAR/test.csv")
    testdata = testframe
    testdata = np.array(testframe)
    X_test = testdata.reshape(len(testdata), -1)
    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(x_train, y_train)
    predicted = rfc_clf.predict(X_test)
    result = predicted[0]
    print("res=", result)

    if result==0:
        result="lying"
    elif result==1:
        result="Running 3 METs"
    elif result==2:
        result = "Running 5 METs"
    elif result==3:
        result = "Running 7 METs"
    elif result==4:
        result="Self Pace walk"
    else:
        result = "Sitting"

    result = result
    print("result=",result)

    db = DBConnection.getConnection()
    cursor = db.cursor()

    qry = "select report from reports where conditions = '" + result + "' "

    cursor.execute(qry)
    report = cursor.fetchall()[0][0]
    print("report=", report)

    return render_template("prediction.html", result=result,report=report)



if __name__ == '__main__':
    app.run(host="localhost", port=2222, debug=True)
