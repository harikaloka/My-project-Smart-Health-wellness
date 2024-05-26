
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix


from DBConfig import DBConnection

def dt_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()
    cursor.execute("delete from evaluations")
    db.commit()

    dtc_clf = DecisionTreeClassifier()

    dtc_clf.fit(X_train, y_train)

    predicted = dtc_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100


    values = ("DTC", str(accuracy), str(precision),str(recall),str(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("DTC=",accuracy,precision,recall,fscore)
    return accuracy,precision,recall,fscore





