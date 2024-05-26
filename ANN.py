
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from DBConfig import DBConnection
def ann_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()

    ann_clf = MLPClassifier()

    ann_clf.fit(X_train, y_train)

    predicted = ann_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100



    values = ("ANN", float(accuracy), float(precision),float(recall),float(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("ANN=",accuracy,precision,recall,fscore)

    return accuracy, precision, recall, fscore




