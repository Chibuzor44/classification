from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from util import load_data
from util import metric
import pickle
from imblearn.over_sampling import SMOTE


svc = SVC(kernel="rbf", probability=True)


class SVC_Classification():

    def __init__(self):
        self.svc = svc


    def fit(self, X, y):
        """The takes the data and fits the model.

        Parameters
        ----------
        X: A 2-dimensional numpy array of numerical values, of list of lists.
        y: A 1-dimensional numpy array or python list of labels, response variable.
        Returns
        -------
        self: The fitted model object.
        """

        self.svc = self.svc.fit(X, y)
        return self


    def probability(self, X):
        """Make probability predictions on new data."""

        return self.svc.predict_proba(X)


    def predict_label(self, X):
        """Make predictions of label from new data."""

        return self.svc.predict(X)



if __name__ == "__main__":
    data_path = "../../../downloads/class_exercise_01/exercise_01_train.csv"
    df = load_data(data_path)
    y = df.pop("y")
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=1)

    # Resampling the dataset by oversampling minority class
    sm = SMOTE(random_state=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scaling the the training set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_res)

    # Fit the data to the model
    model = SVC_Classification()
    model.fit(X_scaled, y_train_res)

    # Save the fitted model
    with open("models/svc_model.pkl", "wb") as m:
        pickle.dump(model, m)

    # Save the fitted scaler
    with open("models/svc_scaler.pkl", "wb") as s:
        pickle.dump(scaler, s)


    with open("models/svc_model.pkl", "rb") as m:
        model = pickle.load(m)

    with open("models/svc_scaler.pkl", "rb") as s:
        scaler = pickle.load(s)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict_label(X_test_scaled)
    precision, recall, accuracy, matrix = metric(y_test, y_pred)
    print(matrix)
    print("Recall: {}".format(round(recall, 2)))
    print("Precision: {}".format(round(precision, 2)))