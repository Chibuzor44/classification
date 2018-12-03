from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from util import load_data
from util import metric
import pickle
from imblearn.over_sampling import SMOTE


gb = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                              learning_rate=0.8, loss='deviance', max_depth=5,
                              max_features='log2', max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=600,
                              n_iter_no_change=None, presort='auto', random_state=None,
                              subsample=1.0, tol=0.0001, validation_fraction=0.1,
                              verbose=0, warm_start=False)


class GB_Classification():

    def __init__(self):
        self.gb = gb


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

        self.gb = self.gb.fit(X, y)
        return self


    def probability(self, X):
        """Make probability predictions on new data."""

        return self.gb.predict_proba(X)


    def predict_label(self, X):
        """Make predictions of label from new data."""

        return self.gb.predict(X)



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
    model = GB_Classification()
    model.fit(X_scaled, y_train_res)

    # Save the fitted model
    with open("models/gb_model.pkl", "wb") as m:
        pickle.dump(model, m)

    # Save the fitted scaler
    with open("models/gb_scaler.pkl", "wb") as s:
        pickle.dump(scaler, s)


    # with open("models/gb_model.pkl", "rb") as m:
    #     model = pickle.load(m)
    #
    # with open("models/gb_scaler.pkl", "rb") as s:
    #     scaler = pickle.load(s)
    #
    # X_test_scaled = scaler.transform(X_test)
    # y_pred = model.predict_label(X_test_scaled)
    # precision, recall, accuracy, matrix = metric(y_test, y_pred)
    # print(matrix)
    # print("Recall: {}".format(round(recall, 2)))
    # print("Precision: {}".format(round(precision, 2)))