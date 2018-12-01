from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import util
import pickle
from imblearn.over_sampling import SMOTE


rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                            max_depth=20, max_features=5, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,
                            warm_start=False)


class RF_Classification():

    def __init__(self):
        self.rf = rf


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

        self.rf = self.rf.fit(X, y)
        return self


    def probability(self, X):
        """Make probability predictions on new data."""

        return self.rf.predict_proba(X)


    def predict_label(self, X):
        """Make predictions of label from new data."""

        return self.rf.predict(X)


def load_data(path):
    df = pd.read_csv(path)
    util.regex(df, ["x41", "x45"])
    util.retructure_cols(df)
    df = pd.get_dummies(df, prefix=["x34","x35","x68","x93"])
    df = util.flter_columns(df)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    data_path = "../../../downloads/class_exercise_01/exercise_01_train.csv"
    df = load_data(data_path)
    y = df.pop("y")
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=1)

    sm = SMOTE(random_state=1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RF_Classification()
    model.fit(X_train_res, y_train_res)
    with open("model.pkl", "wb") as m:
        pickle.dump(model, m)


    with open("model.pkl", "rb") as m:
        model = pickle.load(m)
    y_pred_best = model.predict_label(X_test)
    precision_best, recall_best, accuracy_best, matrix_best = util.metric(y_test, y_pred_best)
    print(matrix_best)
    print("Recall: {}".format(round(recall_best, 2)))
    print("Precision: {}".format(round(precision_best, 2)))