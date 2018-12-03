from GB_model import GB_Classification
from util import load_data
import numpy as np
import pickle



if __name__=="__main__":

    # Load in the test dataset
    test_data_path = "../../../downloads/class_exercise_01/exercise_01_test.csv"
    test_df = load_data(test_data_path)
    X_test = test_df.values

    # Loading in saved GB_Classification model
    with open("models/gb_model.pkl", "rb") as m:
        model = pickle.load(m)

    # Loading in saved fitted scaler
    with open("models/gb_scaler.pkl", "rb") as s:
        scaler = pickle.load(s)

    # Transform test data scaled form
    X_test_scaled = scaler.transform(X_test)

    # Predict probabilities
    y_pred = model.probability(X_test_scaled)

    # Save results to a text file
    np.savetxt("result/results1.csv", y_pred[:,1], delimiter=",")
    print(y_pred)