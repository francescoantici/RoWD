import pickle
import pandas as pd 
import numpy as np 
import os 
import argparse
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


from ml_models.rf import RF

if __name__ == "__main__":

    # Add parser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, required=True, help="Path to the job execution dataset")
    parser.add_argument("-a", "--alpha", type=int, default=15, required = False, help="The alpha parameter for the RF model")
    parser.add_argument("-o", "--output-dir", type=str, default=".", required = False, help="Directory to save the output model")

    # Define the feature of the jobs 
    features = ["econ","avgpcon","cnumat","elp","elpl","idle_time_ave","mbwidth","flops"]

    args = parser.parse_args()

    dataset_path = args.scriptai_dataset

    # Read feature files 
    df = pd.read_csv(os.path.join(dataset_path, "job_dataset.csv"))

    # Initialize the execution time classifier
    execution_time_classifier = RF(random_state=42)

    # Extract data 
    df_train = df[df.edt >= (datetime.now() - timedelta(args.alpha))]
    x_train = df_train[features].values
    y_train = df_train["label"].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_post = scaler.fit_transform(x_train)
    # Fit the classifier
    execution_time_classifier.fit(x_train, y_train)

    # Save the model to a file
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, 'etc_model.pkl')
    execution_time_classifier.save(filename)
    pickle.dump(scaler, open(os.path.join(out_dir, 'scaler.pkl'), 'wb'))






