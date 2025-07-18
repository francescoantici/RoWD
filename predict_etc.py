import argparse
import pickle
import pandas as pd
import os
import time 

from ml_models.rf import RF

if __name__ == "__main__":

    # Add parser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job-features", type=str, required=True, help="The job features to predict the label for")
    parser.add_argument("-m", "--model-dir", type=str, default=".", required = False, help="Path to the saved model")
    parser.add_argument("-l", "--log-file", type=str, default="rowd_logs.log", required = False, help="Path to the RoWD log")


    args = parser.parse_args()

    # Load the model from the file
    etc_classifier = RF(random_state=42)
    is_load_successfull = etc_classifier.load(os.path.join(args.model_dir, "stc_model.pkl"))
    if not is_load_successfull:
        print("Failed to load the model. Exiting.")
        exit(1)
    
    # Load the scaler
    try:
        scaler = pickle.load(open(os.path.join(args.model_dir, 'scaler.pkl'), 'rb'))
    except FileNotFoundError:
        print(f"Scaler file not found at {os.path.join(args.model_dir, 'scaler.pkl')}. Please check the path and try again.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the scaler: {e}")
        exit(1)
        
    # Compute the prediction
    job_features = scaler.transform([args.job_features])
    pred = etc_classifier.predict([job_features])

    # Log the prediction
    with open(args.log_file, 'a') as log_file:
        log_file.write(f"Execution Time Classifier\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Job features: {args.job_features} - Prediction: {pred[0]}\n")
    

    

    
    
    