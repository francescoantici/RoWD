import argparse
import pandas as pd
import os
import time 

from script_preprocessors.script_preprocessor import script_preprocessor
from script_encoders.sbert_encoder import SBEncoding
from ml_models.lr import LR

if __name__ == "__main__":

    # Add parser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--job-script", type=str, required=True, help="The job script to predict the label for")
    parser.add_argument("-m", "--model-dir", type=str, default=".", required = False, help="Path to the saved model")
    parser.add_argument("-l", "--log-file", type=str, default="rowd_logs.log", required = False, help="Path to the RoWD log")


    args = parser.parse_args()

    # Load the model from the file
    stc_classifier = LR(random_state=42)
    is_load_successfull = stc_classifier.load(os.path.join(args.model_dir, "stc_model.pkl"))
    if not is_load_successfull:
        print("Failed to load the model. Exiting.")
        exit(1)
    
    # Preprocess the job script
    job_script = script_preprocessor(args.job_script)
    
    # Encode the job script
    script_encoder = SBEncoding()
    encoded_script = script_encoder.encode_job(job_script)
    
    # Compute the prediction
    pred = stc_classifier.predict([encoded_script])

    # Log the prediction
    with open(args.log_file, 'a') as log_file:
        log_file.write(f"Submission Time Classifier\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Job script: {args.job_script} - Prediction: {pred[0]}\n")
    

    

    
    
    