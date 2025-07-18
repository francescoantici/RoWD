import pandas as pd 
import numpy as np 
import os 
import argparse

from script_preprocessors.script_preprocessor import script_preprocessor
from script_encoders.sbert_encoder import SBEncoding
from ml_models.lr import LR

if __name__ == "__main__":

    # Add parser for the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--scriptai-dataset", type=str, required=True, help="Path to the SCRIPT-AI dataset")
    parser.add_argument("-o", "--output-dir", type=str, default=".", required = False, help="Directory to save the output model")

    args = parser.parse_args()

    dataset_path = args.scriptai_dataset

    # Read feature files 
    df = pd.read_csv(os.path.join(dataset_path, "features.csv"))

    # Load the scripts 
    df["scripts"] = df.jid.apply(lambda jid: script_preprocessor(os.path.join(dataset_path, "job_scripts", jid)))

    # Initialize the SBERT encoder
    script_encoder = SBEncoding()

    # Encode the scripts 
    encodings = script_encoder.encode_data(df["scripts"].values)

    # Initialize the submission time classifier
    submission_time_classifier = LR(random_state=42)

    # Fit the classifier
    submission_time_classifier.fit(encodings, df["label"].values)

    # Save the model to a file
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, 'stc_model.pkl')
    submission_time_classifier.save(filename)






