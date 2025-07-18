# RoWD

RoWD is the first security screening framework for the job submission workflow of HPC systems. The framework consists of several plugins, which act as detectors of different types of jobs (e.g. \textit{AI} jobs, cybercrime, cryptomining jobs, etc), and it is built to automatically and systematically scan the job submissions and find rogue ones.

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Initial Deploy**
    - Execute the `train_stc.py` file to generate a trained instance of the submission time classifier. To this end, the script requires the path to the [SCRIPT-AI dataset](https://github.com/francescoantici/SCRIPT-AI). Such script takes as input the SCRIPT-AI path as the argument `-d` and the output folder to store the saved model (`-o`).

3. **Run the framework:**
    - Once the training is performed, you can feed a job script to the `predict_stc.py` to perform a prediction with the submission time classifiers. Such script takes as input the job script as the argument `-s`, the saved model path as `-m`, and the log path as `-l`.
    - Once historical data of job execution is collected, run the `train_etc.py` to generate a trained instance of the execution time classifier. The script takes as input the dataset path (`-d`), the alpha parameter to decide how many data to use for the training (`-a`) and the output path to store the saved model (`-o`).
    - Then, infer on job execution data with the `predict_etc.py` file, which takes as input the list of the job features with the argument `-j`, and the log path `-l`.



