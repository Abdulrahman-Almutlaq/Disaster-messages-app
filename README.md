# Disaster Response Pipeline Project

### Description:
- The project is aimed to analyze disaster messages and classify them into the correct needs of the one in disaster. This project is divided into three main parts:
1. The `app` folder, where the backend (flask, which is run.py) and the frontend (go.html, master.html) exists.

2. The `data` folder, where the data exists in its two formats as csv files and as a database, and the script `process_data.py` that is used to preprocesses the data. `process_data.py` is based on the `ETL Pipeline Preparation.ipynb` notebook.

3. The `model` folder, where the best saved model exists in a pickle format, and the script used to train the model, which is based on the `ML Pipeline Preparation.ipynb` notebook.

- In order to start up the application you need to follow the following steps:

### Quickstart:

1. Clone this repository.

2. Install the packages in requirements.txt using `pip install -r requirements.txt`.

3. Run `process_data.py` under the `data` folder to preprocess the data.

4. Run `train_classifier.py` under the `models` folder to build up the model.

5. Run `run.py` under the `app` folder to build up the model.

5. A link is displayed (with port 3000), clicking it should do it!

