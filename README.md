# Disaster Response Pipeline Project

### Description:
- In times of crisis, fast and efficient aid distribution is crucial to support affected communities. This project addresses this challenge by leveraging machine learning techniques to classify disaster-related messages and efficiently direct aid to those who need it the most. The system takes in data (from Figure-Eight) of messages received during a disaster and uses natural language processing algorithms to analyze and categorize the messages into urgent needs categories, such as water, food, medical assistance, shelter, and other essential supplies. 

- By harnessing the power of machine learning and natural language processing, we aim to enhance the resilience and well-being of communities facing adversity during crises, ensuring that assistance reaches those in urgent need, efficiently and effectively.

 This project structure is as follows:
- **Disaster-messages-app (Root Directory)**: The main directory of the project.
    - **app (Subdirectory)**: Contains the files related to the web application.
        - **templates (Subdirectory)**: Contains the HTML templates for rendering web pages.
            - **go.html**: The HTML template for displaying the results of message classification.
            - **master.html**: The HTML template for the main web page layout.
        - **run.py**: The Python script responsible for running the web application and handling user interactions.
    - **data (Subdirectory)**: Contains files related to data processing.
        - **process_data.py**: The Python script responsible for cleaning and preprocessing the disaster messages and categories data. It prepares the data for the machine learning model.
    - **models (Subdirectory)**: Contains files related to machine learning model training and evaluation.
        - **train_classifier.py**: The Python script responsible for training a machine learning model using the preprocessed data and saving the trained model for web application usage.
    - **ETL Pipeline Preparation.ipynb**: The notebook is used to complete the **process_data.py** file, as the original script was written in the notebook.
    - **ML Pipeline Preparation.ipynb**: The notebook is used to complete the **train_classifier.py** file, as the original script was written in the notebook.
    - **requirements.txt**: contains all the required packages for this repository.

### Quickstart (*one needs to run commands in the root Root Directory*):

1. Clone this repository using: *`git clone https://github.com/Abdulrahman-Almutlaq?tab=repositories`* 

2. Install the packages in requirements.txt using: *`pip install -r requirements.txt`*

3. Get the Figure-Eight data using the following commands:
    1. *`dvc init`*
    2. *`dvc pull`*
    3. If it does not work, here is a tutorial on [DVC](https://www.youtube.com/watch?v=kLKBcPonMYw&ab_channel=DVCorg), if that does not work, here is the [data](https://drive.google.com/drive/u/1/folders/1f3OMLLD_Erpzb08YK2b9u1YBi9AjucYk), it should be placed under the data folder.

4. Preprocess the data and create the database by running `process_data.py` under the `data` folder using: *`python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`*

5. Build up the model and save it by running `train_classifier.py` under the `models` folder using: *`python models/train_classifier.py data/DisasterResponse.db models/best_model.pkl`*

6. Start up the web  application by running `run.py` under the `app` folder using: *`python app/run.py`*

7. A link is displayed (with port 3000), clicking it should do it!

