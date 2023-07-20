# Disaster Response Pipeline Project

### Description:
- In times of crisis, fast and efficient aid distribution is crucial to support affected communities. This project addresses this challenge by leveraging machine learning techniques to classify disaster-related messages and efficiently direct aid to those who need it the most. The system takes in data (from Figure-Eight) of messages received during a disaster and uses natural language processing algorithms to analyze and categorize the messages into urgent needs categories, such as water, food, medical assistance, shelter, and other essential supplies. 

- By harnessing the power of machine learning and natural language processing, we aim to enhance the resilience and well-being of communities facing adversity during crises, ensuring that assistance reaches those in urgent need, efficiently and effectively.

 This project structure is as follows:
- **Disaster-messages-app**
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

### Quickstart:
https://drive.google.com/drive/u/1/folders/1Mm6ZywZ81L7kViC4oU9yFgHByln5R49C

1. Clone this repository.

2. Install the packages in requirements.txt using `pip install -r requirements.txt`.

3. Run `process_data.py` under the `data` folder to preprocess the data.

4. Run `train_classifier.py` under the `models` folder to build up the model.

5. Run `run.py` under the `app` folder to build up the model.

5. A link is displayed (with port 3000), clicking it should do it!

