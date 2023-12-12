# Vietnamese Text Classification API

The Vietnamese Text Classification API is a tool for profanity checking and topic classification. 

## Prerequisites

Before you proceed, make sure you have the following prerequisites in place:

1. **Python Version** : Ensure you have Python 3.8 installed on your system.
2. **AI Model File** :  The file is named `pytorch_model.bin`. Due to its large size, it is not suitable for pushing to this github. If you need it, please contact me via email.

## Setup Instructions

Follow these steps to set up and run the Data Moderation API:

1. **Move AI Model File** :

* Place `pytorch_model.bin` file into `deployment/models` .

2. **Install Required Libraries** :

* Install the necessary libraries by running:
  ```
  pip install -r requirements.txt
  ```

3. **Navigate to the Data-moderation-API Directory** :

* Change your working directory to the `src` folder:
  ```
  cd /deployment/src
  ```

4. **Run the Application** :

* Execute the `app.py` file:
  ```
  python3 app.py
  ```

## API Usage

This application provides two primary API endpoints:

1. **`/`**: This endpoint is used to verify the server's status. A successful response indicates that the server is operational.

2. **`/classify_text`**: This endpoint is used to classify the topic of a given text. Please note that the system will not classify topics if your texts that contain inappropriate words like 'đụ má', 'thằng chó'

For a more detailed description of these APIs, including request and response formats, please refer to the Swagger documentation available in the [openapi.json](./docs/openapi.json). file.