# Gradio Customer Churn App 💻
![Gradio x Huggingface](https://github.com/Azie88/Gradio-ML-classification-model-app-deployment/assets/101363399/9c5915fd-9ed8-41be-b725-50a8bc0e2548)

This project uses a trained machine learning model to build a web interface that allows anyone to use the classification model to predict whether a customer will churn(leave) or not churn (stay).

## Summary

| Code      | Name        | Deployed App |
|-----------|-------------|:-------------:|
| LP4 | Customer Churn Prediction with Gradio |  [Huggingface Space](https://huggingface.co/spaces/Azie88/Customer-Churn-Classification) |

## Introduction

After building a machine learning model, how do we enable other people to use it? We can't expect people to understand the code and interact with the model through a jupyter notebook or python file, so we must build a graphical user interface that is user friendly.

In this project, a classification ML model was embedded into a web app with [Gradio](https://gradio.app/). The user will interact with it by inputing the required information for the model to predict and classify if a customer will churn or not. This repo contains the project notebook showing the process of training the model.

## Process

- Research the documentation on Gradio

- Build a basic interface with inputs and outputs

- Import the model and the requirements(Scaler, Encoder, featurenengineering functions, etc.)

- Retrieve the input values and process them

- Pass the processed data through the ML model to predict

- Format the prediction output and send it to the interface to be displayed

## Setting up the Environment 🍀

For manual installation, you need to have `Python 3` on your system. Then you can clone this repo and be at the repo's root `https://github.com/Azie88/Gradio-ML-classification-model-app-deployment`, then follow the steps as outlined below;

1. Create a Python virtual environment to isolate the project's required libraries and avoid conflicts. Execute the following command in your terminal:

    ```bash
    python -m venv venv
    ```
 
2. Activate the Python virtual environment to use the isolated Python kernel and libraries. Run the appropriate command based on your operating system:

    - For Windows:

    ```bash
    venv\Scripts\activate
    ```
    - For Linux and MacOS:

    ```bash
    source venv/bin/activate
    ```

3. Upgrade Pip, the package manager, to ensure you have the latest version. Use the following command:

    ```bash
    python -m pip install --upgrade pip
    ```

4. Install the required libraries listed in the `requirements.txt` file. Run the command:

    ```bash
    python -m pip install -r requirements.txt
    ```

*Note: If you encounter any issues on MacOS, please make sure Xcode is installed.*

- Run the app.py file (being at the repository root):

  Gradio: 
  
    For development

      gradio app.py
    
    For normal deployment/execution

      python app.py  

  - Go to your browser at the following address :
        
      http://127.0.0.1:7860

## Screenshots 🖼️

![Gradio App Screenshot 1](https://github.com/Azie88/Gradio-ML-classification-model-app-deployment/assets/101363399/a07deca0-3cc9-411c-868f-7fc5245bea3c)

![Gradio App Screenshot 2](https://github.com/Azie88/Gradio-ML-classification-model-app-deployment/assets/101363399/1506b677-b446-4a31-90a4-cb0b2d3e2017)


## Author 🖊️

Andrew Obando

Connect with me on LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/andrewobando/)

---

Feel free to star ⭐ this repository if you find it helpful!
