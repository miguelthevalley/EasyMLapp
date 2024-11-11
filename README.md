EasyMLApp

This project is an interactive application built with Streamlit to perform Exploratory Data Analysis (EDA) and easily run different machine learning models. The tool allows you to upload CSV and XLSX files, visualize data, and test simple ML models.


Prerequisites

	•	Python 3.12 or higher
    # For MacOS users, make sure to install `libomp` before running the application: brew install libomp


Setup Instructions

	1.	Clone the repository:
git clone https://github.com/miguelthevalley/EasyMLApp.git
cd EasyMLapp

	2.	Create a virtual environment:
On Windows: python -m venv env
On macOS/Linux: python3 -m venv env

	3.	Activate the virtual environment:
On Windows: .\env\Scripts\activate
On macOS/Linux: source env/bin/activate

	4.	Install the dependencies:
pip install -r requirements.txt

	5.	Run the application:
streamlit run app.py

    6. Stop the application:
crtl+C


Features

	•	Upload CSV and XLSX files.
	•	Perform Exploratory Data Analysis (EDA).
	•	Run machine learning models such as Linear Regression and XGBoost.
	•	Interactive visualization with Plotly and Seaborn.


Author
Miguel Molina-Álvarez