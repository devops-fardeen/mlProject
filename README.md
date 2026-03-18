# 📈 S.I.F.E - Sensex Intelligent Forecasting Engine

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=for-the-badge&logo=flask)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css)

**S.I.F.E** is an advanced, full-stack Machine Learning web application designed to forecast the BSE Sensex closing price. Built by **Team Alpha**, this project leverages a Deep Learning Long Short-Term Memory (LSTM) neural network to analyze 60-day rolling windows of historical market momentum and predict future price movements.

## 🚀 Key Features

* **Deep Learning AI Brain:** Powered by a Univariate LSTM model trained on historical BSE Sensex (`^BSESN`) data, capturing complex sequential patterns.
* **Automated Data Pipeline:** Dynamically fetches and cleans live market data using the `yfinance` API, engineering technical indicators like 50-SMA, 200-SMA, and 14-day RSI.
* **Interactive Analytics Dashboard:** A responsive, modern frontend built with Tailwind CSS and Plotly.js for rendering massive datasets without lag.
* **Scenario Testing Module:** Allows users to input hypothetical market conditions (OHLCV) to run forward-testing and sensitivity analysis.
* **Dynamic UI/UX Engine:** Features an ultra-glossy iOS-inspired Glassmorphism design with a fully functional theme switcher (including a custom "Trainer" aesthetic mode).

## 📂 Project Structure

    📦 python_project
     ┣ 📂 model_data/               # Contains the trained LSTM (.keras) and Scalers (.pkl)
     ┣ 📂 project_graphs/           # Saved visual analytics of model performance
     ┣ 📂 templates/                # HTML/Tailwind frontend files (index.html)
     ┣ 📜 app.py                    # Main Flask backend server and API endpoints
     ┣ 📜 download_data.py          # Pipeline to fetch and clean ^BSESN data
     ┣ 📜 model_preprocessing.py    # Data scaling and 60-day window sequencing
     ┣ 📜 train_model.py            # LSTM architecture and training script
     ┣ 📜 model_evaluation.py       # Calculates R-Squared, MAPE, and Accuracy
     ┣ 📜 predict_tomorrow.py       # CLI script for quick terminal predictions
     ┣ 📜 requirements.txt          # Python dependencies
     ┗ 📜 README.md                 # Project documentation


## 🛠️ Installation & Setup

To run S.I.F.E locally on your machine, follow these steps:

**1. Clone the repository**
    git clone https://github.com/Mohammad-Altamash-zia/mlProject.git
    cd mlProject

**2. Install dependencies**
It is recommended to use a virtual environment. Install the required packages via pip:
    pip install -r requirements.txt
*(Note: This project specifically requires `tensorflow-cpu==2.20.0` to ensure stable deployment).*

**3. Run the Data Pipeline (Optional but recommended)**
To fetch the latest stock market data before running the app:
    python download_data.py

**4. Launch the Application**
Start the Flask backend server:
    python app.py

**5. Access the Dashboard**
Open your web browser and navigate to:
`http://127.0.0.1:5000`

## 🧠 Model Architecture
* **Type:** Long Short-Term Memory (LSTM) Recurrent Neural Network.
* **Input Shape:** `(1, 60, 1)` - The model analyzes bundles of 60 consecutive trading days.
* **Target Variable:** 'Close' Price.
* **Evaluation Metrics:** Evaluated using Mean Absolute Percentage Error (MAPE) to ensure functional real-world accuracy.

## 👥 Contributors (Team Alpha)
* **Mohammad Altamash Zia** - Lead Engineer
* **Mohammad Fardeen** - Frontend Developer
* **Mohammad Arshil** - Frontend Developer
* **Tayyab Farooq** - Data Engineer
* **Mohammad Saqlain** - Data Engineer
* **Harshita Saxena** - Machine Learning Engineer
* **Ayan** - Machine Learning Engineer