# Data-Collection-Management-Lab - Project

# 🏡 Airbnb Smart Matcher

**A Data-Driven, AI-Powered Recommendation Engine for Airbnb Listings.**

## 🚀 Overview
The **Airbnb Smart Matcher** is a sophisticated web application designed to help users find their ideal vacation rental. Unlike standard filters, this tool leverages **Big Data processing** and **Generative AI** to understand natural language requirements and match them with properties based on a multi-dimensional scoring system.

The project demonstrates an End-to-End Data Science pipeline: from data ingestion and cleaning using **PySpark** on **Databricks**, to a user-friendly frontend built with **Streamlit**.

## ✨ Key Features
* **Natural Language Search:** Users can describe their dream vacation (e.g., "A quiet place near the beach suitable for remote work") and GPT-4 translates it into technical query parameters.
* **Smart Partitioning:** optimized data retrieval using Parquet partitioning strategies to handle large datasets efficiently.
* **Similarity Matching:** Utilizes **KNN (K-Nearest Neighbors)** to find properties that mathematically resemble the user's ideal profile.
* **Dynamic Scoring:** Visualizes property metrics (Safety, Location, Host Quality) using color-coded indicators.
* **Interactive UI:** A Tinder-like "Swipe" interface to review, like, or pass on properties.

## 🛠️ Tech Stack
* **Infrastructure:** Databricks, Azure
* **Data Processing:** PySpark, Pandas
* **AI & ML:** OpenAI (GPT-4), Scikit-Learn (KNN, StandardScaler)
* **Frontend:** Streamlit
* **Storage:** Parquet (Partitioned by State)

## 📂 Project Structure
* `app.py`: The main application script containing the UI and logic.
* `notebooks/`: Contains the Data Engineering pipeline (Data ingestion, cleaning, and optimization).

## 🚀 How to Run
This application is designed to run within a **Databricks Environment** due to its dependency on the distributed file system (DBFS) and specific dataset paths.

1.  **Clone the Repository** to your Databricks Workspace or local machine.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Data:** Ensure the optimized Parquet data is available in the `/Workspace/.../app_data/` directory (see `notebooks/` for the generation script).
4.  **Configure Secrets:** Set your OpenAI API key in the environment variables or Streamlit secrets.
5.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 👨‍💻 Developers
Developed by **Ido Avital** & **Eliezer Mashihov**.
