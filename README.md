# Wikipedia Question Answering System

## Project Overview
This is a question-answering system based on the Wikipedia API. The system allows users to input questions and returns relevant information from Wikipedia. It handles ambiguous queries and improves accuracy over time through user feedback.

## Main Features
- **Wikipedia Querying**: Extract relevant information from Wikipedia.
- **Disambiguation Handling**: If multiple results are possible, users are prompted to select the relevant one.
- **User Feedback Mechanism**: Users can provide feedback on the answers.
- **Accuracy Metrics**: Based on feedback, the system calculates performance metrics (Precision, Recall, F1 Score).

## How to Run the Project
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

## Dependencies
- openai == 1.47.0
- wikipedia == 1.4.0
- sentence-transformers == 3.1.1
- scikit-learn == 1.5.2
- streamlit == 1.38.0
- pandas == 2.2.3