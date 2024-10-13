"""
Wikipedia Question Answering System
This app uses OpenAI's gpt-4o-mini model to extract the main topic from a user's question 
and then searches Wikipedia for relevant information.
The user can provide feedback on the helpfulness of the information and view Q&A history.
"""
import re
import json
import sqlite3
import openai
import wikipedia
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page configuration
st.set_page_config(
    page_title="Wikipedia Question Answering System",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize OpenAI client
OPENAI_CLIENT = None

# Initialize the database to store queries and feedback
def initialize_db():
    """
    Initialize the SQLite database to store all queries and user feedback.
    The 'queries' table includes:
    - id: Unique identifier for each query.
    - question: The user's query.
    - answer: The system's response.
    - user_feedback: Feedback provided by the user ('yes', 'no', or NULL if no feedback).
    - feedback_submitted: A boolean flag indicating whether feedback was submitted.
    - timestamp: The time the query was submitted.
    """
    try:
        # Connect to the database and create the 'queries' table
        conn = sqlite3.connect('queries.db')
        cursor = conn.cursor()

        # Create the 'queries' table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                user_feedback TEXT, -- 'yes', 'no', or NULL
                feedback_submitted BOOLEAN DEFAULT 0, -- 0 means no feedback has been provided
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        # Display a success message
        st.write("✅ Database initialized successfully.")
    except Exception as e:
        st.error(f"❌ Error initializing the database: {e}")

# Initialize the database for storing queries and feedback
initialize_db()

# Create sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    # OpenAI API Key input
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ''
    api_key_input = st.text_input("Enter OpenAI API Key (optional):", type="password")

    # Save the API key to session state
    if api_key_input:
        st.session_state['openai_api_key'] = api_key_input
    else:
        st.info(
            "If no OpenAI API Key is provided, "
            "the system will use an open-source model for similarity calculations."
        )

    # Help section
    if st.button("Help"):
        st.info("""
        **How to use the Wikipedia Question Answering System:**

        1. (Optional) Enter your OpenAI API Key in the sidebar. If no key is provided, an open-source model will be used for similarity calculations.

        2. Type your question in the input box. This can be a general or specific query about any topic.

        3. Click 'Get Answer' to retrieve information from Wikipedia. The system will attempt to find the most relevant Wikipedia page based on your question.

        4. If the topic is ambiguous (e.g., the query "Java" can refer to a programming language or an island), the system will show you multiple options with relevance scores. Select the most relevant option from the dropdown menu.

        5. After selecting the topic, view the summary of the Wikipedia page. A link to the full Wikipedia page will also be provided for further reading.

        6. You can provide feedback on the helpfulness of the information by selecting 'Yes' or 'No'. Your feedback will help improve future results.

        7. The system maintains a Q&A history. You can view previous queries and answers, and export feedback data if needed.
        """)

# Initialize session state variables
if 'feedback_submitted' not in st.session_state:
    st.session_state['feedback_submitted'] = {}  # Tracks feedback submission status per query_id
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = ''  # Stores the selected option
if 'show_summary' not in st.session_state:
    st.session_state['show_summary'] = False  # Controls the display of the summary
if 'relevance_scores' not in st.session_state:
    st.session_state['relevance_scores'] = {}  # Stores relevance scores
if 'ranked_options_with_scores' not in st.session_state:
    st.session_state['ranked_options_with_scores'] = []  # Stores ranked options with scores
if 'ranked_options' not in st.session_state:
    st.session_state['ranked_options'] = []  # Stores ranked options
if 'qa_history' not in st.session_state:
    st.session_state['qa_history'] = []  # Stores Q&A history

# Initialize the SentenceTransformer model for sentence embeddings (if no OpenAI API Key provided, this model will be used)
if 'sentence_model' not in st.session_state:
    st.session_state['sentence_model'] = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Configure Wikipedia settings
wikipedia.set_lang('en')
wikipedia.set_user_agent('WikiQuestionAnsweringBot/1.0 (your_email@example.com)')

# Function to extract the main concept from the user's question using OpenAI
def get_concept_from_llm(question: str) -> str:
    """
    Extract the main topic from the user's question using OpenAI's language model.

    Parameters:
    - question: str, the user's question.

    Returns:
    - concept: str, the main topic extracted from the question.
    """
    prompt = f"""Extract the main topic from the user's question: \"{question}\". Provide the topic as a concise noun phrase. 
                But if the question is ambiguous, please provide the most likely topic. 
                If the question is too ambiguous to determine a specific topic, please provide a general topic.
                If the question is just one term, then just repeat the term."""
    try:
        with st.spinner('Extracting concept...'):
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            concept = response.choices[0].message.content.strip()
            return concept
    except openai.OpenAIError as e1:
        st.error(f"OpenAI API error: {e1}")
        return ""
    except Exception as e2:
        st.error(f"Unexpected error: {e2}")
        return ""

# Function to save the user's query and system's answer to the database
def save_query(question: str, answer: str) -> int:
    """
    Save the user's query and the system's answer to the database.
    Returns the unique ID of the inserted query.

    Parameters:
    - question: str, the user's question.
    - answer: str, the system's answer.

    Returns:
    - query_id: int, the unique ID of the inserted query.
    """
    try:
        conn = sqlite3.connect('queries.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO queries (question, answer, feedback_submitted)
            VALUES (?, ?, 0)
        ''', (question, answer))
        conn.commit()
        query_id = cursor.lastrowid  # Get the ID of the inserted row
        conn.close()
        return query_id
    except Exception as e:
        st.error(f"❌ Error saving query: {e}")
        return None

# Function to update feedback for a specific query using its ID
def update_feedback(query_id: int, feedback: str):
    """
    Update the feedback for a specific query in the database using its ID.

    Parameters:
    - query_id: int, the unique ID of the query.
    - feedback: str, the user's feedback ('yes' or 'no').
    """
    try:
        # Connect to the database and update the feedback for the given query ID
        conn = sqlite3.connect('queries.db')
        cursor = conn.cursor()

        # Update the user feedback and set the feedback_submitted flag to 1
        cursor.execute('''
            UPDATE queries 
            SET user_feedback = ?, feedback_submitted = 1 
            WHERE id = ?
        ''', (feedback, query_id))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        st.success("✅ Your feedback has been saved. Thank you!")
    except Exception as e:
        st.error(f"❌ Error updating feedback: {e}")

# # Function to display the Wikipedia page summary
# def display_summary(page_title: str):
#     """
#     Display the summary of the selected Wikipedia page.

#     Parameters:
#     - page_title: str, the title of the Wikipedia page.
#     """
#     try:
#         summary = get_page_summary_cached(page_title)
#         if summary:
#             st.write(f"### Information about '{page_title}':")
#             st.write(summary)
#             wiki_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
#             st.markdown(f"[🔗 Read more on Wikipedia]({wiki_url})")
#         else:
#             st.warning(f"⚠️ Sorry, the page '{page_title}' does not have an available summary.")
#     except Exception as e:
#         st.error(f"❌ Error displaying summary: {e}")

def display_summary(page_title: str, question: str):
    """
    Display an answer to the user's question using OpenAI's API (if available),
    followed by the Wikipedia summary.

    Parameters:
    - page_title: str, the title of the Wikipedia page.
    - question: str, the user's question.
    """
    try:
        summary = get_page_summary_cached(page_title)
        if summary:
            # Display the AI-generated answer if OpenAI API key is provided
            if st.session_state['openai_api_key']:
                # Generate an answer using OpenAI's API
                answer = generate_answer_with_openai(question, summary)
                if answer:
                    st.write(f"### Answer to your question:")
                    st.write(answer)
                else:
                    st.warning("⚠️ Unable to generate an answer at this time.")
            # Display the Wikipedia summary
            st.write(f"### Summary of '{page_title}':")
            st.write(summary)
            # Provide a link to the full Wikipedia page
            st.markdown(f"[🔗 Read more about '{page_title}' on Wikipedia](https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')})")
        else:
            st.warning(f"⚠️ Sorry, the page '{page_title}' does not have an available summary.")
    except Exception as e:
        st.error(f"❌ Error displaying summary: {e}")

def generate_answer_with_openai(question: str, summary: str) -> str:
    """
    Generate an answer to the user's question using OpenAI's API, based on the Wikipedia summary.

    Parameters:
    - question: str, the user's question.
    - summary: str, the summary of the Wikipedia page.

    Returns:
    - answer: str, the generated answer.
    """
    # Ensure OpenAI API key is provided
    if not st.session_state['openai_api_key']:
        st.error("OpenAI API key not provided.")
        return ""

    prompt = f"""You are a helpful assistant who uses the provided context to answer the question.

            Context:
            \"\"\"
            {summary}
            \"\"\"

            Question:
            \"\"\"
            {question}
            \"\"\"

            Answer the question based on the context above. If the answer is not in the context, politely say that you don't have enough information.
            """
    try:
        with st.spinner('Generating answer with OpenAI...'):
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()
            return answer
    except openai.OpenAIError as e1:
        st.error(f"OpenAI API error: {e1}")
        return ""
    except Exception as e2:
        st.error(f"Unexpected error: {e2}")
        return ""

# Function to get the cached Wikipedia page summary
@st.cache_data(show_spinner=False) # Cache the results to avoid repeated API calls
def get_page_summary_cached(page_title: str) -> str:
    """
    Get the summary of a Wikipedia page based on the page title.

    Parameters:
    - page_title: str, the title of the Wikipedia page.

    Returns:
    - summary: str, the summary of the Wikipedia page.
    """
    try:
        # Fetch the summary of the Wikipedia page
        summary = wikipedia.summary(page_title)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"Disambiguation error: {e}")
        return ""
    except wikipedia.exceptions.PageError:
        st.error(f"The page '{page_title}' does not exist.")
        return ""
    except Exception as e:
        st.error(f"Error fetching page summary: {e}")
        return ""

# Function to extract the title from the selected label
def extract_title(selected_label: str) -> str:
    """
    Extract the title from the selected label (with relevance score).
    e.g., "Java (Relevance: 0.85)" -> "Java"

    Parameters:
    - selected_label: str, the selected label with relevance score.

    Returns:
    - title: str, the extracted title.
    """
    return re.sub(r'\s*\(Relevance: [0-9.]+\)$', '', selected_label).strip()

# Collect feedback on the information provided
def collect_feedback(query_id: int):
    """
    Collect feedback from the user on the helpfulness of the information.
    The user can provide feedback by clicking 'Yes' or 'No' buttons.

    Parameters:
    - query_id: int, the unique ID of the query in the database.
    """
    # Initialize the feedback status for the current query
    if query_id not in st.session_state['feedback_submitted']:
        st.session_state['feedback_submitted'][query_id] = False

    # Display the feedback options if feedback has not been submitted
    if not st.session_state['feedback_submitted'][query_id]:
        st.write("Was this information helpful?")

        # Display 'Yes' and 'No' buttons for feedback
        col1, col2 = st.columns(2)
        with col1:
            # if 'yes' button is clicked, update feedback to 'yes' and set feedback_submitted to True
            if st.button("Yes", key=f'yes_button_{query_id}'):
                update_feedback(query_id, 'yes')
                st.session_state['feedback_submitted'][query_id] = True
        with col2:
            # if 'no' button is clicked, update feedback to 'no' and set feedback_submitted to True
            if st.button("No", key=f'no_button_{query_id}'):
                update_feedback(query_id, 'no')
                st.session_state['feedback_submitted'][query_id] = True

# Function to display precision, recall, and F1 score
def display_metrics():
    """
    This function calculates and displays the Precision, Recall, and F1 Score,
    as well as other related statistics such as total queries, helpful feedback,
    and unhelpful feedback. These metrics are based on user feedback data
    stored in the database.
    
    Precision measures the proportion of helpful feedback (positive feedback) 
    among all feedback (both positive and negative).
    
    Recall measures the proportion of helpful feedback out of all queries made.
    
    F1 Score is the harmonic mean of Precision and Recall, which provides a balance
    between them when both are important.
    """
    try:
        # Connect to the database to retrieve feedback data
        conn = sqlite3.connect('queries.db')
        cursor = conn.cursor()

        # Query to count helpful (positive) and unhelpful (negative) feedback,
        # and total number of queries submitted
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN user_feedback = 'yes' THEN 1 ELSE 0 END) AS helpful_feedback,
                SUM(CASE WHEN user_feedback = 'no' THEN 1 ELSE 0 END) AS unhelpful_feedback,
                COUNT(*) AS total_queries
            FROM queries
        ''')

        # Fetch the result of the query
        result = cursor.fetchone()
        conn.close()

        # Extract helpful and unhelpful feedback, and total queries from the query result
        helpful_feedback = result[0] if result[0] else 0  # Number of 'yes' feedbacks
        unhelpful_feedback = result[1] if result[1] else 0  # Number of 'no' feedbacks
        total_queries = result[2]  # Total number of queries submitted

        # Display the basic statistics
        st.write("## Accuracy Metrics:")
        st.write(f"- **Total Queries**: {total_queries}")  # Total number of queries
        st.write(f"- **Helpful Feedback**: {helpful_feedback}")  # Number of 'yes' feedback
        st.write(f"- **Unhelpful Feedback**: {unhelpful_feedback}")  # Number of 'no' feedback

        # Step 1: Calculate Precision (percentage of helpful feedback among all feedback)
        # Precision formula: helpful_feedback / (helpful_feedback + unhelpful_feedback)
        if helpful_feedback + unhelpful_feedback == 0:
            # If there's no feedback, precision cannot be calculated
            precision = 0
            st.write("- **Precision**: N/A (No valid feedback)")
        else:
            # Precision is calculated as a decimal, and multiplied by 100 to convert to percentage
            precision = helpful_feedback / (helpful_feedback + unhelpful_feedback)
            st.write(f"- **Precision**: {precision * 100:.2f}%")

        # Step 2: Calculate Recall (percentage of helpful feedback among all queries)
        # Recall formula: helpful_feedback / total_queries
        if total_queries == 0:
            # If there are no queries, recall cannot be calculated
            recall = 0
            st.write("- **Recall**: N/A (No query data)")
        else:
            # Recall is calculated as a decimal, and multiplied by 100 to convert to percentage
            recall = helpful_feedback / total_queries
            st.write(f"- **Recall**: {recall * 100:.2f}%")

        # Step 3: Calculate F1 Score (harmonic mean of precision and recall)
        # F1 Score formula: 2 * (precision * recall) / (precision + recall)
        if precision + recall == 0:
            # If both precision and recall are 0, F1 score cannot be calculated
            st.write("- **F1 Score**: N/A (Both Precision and Recall are 0)")
        else:
            # F1 Score is calculated using the precision and recall values as decimals
            f1_score = 2 * (precision * recall) / (precision + recall)
            st.write(f"- **F1 Score**: {f1_score:.2f}")

    except Exception as e:
        # Handle any potential database connection or query errors
        st.error(f"❌ Error calculating metrics: {e}")

# Function to calculate relevance scores using OpenAI
def calculate_relevance_batch_llm(question: str, options: list) -> dict:
    """
    Calculate relevance scores for a list of options based on the user's question using OpenAI's language model.

    Parameters:
    - question: str, the user's question.
    - options: list, a list of options to rate for relevance.

    Returns:
    - relevance_scores: dict, a dictionary of option-to-relevance score mapping.
    """
    # Format the question and options for the OpenAI API
    options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
    prompt = f"""Given the question: "{question}", rate the relevance of the following options on a scale from 0 to 1, 
    0 means not relevant and 1 means highly relevant:
    {options_text}
    Please return the results in JSON format, with each option number as the key and the relevance score as the value.
    Like this: {{1: 0.15, 2: 0.75, 3: 0.85}}"""

    try:
        with st.spinner('Calculating relevance scores...'):
            # Send the prompt to the OpenAI API to get relevance scores
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0
            )
            relevance_response = response.choices[0].message.content.strip()

            # Extract the JSON content from the response
            json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
            if json_match:
                relevance_json = json_match.group(0)
                relevance_scores = json.loads(relevance_json)

                # Map relevance scores to the actual options
                mapped_relevance_scores = {
                    options[int(key) - 1]: value for key, value in relevance_scores.items()
                }

                # Return the mapped scores with option text as the key
                return mapped_relevance_scores
            st.error("No JSON content found in the response.")
            return {}
    except openai.OpenAIError as e1:
        st.error(f"OpenAI API error: {e1}")
        return {}
    except Exception as e2:
        st.error(f"Unexpected error: {e2}")
        return {}

# Function to calculate similarity scores using SentenceTransformer
def calculate_similarity(question: str, options: list) -> dict:
    """
    Calculate the cosine similarity between the user's question and a list of options using the SentenceTransformer model.

    Parameters:
    - question: str, the user's question.
    - options: list, a list of options to calculate similarity with.

    Returns:
    - similarity_scores: dict, a dictionary of option-to-similarity score mapping.
    """
    try:
        with st.spinner('Calculating similarity scores...'):
            # Encode the question using the SentenceTransformer model
            question_embedding = st.session_state['sentence_model'].encode([question])
            # Encode the options using the SentenceTransformer model
            options_embeddings = st.session_state['sentence_model'].encode(options)
            # Calculate the cosine similarity between the question and options
            similarities = cosine_similarity(question_embedding, options_embeddings)[0]
            # Normalize the similarity scores to a range of [0, 1]
            similarities_normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-10)
            # Map the options to the normalized similarity scores
            similarity_scores = {option: float(similarity) for option, similarity in zip(options, similarities_normalized)}
            # Return the similarity scores
            return similarity_scores
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return {}

# Function to search Wikipedia based on a given query
def search_wikipedia(query: str) -> list:
    """
    Search Wikipedia for the given query and return the search results.

    Parameters:
    - query: str, the search query.

    Returns:
    - search_results: list, a list of search results.
    """
    try:
        with st.spinner('Searching Wikipedia...'):
            # Search Wikipedia for the given query
            search_results = wikipedia.search(query)
            return search_results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Main function
def main():
    """
    Main function for the Wikipedia Question Answering System app.
    """
    st.write("If the topic is ambiguous, you will be prompted to select the specific topic.")

    # Initialize session state variables
    if 'question' not in st.session_state:
        st.session_state['question'] = ''
    if 'concept' not in st.session_state:
        st.session_state['concept'] = ''
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = []
    if 'page_title' not in st.session_state:
        st.session_state['page_title'] = ''
    if 'summary' not in st.session_state:
        st.session_state['summary'] = ''
    if 'ambiguous' not in st.session_state:
        st.session_state['ambiguous'] = False
    if 'query_id' not in st.session_state:
        st.session_state['query_id'] = None

    # User input for the question
    question = st.text_input("❓ Enter your question:", value=st.session_state['question'])

    # Get answer button
    get_answer_clicked = st.button("🔍 Get Answer")

    # Process the user's question and get the answer
    if get_answer_clicked and question:
        # Reset session state variables
        st.session_state['question'] = question
        st.session_state['concept'] = ''
        st.session_state['search_results'] = []
        st.session_state['page_title'] = ''
        st.session_state['summary'] = ''
        st.session_state['show_summary'] = False
        st.session_state['selected_option'] = ''
        st.session_state['ambiguous'] = False
        st.session_state['relevance_scores'] = {}
        st.session_state['ranked_options_with_scores'] = []
        st.session_state['ranked_options'] = []
        st.session_state['qa_history'] = st.session_state.get('qa_history', [])
        st.session_state['query_id'] = None  # Reset query_id

        # Extract concept from the user's question
        if st.session_state['openai_api_key']:
            # if OpenAI API key is provided, use the language model to extract the concept
            concept = get_concept_from_llm(question)
        else:
            # if no OpenAI API key is provided, use the question as the concept
            concept = question

        if concept:
            st.session_state['concept'] = concept

            # Search Wikipedia for the concept
            search_results = search_wikipedia(concept)
            st.session_state['search_results'] = search_results

            if len(search_results) == 0:
                # If no search results are found, display a warning message
                st.warning("⚠️ No matching pages found on Wikipedia.")
            elif len(search_results) > 1:
                # If multiple search results are found, set the ambiguous flag to True
                st.session_state['ambiguous'] = True
            else:
                # If only one search result is found, display the summary
                st.session_state['page_title'] = search_results[0]
                st.session_state['show_summary'] = True
                st.session_state['qa_history'].append({
                    'question': question,
                    'answer': search_results[0]
                })
                # Save query and get query_id
                st.session_state['query_id'] = save_query(question, search_results[0])

    # Display the ambiguous options if the concept is ambiguous
    if st.session_state.get('ambiguous', False):
        st.write(f"🔄 The term '{st.session_state['concept']}' is ambiguous. Please select one of the following options:")

        # Calculate relevance scores if not already calculated
        if not st.session_state['relevance_scores']:
            # Calculate relevance scores using OpenAI API if the key is provided
            if st.session_state['openai_api_key']:
                relevance_scores = calculate_relevance_batch_llm(
                    st.session_state['question'],
                    st.session_state['search_results']
                )

            # Calculate relevance scores using SentenceTransformer model if no OpenAI API key is provided
            else:
                relevance_scores = calculate_similarity(
                    st.session_state['question'],
                    st.session_state['search_results']
                )

            st.session_state['relevance_scores'] = relevance_scores
        else:
            # Relevance scores are already calculated
            relevance_scores = st.session_state['relevance_scores']

        # Sort the search results based on relevance scores
        ranked_options = sorted(
            st.session_state['search_results'],
            key=lambda x: relevance_scores.get(x, 0),
            reverse=True
        )
        st.session_state['ranked_options'] = ranked_options

        # Display the ranked options with relevance scores
        ranked_options_with_scores = [
            f"{option} (Relevance: {relevance_scores.get(option, 0):.2f})"
            for option in ranked_options
        ]
        st.session_state['ranked_options_with_scores'] = ranked_options_with_scores

        # Set a select box to let the user choose the most relevant option
        selected_label = st.selectbox(
            "Select the topic:",
            ranked_options_with_scores
        )

        # Extract the selected option from the label
        selected_title = extract_title(selected_label)
        st.session_state['selected_option'] = selected_title

        # if the user selects an option, display the summary
        if st.button("✅ Show Answer"):
            st.session_state['page_title'] = selected_title
            st.session_state['show_summary'] = True

            # Add the question and answer to the Q&A history
            st.session_state['qa_history'].append({
                'question': st.session_state['question'],
                'answer': selected_title
            })

            # Save query and get query_id
            st.session_state['query_id'] = save_query(st.session_state['question'], selected_title)

    # if the user clicks the 'Show Summary' button, display the summary
    if st.session_state.get('show_summary', False):
        with st.spinner('📄 Fetching summary...'):
            # Display the summary of the selected page
            # display_summary(st.session_state['page_title'])
            # Display the answer and summary
            display_summary(st.session_state['page_title'], st.session_state['question'])

        # Collect feedback using query_id
        if st.session_state['query_id'] is not None:
            collect_feedback(st.session_state['query_id'])
        else:
            st.warning("⚠️ Cannot collect feedback because query_id was not found.")

    # Display metrics button
    if st.button("📊 Show Precision, Recall, and F1 Score Metrics"):
        display_metrics()

    # Display Q&A history
    if st.session_state['qa_history']:
        st.write("### Q&A History")
        for entry in st.session_state['qa_history']:
            st.write(f"**Q:** {entry['question']}")
            st.write(f"**A:** {entry['answer']}\n")

    # Export feedback data button
    if st.button("💾 Export Feedback Data"):
        try:
            # Connect to the database and fetch all data from the 'queries' table
            conn = sqlite3.connect('queries.db')
            cursor = conn.cursor()

            # Fetch all data from the 'queries' table
            cursor.execute('SELECT * FROM queries')
            data = cursor.fetchall()

            # Close the connection
            conn.close()

            # Check if there is data to export
            if data:
                # Create a DataFrame from the fetched data
                feedback_df = pd.DataFrame(data, columns=['ID', 'Question', 'Answer', 'User Feedback', 'Feedback Submitted', 'Timestamp'])
                # set a download button to export the feedback data as a CSV file
                st.download_button(
                    label="📥 Download Feedback as CSV",
                    data=feedback_df.to_csv(index=False),
                    file_name='feedback_data.csv',
                    mime='text/csv',
                )
            # If there is no data to export, show an info message
            else:
                st.info("ℹ️ No feedback data to export.")
        except Exception as e:
            st.error(f"❌ Error exporting feedback data: {e}")

if __name__ == "__main__":
    # Initialize the OpenAI client if an API key is provided
    if st.session_state['openai_api_key']:
        try:
            OPENAI_CLIENT = openai.OpenAI(api_key=st.session_state['openai_api_key'])
        except Exception as e:
            st.error(f"Error setting OpenAI API key: {e}")
    
    # Run the main function
    main()
