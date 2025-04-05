import streamlit as st
import requests
import json
import re
import os

# Set page config
st.set_page_config(page_title="LMStudio Chatbot", page_icon="🤖", layout="wide")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "streaming" not in st.session_state:
    st.session_state.streaming = True

# Define system message
SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}

# Define maximum recursion depth
MAX_RECURSION_DEPTH = 3


def save_to_json(prompt_id, question, response):
    """
    delete the json name "db.json" if it exists
    and create a new one with the same name
    and save the prompt_id, question and response
    """
    # Define the file name
    file_name = "db.json"

    # Delete the file if it exists
    if os.path.exists(file_name):
        os.remove(file_name)

    # Create a new JSON object
    data = {
        "prompt_id": prompt_id,
        "question": question,
        "response": response
    }

    # Write the JSON object to the file
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)


def append_to_json(prompt_id, question, response):
    """
    Append new question-answer pair to existing JSON file or create if not exists
    """
    # Define the file name
    file_name = "qa_data.json"

    # Load existing data or create new structure
    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
    else:
        data = {"qa_pairs": []}

    # Append new data
    data["qa_pairs"].append({
        "prompt_id": prompt_id,
        "question": question,
        "response": response
    })

    # Write the updated data to file
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_qa_data():
    """
    Load all question-answer pairs from JSON file
    """
    file_name = "qa_data.json"

    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            return json.load(json_file)
    else:
        return {"qa_pairs": []}


def load_ques_ans():
    """
    Load the question and answer from the JSON file
    and return it as a dictionary
    """
    # Define the file name
    file_name = "db.json"

    # Check if the file exists
    if os.path.exists(file_name):
        # Open the JSON file and load its content
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
            return data
    else:
        return None


def query_llm(messages, stream=True):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma-3-4b-it",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": stream
    }

    if stream:
        response = requests.post(url, headers=headers,
                                 data=json.dumps(data), stream=True)
        return response
    else:
        response = requests.post(url, headers=headers, json=data)
    return response.json()


def get_answer(question, streaming=False):
    """Get answer for a simple question"""
    messages = [SYSTEM_MESSAGE, {"role": "user", "content": question}]

    if streaming:
        # Handle streaming response case
        response_stream = query_llm(messages, stream=True)
        # We need to handle this in the UI so return the stream object
        return response_stream
    else:
        # For non-streaming, return just the text
        response = query_llm(messages, stream=False)
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError):
            return "Error getting response from LLM"


def classify_query(query):
    """Classifies a question into one of the predefined types."""

    question_types = [
        "Fact-Based", "Definition", "Yes/No", "Multiple Choice", "Comparison",
        "Cause and Effect", "Hypothetical", "Planning", "Problem-Solving",
        "Opinion", "Analogical", "Logical", "Constraint"
    ]

    prompt = f"""
    From the following list of question types, identify the most appropriate type for the given question. Provide your answer in a single word.

    Question Types:
    - Fact-Based
    - Definition
    - Yes/No
    - Multiple Choice
    - Comparison
    - Cause and Effect
    - Hypothetical
    - Planning
    - Problem-Solving
    - Opinion
    - Analogical
    - Logical
    - Constraint

    Question: {query}

    Answer:
    """

    messages = [{"role": "user", "content": prompt}]
    response = query_llm(messages, stream=False)

    try:
        classification = response["choices"][0]["message"]["content"].strip()
        if classification in question_types:
            return classification
        else:
            return "Unknown"  # Handle cases where the LLM doesn't classify correctly
    except (KeyError, IndexError, json.JSONDecodeError):
        return "Error"  # handle error case.


def generate_sub_questions(question, question_type):
    """Generates sub-questions for complex question types."""

    # Dictionary of prompts for different question types
    prompt_templates = {
        "Planning": f"""
        Given the following question, break it down into a list of smaller, more specific questions that need to be answered to create a comprehensive plan and response only a list of string of subquestion.

        Question: {question}

        Sub-questions:
        """,

        "Problem-Solving": f"""
        Given the following problem, break it down into a list of smaller, more specific questions that need to be answered to find a solution and response only a list of string of subquestion.

        Problem: {question}

        Sub-questions:
        """,

        "Comparison": f"""
        Given the following comparison question, break it down into a list of specific aspects that need to be compared and response only a list of string of subquestion.

        Comparison Question: {question}

        Comparison Aspects:
        """,

        "Cause and Effect": f"""
        Given the following cause and effect question, break it down into separate questions for causes and effects and response only a list of string of subquestion.

        Question: {question}

        Sub-questions:
        """,

        "Hypothetical": f"""
        Given the following hypothetical question, break it down into smaller questions that address the conditions and consequences and response only a list of string of subquestion.

        Question: {question}

        Sub-questions:
        """,

        "Logical": f"""
        Given the following Logical question, break it down into smaller questions that address the premises and conclusions and response only a list of string of subquestion.

        Question: {question}

        Sub-questions:
        """,

        "Constraint": f"""
        Given the following constraint question, break it down into smaller questions that address the constraints and the goal and response only a list of string of subquestion.

        Question: {question}

        Sub-questions:
        """
    }

    # Check if we have a template for this question type
    if question_type in prompt_templates:
        prompt = prompt_templates[question_type]
        messages = [{"role": "user", "content": prompt}]
        response = query_llm(messages, stream=False)
        try:
            sub_questions_text = response["choices"][0]["message"]["content"].strip(
            )
            sub_questions = [q.strip()
                             for q in sub_questions_text.split("\n") if q.strip()]

            # Filter out lines that aren't questions (e.g., numbering, explanatory text)
            # Keep only items that are likely questions (e.g., those ending with a question mark)
            # or those that start with numbers followed by a period (e.g., "1. What...")
            question_patterns = [
                r'.*\?$',  # Ends with question mark
                r'^\d+\.\s+.*',  # Starts with number + period
                r'^-\s+.*',  # Starts with dash
                r'^•\s+.*',  # Starts with bullet
                r'^What\s+.*',  # Starts with question words
                r'^How\s+.*',
                r'^Why\s+.*',
                r'^When\s+.*',
                r'^Where\s+.*',
                r'^Who\s+.*',
                r'^Which\s+.*',
                r'^Can\s+.*',
                r'^Should\s+.*',
                r'^Is\s+.*',
                r'^Are\s+.*',
                r'^Will\s+.*',
                r'^Could\s+.*',
                r'^Would\s+.*'
            ]

            # Clean up the sub-questions
            cleaned_questions = []
            for q in sub_questions:
                # Remove numbering and leading/trailing spaces
                cleaned_q = re.sub(r'^\d+[\.)]\s*', '', q).strip()
                cleaned_q = re.sub(r'^[-•]\s*', '', cleaned_q).strip()

                # Check if it matches any question pattern
                is_question = any(re.match(pattern, cleaned_q)
                                  for pattern in question_patterns)

                # Add it to cleaned questions if it's reasonably long and likely a question
                if len(cleaned_q) > 10 and (is_question or any(re.match(pattern, q) for pattern in question_patterns)):
                    cleaned_questions.append(cleaned_q)

            return cleaned_questions
        except (KeyError, IndexError, json.JSONDecodeError):
            return []  # Return empty list on error
    else:
        return []  # No sub-questions for other types


def recursive_question_breakdown(question, depth=0, qa_tree=None, prompt_id=1):
    """
    Recursively break down complex questions into simpler ones with depth limiting.
    Returns a tree structure of questions and answers.
    """
    # Initialize the QA tree at the top level
    if qa_tree is None:
        qa_tree = {"question": question, "type": None,
                   "answer": None, "subquestions": []}

    # Check if we've hit the recursion limit
    if depth >= MAX_RECURSION_DEPTH:
        # Get direct answer for the question without further breakdown
        answer = get_answer(question, streaming=False)
        qa_tree["answer"] = answer
        append_to_json(prompt_id, question, answer)
        return qa_tree, prompt_id + 1

    # Classify the question
    question_type = classify_query(question)
    qa_tree["type"] = question_type

    # For simple question types, get direct answer
    complex_types = ["Planning", "Problem-Solving", "Comparison", "Cause and Effect",
                     "Hypothetical", "Logical", "Constraint"]

    if question_type not in complex_types:
        # Simple question - get direct answer
        answer = get_answer(question, streaming=False)
        qa_tree["answer"] = answer
        append_to_json(prompt_id, question, answer)
        return qa_tree, prompt_id + 1

    # For complex questions, generate subquestions
    subquestions = generate_sub_questions(question, question_type)

    # If no subquestions were generated, treat as simple question
    if not subquestions:
        answer = get_answer(question, streaming=False)
        qa_tree["answer"] = answer
        append_to_json(prompt_id, question, answer)
        return qa_tree, prompt_id + 1

    # Process each subquestion recursively
    for subq in subquestions:
        subq_tree = {"question": subq, "type": None,
                     "answer": None, "subquestions": []}
        qa_tree["subquestions"].append(subq_tree)

        # Recursively process the subquestion
        subq_tree, prompt_id = recursive_question_breakdown(
            subq, depth + 1, subq_tree, prompt_id)

    # Once all subquestions are answered, synthesize a comprehensive answer for the original question
    if qa_tree["subquestions"]:
        qa_tree["answer"] = synthesize_answers(qa_tree)
        append_to_json(prompt_id, question, qa_tree["answer"])
        prompt_id += 1

    return qa_tree, prompt_id


def synthesize_answers(qa_tree):
    """
    Synthesize answers from subquestions into a comprehensive answer for the parent question.
    """
    original_question = qa_tree["question"]
    subq_data = []

    # Collect data from subquestions
    for subq in qa_tree["subquestions"]:
        if subq["answer"]:
            subq_data.append(f"Q: {subq['question']}\nA: {subq['answer']}")

    # If there are no subquestion answers, return a direct answer
    if not subq_data:
        return get_answer(original_question, streaming=False)

    # Create a prompt to synthesize answers
    synthesis_prompt = f"""
    Given the following question and its sub-questions with their answers, provide a comprehensive answer to the original question.
    
    Original Question: {original_question}
    
    Sub-questions and Answers:
    {"\n\n".join(subq_data)}
    
    Please synthesize a comprehensive answer to the original question, integrating information from all the sub-questions.
    """

    messages = [SYSTEM_MESSAGE, {"role": "user", "content": synthesis_prompt}]
    response = query_llm(messages, stream=False)

    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, json.JSONDecodeError):
        return "Error synthesizing answer. Please check your LMStudio connection."


def display_qa_tree(qa_tree, depth=0):
    """
    Display the question-answer tree in the UI
    """
    indent = "  " * depth

    # Display question and its type
    st.markdown(f"{indent}**Q:** {qa_tree['question']}")
    if qa_tree['type']:
        st.caption(f"{indent}Type: {qa_tree['type']}")

    # Display answer if available
    if qa_tree['answer']:
        with st.chat_message("assistant"):
            st.markdown(f"{qa_tree['answer']}")

    # Recursively display subquestions
    for subq in qa_tree['subquestions']:
        display_qa_tree(subq, depth + 1)


# App title and description
st.title("🤖 LMStudio Chatbot")
st.subheader("Chat with your local LLM using LMStudio")

# Toggle for streaming
st.sidebar.title("Settings")
streaming_enabled = st.sidebar.checkbox(
    "Enable streaming", value=st.session_state.streaming)
st.session_state.streaming = streaming_enabled

# Allow user to adjust max recursion depth
max_depth = st.sidebar.slider(
    "Max question breakdown depth",
    min_value=1,
    max_value=5,
    value=MAX_RECURSION_DEPTH
)
MAX_RECURSION_DEPTH = max_depth

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize prompt ID counter for the session if it doesn't exist
if "prompt_id" not in st.session_state:
    st.session_state.prompt_id = 1

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Clear previous QA data
    if os.path.exists("qa_data.json"):
        os.remove("qa_data.json")

    # Status indicator for processing
    with st.status("Processing your question...", expanded=True) as status:
        st.write("Breaking down the question...")
        # Show the current question being processed
        st.write(f"**Currently processing:** {prompt}")

        # Load previously answered questions
        qa_data = load_qa_data()
        if qa_data["qa_pairs"]:
            st.write("**Previously answered questions:**")
            for qa_pair in qa_data["qa_pairs"]:
                st.markdown(f"- **Q:** {qa_pair['question']}")

        # Process the question recursively
        qa_tree, new_prompt_id = recursive_question_breakdown(
            prompt,
            prompt_id=st.session_state.prompt_id
        )

        st.session_state.prompt_id = new_prompt_id

        st.write("Synthesizing final answer...")
        status.update(label="Analysis complete!", state="complete")

    # Display the final answer
    with st.chat_message("assistant"):
        if st.session_state.streaming:
            message_placeholder = st.empty()
            full_response = qa_tree["answer"]

            # Simulate streaming for the synthesized answer
            displayed = ""
            for i in range(len(full_response)):
                displayed += full_response[i]
                message_placeholder.markdown(displayed + "▌")
                if i % 3 == 0:  # Adjust speed
                    st.empty()  # This creates a small delay

            message_placeholder.markdown(full_response)
        else:
            st.markdown(qa_tree["answer"])

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": qa_tree["answer"]}
    )

    # Option to show the question breakdown
    if st.button("Show question breakdown"):
        st.subheader("Question Analysis")
        display_qa_tree(qa_tree)

# Add information about the app in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This chatbot connects to your local LMStudio instance running on port 1234.
- The app automatically classifies questions as simple or complex
- Complex questions are broken down recursively up to the specified depth
- Each subquestion gets analyzed and answered
- Final answers are synthesized from all subquestions
""")
