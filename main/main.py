import json
import requests
import streamlit as st
from db_ops import save_ques, save_user_prompt


def classify_query(query):
    """Classifies a user request into one of the predefined types."""
    question_types = [
        "Fact-Based", "Definition", "Yes/No", "Multiple Choice", "Comparison",
        "Cause and Effect", "Hypothetical", "Planning", "Problem-Solving",
        "Opinion", "Analogical", "Logical", "Constraint"
    ]

    prompt = f"""
    From the following list of user request types, identify the most appropriate type for the given prompt. Provide your answer in a single word.

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

    user: {query}

    Answer:
    """

    messages = [{"role": "user", "content": prompt}]
    response = query_llm(messages, temp=0.3)
    try:
        classification = response["choices"][0]["message"]["content"].strip()
        return classification if classification in question_types else "Unknown"
    except (KeyError, IndexError):
        return "Error"


# Define system message
SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
st.title("🤖 Ploy-chat")
st.sidebar.title("Settings")

# Allow user to adjust max recursion depth
MAX_RECURSION_DEPTH = 3
max_depth = st.sidebar.slider(
    "Max question breakdown depth",
    min_value=1,
    max_value=5,
    value=MAX_RECURSION_DEPTH
)
MAX_RECURSION_DEPTH = max_depth

# Query function (streaming removed)


def query_llm(messages, temp=0.7):
    """Send a request to the LLM and return the response."""
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gemma-3-4b-it",
        "messages": messages,
        "temperature": temp,
        "max_tokens": -1
    }
    response = requests.post(url, headers=headers, json=data)
    try:
        return response.json()
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON response - {response.text}")
        return {"error": "Invalid JSON response"}


def get_answer(question, temp=0.3):
    """Get answer for a simple question."""
    messages = [SYSTEM_MESSAGE, {"role": "user", "content": question}]
    response = query_llm(messages, temp)
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "Error"


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Assume these functions are defined elsewhere
# save_ques(question, type): Saves a question with its type to data.json
# save_user_prompt(prompt): Saves the refined prompt as main_prompt in data.json
# classify_query(question): Returns the type of the question
# query_llm(messages, temp): Queries the LLM and returns a response
# get_answer(prompt, temp): Wrapper for LLM query

MAX_RECURSION_DEPTH = 3  # Example value; adjust as needed


def recursive_ques_gen(question, type, depth=0):
    """Recursively generate sub-questions based on the type and depth."""
    # Stop if the question is an atomic type
    if type in ["Fact-Based", "Definition", "Yes/No", "Multiple Choice"]:
        save_ques(question, type)
        return

    # Stop if maximum recursion depth is reached
    if depth >= MAX_RECURSION_DEPTH:
        save_ques(question, type)
        return

    # Improved prompt with clear instructions and an example
    prompt = (
        f"You are a researcher tasked with breaking down complex questions into simpler, more specific sub-questions. "
        f"Given the main question: '{question}', please generate 2-3 sub-questions that address different aspects or components "
        f"necessary to answer the main question. Each sub-question should be more focused and detailed than the main question. "
        f"For example, if the main question is 'How to plan a wedding?', sub-questions could be 'What is the budget for the wedding?', "
        f"'Who is on the guest list?', 'What venues are available?'. "
        f"Respond only with a JSON list of strings containing the sub-questions, like: [\"sub-question 1\", \"sub-question 2\", \"sub-question 3\"]."
    )
    messages = [{"role": "user", "content": prompt}]

    # Query the LLM
    response = query_llm(messages, temp=0.8)
    if "error" in response:
        print(f"Error from API: {response['error']}")
        save_ques(question, type)
        return

    # Extract and parse the LLM response
    sub_questions_str = response["choices"][0]["message"]["content"]
    print(
        f"Depth {depth} - Question: {question} | LLM Response: {sub_questions_str}")

    try:
        sub_questions_list = json.loads(sub_questions_str)
        print(f"Parsed sub-questions: {sub_questions_list}")
        # Check if the result is a non-empty list
        if isinstance(sub_questions_list, list) and sub_questions_list:
            for sub_question in sub_questions_list:
                # Ensure sub-question is a string
                if isinstance(sub_question, str):
                    new_type = classify_query(sub_question)
                    save_ques(sub_question, new_type)
                    # Skip recursion if sub-question is identical to the original
                    if sub_question.strip().lower() != question.strip().lower():
                        recursive_ques_gen(sub_question, new_type, depth + 1)
                    else:
                        print(
                            f"Sub-question identical to original: {sub_question} - Skipping recursion")
                else:
                    print(
                        f"Invalid sub-question type: {type(sub_question)} - Skipping")
        else:
            print(
                f"Response is not a valid non-empty list: {sub_questions_list}")
            save_ques(question, type)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {sub_questions_str}")
        save_ques(question, type)


if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        type = classify_query(prompt)
        st.write(f"Question Type: {type}")

        if type not in ["Fact-Based", "Definition", "Yes/No", "Multiple Choice"]:
            new_prompt = (
                f"refine this prompt. Determine the precise intent and underlying question of the user. "
                f"Then, create a refined version of the prompt that is clear, specific, and actionable. "
                f"Ensure only the refined prompt is returned in the response, nothing else is required. "
                f"User prompt: {prompt}"
            )
            response = get_answer(new_prompt, temp=0.5)
            if response == "Error":
                st.markdown("Error processing the prompt.")
            else:
                save_user_prompt(response)
                # Classify the refined prompt and start recursion with it
                refined_type = classify_query(response)
                recursive_ques_gen(response, refined_type)
        else:
            # If the original prompt is atomic, save it directly
            save_ques(prompt, type)

        # here we have a plan to

        # answer = get_answer(prompt, temp=0.8)
        # st.markdown(answer)

    # Save assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": answer})
