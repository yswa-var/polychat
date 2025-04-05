import json
import random


def json_init():
    """Initialize the JSON file if it doesn't exist."""
    try:
        with open("data.json", "r") as f:
            content = f.read()
            if not content:  # Check if the file is empty
                data = {"main_prompt": None, "questions": []}
                with open("data.json", "w") as f:
                    json.dump(data, f)
            else:
                data = json.loads(content)
    except FileNotFoundError:
        data = {"main_prompt": None, "questions": []}
        with open("data.json", "w") as f:
            json.dump(data, f)
    return data


def save_user_prompt(prompt):
    """Save the user prompt to the JSON file."""
    data = json_init()
    data["main_prompt"] = prompt
    with open("data.json", "w") as f:
        json.dump(data, f)
    return True


def save_ques(question, type):
    """Save the question and its type to the JSON file."""
    data = json_init()
    # add answer null for initial
    data["questions"].append(
        {"question": question, "type": type, "answer": None})
    with open("data.json", "w") as f:
        json.dump(data, f)
    return True


def get_ques():
    """Retrieve all questions from the JSON file."""
    data = json_init()
    questions = data.get("questions", [])
    return questions


def get_ques_count():
    """Retrieve the count of questions from the JSON file."""
    data = json_init()
    questions = data.get("questions", [])
    return len(questions)


def save_answers(question_index, answer):
    """Save the answers to the JSON file."""
    data = json_init()
    if 0 <= question_index < len(data["questions"]):
        data["questions"][question_index]["answer"] = answer
        with open("data.json", "w") as f:
            json.dump(data, f)
            return True
    else:
        return False
