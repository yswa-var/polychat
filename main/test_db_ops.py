# Test Script
import json
from db_ops import get_ques, json_init, save_answers, save_ques, save_user_prompt


def test_data_saving():
    """Test the data saving functions."""

    # Clear existing data (for clean test)
    with open("data.json", "w") as f:
        json.dump({"main_prompt": None, "questions": []}, f)

    # Test save_user_prompt
    user_prompt = "What is the capital of France?"
    save_user_prompt(user_prompt)
    data = json_init()
    assert data["main_prompt"] == user_prompt, "User prompt not saved correctly."
    print("User prompt saved correctly.")

    # Test save_ques
    questions = [
        ("What is the capital of Japan?", "geography"),
        ("What is 2 + 2?", "math"),
        ("Who wrote Hamlet?", "literature"),
    ]
    for question, q_type in questions:
        save_ques(question, q_type)
    data = json_init()
    assert len(data["questions"]) == len(
        questions), "Questions not saved correctly."
    print("Questions saved correctly.")

    # Test save_answers
    answers = ["Tokyo", "4", "Shakespeare"]
    for i, ans in enumerate(answers):
        save_answers(i, ans)

    data = json_init()
    for i in range(len(answers)):
        assert data["questions"][i]["answer"] == answers[
            i], f"Answer for question {i} not saved correctly"
    print("Answers saved correctly")

    print("All tests passed!")


# Run the test
test_data_saving()

# Test get_ques


def test_get_ques():
    data = json_init()
    questions = get_ques()
    assert questions == data["questions"], "get_ques does not retrieve correct data"
    print("get_ques works correctly")


test_get_ques()
