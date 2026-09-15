"""
Testing file that checks that text preprocessing code works
"""

import pytest
import src.data_preprocessing.text_preprocessing as text_pre
import pandas as pd
@pytest.fixture
def qa_dataset():
    data= pd.DataFrame(
        {
            "question": [
                "What is the plant in this image?",
                "Is this plant sick?",
                "Does this tomato look healthy",
                None, #empty value
            ],
            "answer": [
                "It is a cucumber!",
                "No",
                "Yes, it is red and round.",
                None,
            ],
            "path": [
                "tmp.png", "tmp.png", "tmp.png", "tmp.png",
            ],
            "remove": [
                None, None, None, None,
            ],
        }
    )
    return data

def test_preprocesses_dataframe(qa_dataset):
    text_pre.preprocess_dataframe(
        qa_dataset,
        ["question", "answer"],
        ["remove"],
        {"question": "bababooey", "answer": "I fight cars"},
        ["path"],
        "test",
    )

    assert "remove" not in qa_dataset
    assert qa_dataset["question"].iloc()[-1] == "bababooey"
    assert qa_dataset["answer"].iloc()[-1] == "fight cars"
    assert qa_dataset["path"].iloc()[0] == "test/tmp.png"
    assert len(qa_dataset["question"]) == 4
    assert qa_dataset["answer"].iloc()[0] == "cucumber"


def test_remove_symbols():
    test_1= "Personally, I love  cats!"
    test_2= "./, a38, ^*#() [] {} 444"
    test_3= ""
    test_4= " "

    result_1= text_pre.remove_symbols(test_1)
    result_2= text_pre.remove_symbols(test_2)
    result_3= text_pre.remove_symbols(test_3)
    result_4= text_pre.remove_symbols(test_4)

    assert result_1 == "Personally I love  cats"
    assert result_2 == " a38    444"
    assert result_3 == ""
    assert result_4 == " "

def test_normalize_whitespace():
    test_1= ""
    test_2= " " # one space
    test_3= "  " # 2 spaces
    test_4= "   " # 1 tab
    test_5= "   Hello there !  How   are   you       doing?  "

    result_1= text_pre.normalize_whitespace(test_1)
    result_2= text_pre.normalize_whitespace(test_2)
    result_3= text_pre.normalize_whitespace(test_3)
    result_4= text_pre.normalize_whitespace(test_4)
    result_5= text_pre.normalize_whitespace(test_5)

    assert result_1 == ""
    assert result_2 == ""
    assert result_3 == ""
    assert result_4 == ""
    assert result_5 == "Hello there ! How are you doing?"

