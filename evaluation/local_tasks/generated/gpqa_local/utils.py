import random
import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = str(text).strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        if all(k in doc for k in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")):
            choices = [
                preprocess(doc["Incorrect Answer 1"]),
                preprocess(doc["Incorrect Answer 2"]),
                preprocess(doc["Incorrect Answer 3"]),
                preprocess(doc["Correct Answer"]),
            ]
            correct = preprocess(doc["Correct Answer"])
        elif isinstance(doc.get("choices"), list) and doc.get("answer") is not None:
            choices = [preprocess(x) for x in doc["choices"][:4]]
            ans = doc["answer"]
            if isinstance(ans, int):
                correct = choices[int(ans)]
            else:
                s = str(ans).strip().upper().strip("()")
                letters = ["A", "B", "C", "D"]
                if s not in letters:
                    raise ValueError(f"Unsupported GPQA local answer format: {ans!r}")
                correct = choices[letters.index(s)]
        else:
            raise ValueError(
                "Unsupported GPQA local row schema. Expected official GPQA fields or canonical choices/answer fields."
            )

        rng = random.Random(preprocess(doc.get("Question", doc.get("question", "")))[:200] + correct[:50])
        rng.shuffle(choices)
        correct_answer_index = choices.index(correct)

        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)
