LETTERS = list("ABCDEFGHIJ")


def doc_to_text(doc):
    lines = [
        "Choose the correct option. Reply with the option letter only.",
        "",
        f"Question: {doc['question']}",
    ]
    for index, choice in enumerate(doc["choices"]):
        lines.append(f"{LETTERS[index]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_choice(doc):
    return LETTERS[: len(doc["choices"])]


def doc_to_target(doc):
    return int(doc["gold_idx"])
