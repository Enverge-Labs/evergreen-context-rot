import re


def replace_non_alphanumeric(text):
    return re.sub(r"[^a-zA-Z0-9]", "_", text)
