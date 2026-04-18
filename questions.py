# questions.py
# Loads questions from questions.json and organises them into
# Condition A and Condition B lists for use in the experiment files.

import json
import os

# Load questions from JSON file

# Build path to questions.json relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "questions.json"), "r") as f:
    data = json.load(f)

# Extract each question category

solar_factual            = data["solar_factual"]
solar_conceptual         = data["solar_conceptual"]
solar_document_dependent = data["solar_document_dependent"]
roman_factual            = data["roman_factual"]
roman_conceptual         = data["roman_conceptual"]
roman_document_dependent = data["roman_document_dependent"]

# Build condition lists

# Condition A — factual and conceptual only, no document-dependent questions
CONDITION_A_QUESTIONS = (
    solar_factual +
    solar_conceptual +
    roman_factual +
    roman_conceptual
)

# Condition B — all 30 questions including document-dependent
CONDITION_B_QUESTIONS = (
    solar_factual +
    solar_conceptual +
    solar_document_dependent +
    roman_factual +
    roman_conceptual +
    roman_document_dependent
)