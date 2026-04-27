# evaluate_results.py
# Scores the experiment results against reference answers across three metrics:
# - Answer accuracy
# - Hallucination rate
# - Document grounding
#
# Outputs a summary to the terminal and saves a scored JSON file for each model
# in the scored_results/ folder.

import json
import os
import re
import string

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SCORED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scored_results")


RESULT_FILES = [
    "tinyllama_results.json",
    "gemma_results.json",
    "qwen_results.json",
]

# Keywords that show the model refused to answer using retrieved context
# These are signs of over-reliance on context
REFUSAL_PHRASES = [
    "the passage does not",
    "the context does not",
    "i cannot answer this",
    "not provided in the context",
    "the document does not",
    "does not specify",
    "not mentioned in the",
]

# Keywords that show the model used the retrieved context in Condition B
GROUNDING_PHRASES = [
    "according to the document",
    "according to the context",
    "according to the passage",
    "based on the context",
    "the document states",
    "the passage states",
    "the context provides",
    "as stated in",
    "as mentioned in",
]


def normalise(text):
    # Lowercase, remove punctuation, collapse whitespace
    # Used for exact match comparison
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_exact_match(model_answer, reference_answer):
    # Returns True if the normalised reference appears anywhere in the normalised answer
    # This is more forgiving than a strict exact match since model answers are often longer
    norm_ref = normalise(reference_answer)
    norm_ans = normalise(model_answer)
    return norm_ref in norm_ans


def check_hallucination(model_answer, reference_answer):
    # Flags potential hallucinations by checking if the model answer contains
    # key terms from the reference answer. If it doesn't, it may have fabricated content
    # Returns True if a hallucination is suspected
    ref_words = set(normalise(reference_answer).split())
    ans_words = set(normalise(model_answer).split())

    # Filter out very common words that don't carry meaning
    stopwords = {"the", "a", "an", "is", "it", "of", "and", "or", "in", "to",
                 "was", "were", "are", "that", "this", "with", "for", "on",
                 "as", "by", "at", "be", "has", "had", "not", "from", "its"}
    ref_keywords = ref_words - stopwords

    if not ref_keywords:
        return False

    # If less than half the reference keywords appear in the answer,
    # flag it as a potential hallucination
    overlap = ref_keywords & ans_words
    overlap_ratio = len(overlap) / len(ref_keywords)
    return overlap_ratio < 0.5


def check_refusal(model_answer):
    # Returns True if the model refused to answer using context
    answer_lower = model_answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def check_grounding(model_answer):
    # Returns True if the model explicitly references retrieved context
    answer_lower = model_answer.lower()
    return any(phrase in answer_lower for phrase in GROUNDING_PHRASES)


def score_result(result):
    # Scores a single result entry and returns the entry with scores added
    answer = result["model_answer"]
    reference = result["reference_answer"]
    condition = result["condition"]

    exact_match = check_exact_match(answer, reference)
    hallucination_flag = check_hallucination(answer, reference)
    refusal = check_refusal(answer)

    # Accuracy: exact match passes automatically
    # If not exact match, flag for manual review
    if exact_match:
        accuracy = "correct"
    elif refusal:
        accuracy = "refused"
    else:
        accuracy = "review_needed"

    # Hallucination: only flag if not a refusal and not correct
    # Refusals are not hallucinations, they're a different failure mode
    if accuracy == "correct":
        hallucination = False
    elif refusal:
        hallucination = False
    else:
        hallucination = hallucination_flag

    # Document grounding: Condition B only
    if condition == "B":
        grounded = check_grounding(answer)
        grounding = "grounded" if grounded else "ungrounded"
    else:
        grounding = "n/a"

    result["scores"] = {
        "accuracy": accuracy,
        "hallucination": hallucination,
        "grounding": grounding,
        "exact_match": exact_match,
        "refusal": refusal,
        "manual_score": None,
    }

    return result


def summarise(results, model_name):
    # Prints a summary of scores for a model
    cond_a = [r for r in results if r["condition"] == "A"]
    cond_b = [r for r in results if r["condition"] == "B"]

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")

    for cond_label, cond_results in [("Condition A", cond_a), ("Condition B", cond_b)]:
        total = len(cond_results)
        correct = sum(1 for r in cond_results if r["scores"]["accuracy"] == "correct")
        review  = sum(1 for r in cond_results if r["scores"]["accuracy"] == "review_needed")
        refused = sum(1 for r in cond_results if r["scores"]["refusal"])
        hallucinated = sum(1 for r in cond_results if r["scores"]["hallucination"])

        print(f"\n  {cond_label} ({total} questions):")
        print(f"    Exact match (correct):  {correct}/{total} ({100*correct//total}%)")
        print(f"    Needs manual review:    {review}/{total} ({100*review//total}%)")
        print(f"    Refusals:               {refused}/{total} ({100*refused//total}%)")
        print(f"    Hallucination flags:    {hallucinated}/{total} ({100*hallucinated//total}%)")

        if cond_label == "Condition B":
            grounded = sum(1 for r in cond_results if r["scores"]["grounding"] == "grounded")
            print(f"    Document grounding:     {grounded}/{total} ({100*grounded//total}%)")

    # Print questions flagged for manual review
    flagged = [r for r in results if r["scores"]["accuracy"] == "review_needed"]
    if flagged:
        print(f"\n  Questions flagged for manual review:")
        for r in flagged:
            print(f"    {r['id']} [{r['condition']}]: {r['question'][:55]}...")
            print(f"      REF: {r['reference_answer'][:70]}")
            print(f"      ANS: {r['model_answer'][:70]}")
            print(f"      Hallucination flag: {r['scores']['hallucination']}")


def run():
    for filename in RESULT_FILES:
        filepath = os.path.join(RESULTS_DIR, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        # Score every result
        data["results"] = [score_result(r) for r in data["results"]]

        # Print summary
        summarise(data["results"], data["model"])

        # Save scored results back to a new file
        scored_filename = filename.replace("_results.json", "_scored.json")
        scored_filepath = os.path.join(SCORED_DIR, scored_filename)

        with open(scored_filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  Scored results saved to: {scored_filepath}")


    print("Evaluation complete.")
    print("Review flagged answers manually and update scores.")


if __name__ == "__main__":
    run()