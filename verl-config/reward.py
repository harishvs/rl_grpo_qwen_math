"""Custom reward function for GSM8K - verl compatible interface."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Binary reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    # Try to extract answer after ####
    match = re.search(r"####\s*(\-?[0-9\.\,]+)", solution_str)
    if match is None:
        return 0.0
    predicted = match.group(1).replace(",", "").strip()
    gt = str(ground_truth).replace(",", "").strip()
    try:
        return 1.0 if float(predicted) == float(gt) else 0.0
    except ValueError:
        return 1.0 if predicted == gt else 0.0
