"""Prep GSM8K into verl parquet format (no verl dependency needed)."""
import argparse
import os
import re
import datasets


def extract_solution(solution_str):
    m = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert m is not None
    return m.group(1).replace(",", "")


def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")
        question = question_raw + ' Let\'s think step by step and output the final answer after "####".'
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx, "answer": answer_raw, "question": question_raw},
        }
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/gsm8k")
    args = parser.parse_args()

    ds = datasets.load_dataset("openai/gsm8k", "main")
    train = ds["train"].map(function=make_map_fn("train"), with_indices=True)
    test = ds["test"].map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
    print(f"Saved {len(train)} train, {len(test)} test to {args.local_save_dir}")
