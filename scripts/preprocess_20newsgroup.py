import os
import random
from math import ceil
from math import floor

import fire
import requests
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_URL = "http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-train-all-terms.txt"
TEST_URL = "http://ana.cachopo.org/datasets-for-single-label-text-categorization/20ng-test-all-terms.txt"

random.seed(RANDOM_STATE)


def main(output_dir: str, validation_size: float = 0.10) -> None:
    """Downloads and preprocesses the 20 Newsgroup dataset, saving it to disk at `output_dir`.

    Args:
        output_dir (str): Path to the directory to save the preprocessed data files.
        validation_size (float, optional): Fraction of examples to hold-out from the train set as a
            validation set. Defaults to 0.10.
    """
    processed_data = {partition: {"x": [], "y": []} for partition in ["train", "valid", "test"]}

    # Download the dataset and split it into individual examples
    twenty_ng_train = requests.get(TRAIN_URL).text.strip().split("\n")
    twenty_ng_test = requests.get(TEST_URL).text.strip().split("\n")

    # Accumulate the text (x) and the labels (y)
    for ex in twenty_ng_train:
        label, text = ex.split("\t")
        processed_data["train"]["x"].append(text)
        processed_data["train"]["y"].append(label)
    for ex in twenty_ng_test:
        label, text = ex.split("\t")
        processed_data["test"]["x"].append(text)
        processed_data["test"]["y"].append(label)

    # Create the validation split, stratify by class
    (
        processed_data["train"]["x"],
        processed_data["valid"]["x"],
        processed_data["train"]["y"],
        processed_data["valid"]["y"],
    ) = train_test_split(
        processed_data["train"]["x"],
        processed_data["train"]["y"],
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=processed_data["train"]["y"],
    )

    # Make sure the partition sizes check out
    assert (
        floor(len(twenty_ng_train) * (1 - validation_size))
        == len(processed_data["train"]["x"])
        == len(processed_data["train"]["y"])
    )
    assert (
        ceil(len(twenty_ng_train) * validation_size)
        == len(processed_data["valid"]["x"])
        == len(processed_data["valid"]["y"])
    )
    assert len(twenty_ng_test) == len(processed_data["test"]["x"]) == len(processed_data["test"]["y"])

    print("Processed the 20 Newsgroup dataset. Number of examples:")
    num_processed_examples = 0
    for partition, data in processed_data.items():
        n = len(data["x"])
        print(f"* {partition}: {n}")
        num_processed_examples += n
    print(f"* total: {num_processed_examples}")

    _write_to_disk(output_dir, processed_data)

    print(f"Preprocessed files saved to disk at: {os.path.abspath(output_dir)}")


def _write_to_disk(output_dir: str, processed_data: dict) -> None:
    for partition, data in processed_data.items():
        with open(os.path.join(output_dir, f"{partition}.txt"), "w") as f:
            f.write("\n".join(data["x"]))
        with open(os.path.join(output_dir, f"{partition}_labels.txt"), "w") as f:
            f.write("\n".join(data["y"]))


if __name__ == "__main__":
    fire.Fire(main)
