"""
Loads all scripts in the examples folder and runs them to ensure they execute without error.
"""

import os


def test_all_examples():
    # Gather all .py files in the examples folder
    repository_path = os.path.dirname(os.path.dirname(__file__))
    examples_folder = os.path.join(repository_path, "examples")
    example_files = [
        os.path.join(examples_folder, f)
        for f in os.listdir(examples_folder)
        if f.endswith(".py")
    ]

    # Run each example file, capturing the exit code
    for example_file in example_files:
        print(f"Running {example_file}...")
        exit_code = os.system(f"python {example_file}")
        assert exit_code == 0
