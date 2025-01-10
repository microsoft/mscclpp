# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
from pathlib import Path
import subprocess


def run_examples(input_folder, configs, output_folder):
    for config in configs:
        file_name = config["filename"]
        args = config["args"]

        input_file_path = Path(input_folder) / file_name
        # Strip the ".py" from the filename and add ".output"
        base_file_name = file_name[:-3] if file_name.endswith(".py") else file_name
        base_file_name = base_file_name.replace("/", "_")
        output_file_path = Path(output_folder) / f"{base_file_name}.output"

        # Construct the command to run the Python script
        command = ["python3", str(input_file_path)] + args

        # Run the command and capture output
        with open(output_file_path, "w") as output_file:
            result = subprocess.run(command, stdout=output_file, stderr=subprocess.STDOUT, text=True)

        # Optional: Check the return code to handle errors
        if result.returncode != 0:
            print(f"Error running {file_name}. See {output_file_path} for details.")


def main(input_folder, config_path, output_folder):
    with open(config_path, "r") as f:
        config = json.load(f)

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    run_examples(input_folder, config, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files according to a configuration and save the results.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the input files.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the processed files will be saved.")
    args = parser.parse_args()
    main(args.input_folder, args.config, args.output_folder)
