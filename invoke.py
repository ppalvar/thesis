import os
import argparse
import subprocess


"""
This script provides a command-line utility to run a specified Python script multiple times using the Python interpreter from a virtual environment located at '.env/Scripts/python.exe'.
It is primarily used to repeatedly execute experiment scripts, particularly as a workaround for stability issues with the Octave core, ensuring each run is isolated in a fresh process.
Arguments:
    script (str): The Python script to execute (e.g., experiments.py).
    -n, --num_runs (int, optional): Number of times to run the script (default: 100).
Example usage:
    python invoke.py experiments.py -n 50
# NOTE: This code is used to run the experiments multiple times because of stability issues with the Octave core.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Run a Python script multiple times using the venv interpreter."
    )
    parser.add_argument(
        "script", help="Python script to execute (e.g., experiments.py)"
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=100,
        help="Number of times to run the script (default: 100)",
    )
    args = parser.parse_args()

    # Full path to the venv's python executable
    venv_python = os.path.join(".env", "Scripts", "python.exe")

    for i in range(args.num_runs):
        try:
            subprocess.run([venv_python, args.script], check=True)
            print(f"Run {i + 1} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Run {i + 1} failed with exit code {e.returncode}.")


if __name__ == "__main__":
    main()
