[![Build](https://travis-ci.org/demid5111/ldss-tensor-structures.svg?branch=master)](https://travis-ci.org/demid5111/ldss-tensor-structures)
[![Coverage](https://coveralls.io/repos/github/demid5111/ldss-tensor-structures/badge.svg)](https://coveralls.io/github/demid5111/ldss-tensor-structures)

## Prerequisites

1. Python 3.6 and higher
2. Preferably Anaconda package

## Setting up

1. If you configure for the first time and do not have a `venv` folder, create virtual environment. 
   Instruction for Linux systems:

   ```bash
   python3 -m pip install --user virtualenv
   python3 -m virtualenv -p `which python3` venv
   ```

2. Activate environment:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optionally]) Install graphviz:
    1. macOS:
        - `brew install graphviz`
        - `export PATH=/usr/local/bin/dot:$PATH`
       
     
    ```bash
    pip install -r requirements_dev.txt
    ```

## Benchmarking the solution

1. Add `PROJECT_ROOT` to `PYTHONPATH`. Execute the following command from the project root:
   `export PYTHONPATH=$(pwd)`

2. Run the benchmarking script providing it with proper backend: `nn`, `numpy` or `scipy`.
   For example: `python demo/benchmark/benchmark_decoding.py --mode main --backend nn`

## Running tests

1. Without a coverage:
```
python3 -m unittest discover -p "*_test.py"
```

2. With coverage:
    * Run tests:
    ```
    coverage run --source=math_utils.py -m unittest discover -p "*_test.py"
    ```
    * Visualize coverage:
        - console:
        ```
        coverage report -m
        ```
        - html:
        ```
        coverage html
        open html_cov/index.html
        ```