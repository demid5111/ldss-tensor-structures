[![Build Status](https://app.travis-ci.com/demid5111/ldss-tensor-structures.svg?branch=master)](https://app.travis-ci.com/demid5111/ldss-tensor-structures)

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

2. Activate virtual environment:

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
pytest
```

2. With coverage:
    * Run tests:
    ```
    pytest --cov-report html --cov core
    ```
    * Visualize coverage:
        - html:
        ```
        open htmlcov/index.html
        ```
## Running lines of code calculation

1. Run:
   ```bash
   pygount --format=summary ./core/
   ```
   
## Running PEP8 style check

1. Run:
   ```bash
   pylint ./core/
   ```


