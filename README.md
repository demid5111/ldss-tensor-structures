[![Build](https://travis-ci.org/demid5111/ldss-tensor-structures.svg?branch=master)](https://travis-ci.org/demid5111/ldss-tensor-structures)
[![Coverage](https://coveralls.io/repos/github/demid5111/ldss-tensor-structures/badge.svg)](https://coveralls.io/github/demid5111/ldss-tensor-structures)

## Prerequisites

1. Python 3.6 and higher
2. Preferably Anaconda package

## Setting up

1. Install dependencies:
```
pip3 install -r requirements.txt
```

2. (Optionally]) Install graphviz:
    1. macOS:
        - `brew install graphviz`
        - `export PATH=/usr/local/bin/dot:$PATH`

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