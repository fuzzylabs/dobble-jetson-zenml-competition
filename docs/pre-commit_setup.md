# Setting up pre-commits

This setup is required to configure pre-commit hooks.

1. The first step is creating a virtual environment and install the project requirements, we've used `conda` but feel free to use whatever you prefer (as long as you can install a set of requirements):

```bash
conda create -n dobble_venv python=3.8 -y
pip install -r requirements.txt
```

2. `pre-commit` hooks will run whenever we run `git commit -m` command. To skip some of the checks run

    ```bash
    SKIP=flake8 git commit -m "foo"
    ```

    To run pre-commit before commiting changes, run

    ```bash
    pre-commit run --all-files
    ```

    To check individual fails, run the following commands for particular pre-commit

    Interrogate pre-commit

    ```bash
    interrogate -c pyproject.toml -vv
    ```

    Flake8 pre-commit

    ```bash
    flake8 .
    ```

    isort pre-commit

    ```bash
    isort . --settings-path=pyproject.toml
    ```

    Black pre-commit

    ```bash
    black . --config pyproject.toml --check
    ```

    pydocstyle pre-commit, list of [error codes](https://www.pydocstyle.org/en/stable/error_codes.html)

    ```bash
    pydocstyle .  -e --count --convention=google --add-ignore=D403
    ```

    darglint pre-commit, list of [error codes](https://github.com/terrencepreilly/darglint#error-codes)

    ```bash
    darglint -v 2 .
    ```