# zenml-competition

# Setup

## One-time setup

1. Create a virtual environment (conda, pip, virtualenv, poetry)

2. Install requirements and pre-commit required for setting up the project.

    ```bash
    pip install setup-requirements.txt
    pre-commit install
    ```

    This will install different tools that we use for pre-commit hooks.

3. `pre-commit` hooks will run whenever we run `git commit -m` command. To skip some of the checks run

    ```bash
    SKIP=flake8 git commit -m "foo"
    ```

    To run pre-commit before commiting changes, run

    ```bash
    pre-commit run --all-files
    ```
