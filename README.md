# zenml-competition

# Setup

## One-time setup

1. Create a virtual environment (conda, pip, virtualenv, poetry). Recommened python version 3.8

2. Install requirements and pre-commit required for setting up the project.

    ```bash
    pip install -r setup-requirements.txt
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

    To check individual fails, run

    Interrogate pre-commit

    ```bash
    interrogate -c pyproject.toml --vv
    ```

    Flake8 pre-commit

    ```bash
    flake8 .
    ```

    Black pre-commit

    ```bash
    black . --config pyproject.toml --check
    ```

    pydocstyle pre-commit, list of [error code](https://www.pydocstyle.org/en/stable/error_codes.html)

    ```bash
    pydocstyle .  -e --count --convention=google --add-ignore=D403
    ```

    darglint pre-commit

    ```bash
    darglint -v 2 .
    ```

## ZenML Setup Local

Directory Structure

```bash
.
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt                 # dependencies required for zenml project
├── setup-requirements.txt           # dependencies required for precommit
└── zenml-pipelines
    ├── config_training_pipeline.yaml  # each pipeline will have one config file containing information regarding step and other configuration
    ├── pipelines                  # all pipelines inside this folder
    │   └── training_pipeline.py
    ├── run.py                     # main file where all pipelines can be run
    └── steps                      # all steps inside this folder
        └── data_preprocess_step.py
        └── src                    # extra utilities that are required by steps added in this folder

```

1. Use the same virtual environment created in above step(conda, pip, virtualenv, poetry).

2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```

3. Running ZenML Locally

    Install ZenML integrations required for project

    ```bash
    zenml integration install -y pytorch mlflow
    ```

    Initialize ZenML repo

    ```bash
    cd zenml-pipelines
    zenml init
    ```

    Start ZenServer

    ```bash
    zenml up   # start ZenServer
    ```

    > **Note**
    > Visit  ZenML dashboard is available at 'http://127.0.0.1:8237'. You can connect to it using the 'default' username and an empty password.

    By default zenml comes with a stack that runs locally. We will add mlflow as experiment tracker to this local stack. We use this stack to test pipelines locally.

    ```bash
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml stack register fuzzy_stack \
        -a default \
        -o default \
        -e mlflow_tracker \
        --set
    ```

    > **Note**
    > If there stack already exists by checking `zenml stack list`, activate the stack by running `zenml stack set fuzzy_stack`.

    Run ZenML pipelines.

    ```bash
    python3 run.py
    ```
