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

## ZenML Setup Local

Directory Structure

```bash
.
├── LICENSE
├── pyproject.toml
├── README.md
├── setup-requirements.txt
└── zenml-pipelines
    ├── pipelines                  # all pipelines inside this folder
    │   └── training_pipeline.py
    ├── requirements.txt           # dependencies required for zenml project
    ├── run.py                     # main file where all pipelines can be run
    └── steps                      # all steps inside this folder
        └── data_preprocess_step.py
        └── src                    # extra utilities that are required by steps added in this folder

```

1. Create a virtual environment (conda, pip, virtualenv, poetry). Recommened python version 3.8

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

    By default zenml comes with a stack that runs locally. We will add mlflow as experiment tracker to this local stack. We use this stack to run all tests locally.

    ```bash
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml stack register fuzzy_stack \
        -a default \
        -o default \
        -e mlflow_tracker
        --set
    ```

    Run ZenML pipelines.

    ```bash
    python3 run.py
    ```
