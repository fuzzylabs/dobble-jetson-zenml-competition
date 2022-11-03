- [ZenML Competition](#zenml-competition)
- [Setup](#setup)
  - [One-time setup](#one-time-setup)
  - [ZenML local setup](#zenml-local-setup)
  - [ZenML mlops stack recipe](#zenml-mlops-stack-recipe)

# ZenML Competition

# Setup

## One-time setup

This setup is required to configure pre-commit hooks.

1. Create a virtual environment (conda, pip, virtualenv, poetry) and activate it. Recommened python version 3.8

2. Inside the virtual environment, install requirements and pre-commit required for setting up the project.

    ```bash
    pip install -r setup-requirements.txt
    pre-commit install
    ```

    Or with Conda

    ```bash
    conda install -c conda-forge pre-commit
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

## ZenML Local Setup

This setup is required to run ZenML pipelines locally.

Directory Structure

```bash
.
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt                 # dependencies required for zenml project
├── setup-requirements.txt           # dependencies required for precommit
├── pipelines                  # all pipelines inside this folder
│   └── training_pipeline
        └── training_pipeline.py
        └── config_training_pipeline.yaml  # each pipeline will have one config file containing information regarding step and other configuration
├── run.py                     # main file where all pipelines can be run
└── steps                      # all steps inside this folder
    └── data_preprocess
        └── data_preprocess_step.py
    └── src                    # extra utilities that are required by steps added in this folder

```

1. Use the same virtual environment created in above step(conda, pip, virtualenv, poetry).

2. Install requirements inside the already created environment

    ```bash
    pip install -r requirements.txt
    ```

3. Running ZenML Locally

    Install ZenML integrations required for the project

    ```bash
    zenml integration install -y pytorch mlflow
    ```

    Initialize ZenML repo

    ```bash
    zenml init
    ```

    Start ZenServer

    ```bash
    zenml up   # start ZenServer
    ```

    > **Note**
    > Visit  ZenML dashboard is available at 'http://127.0.0.1:8237'. You can connect to it using the 'default' username and an empty password.
    > If there's a TCP error about port not being available. Run `fuser -k port_no/tcp` to close an open port and run `zenml up` command again, for MacOS, run `kill $(lsof -t -i:8237)`.

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
    python run.py -dp      # run data pipeline only
    python run.py -tp      # run training pipeline only
    python run.py -dp -tp  # run both data and training pipeline
    ```

## ZenML MLOps Stack Recipe

The repo contains terraform configuration for resources for running ZenML pipelines on AWS. A detailed information of how this stack is created is outlined in [ZenML Recipe](ZenML_recipe.md).

Pre-requsities:

- [tfenv](https://github.com/tfutils/tfenv) and Terraform : `tfenv install && tfenv use`
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- Install zenml stack :  `pip install "zenml[stacks]"`
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

1. :zap: Configure AWS : Follow [this guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) to configure aws credentials using AWS CLI (`aws configure`).

    > **Note**
    > Set `Default region name` to `eu-west-2`.

2. :closed_lock_with_key: Add your secret information like keys and passwords into the `values.tfvars.json` file which is not committed and only exists locally.

    Create a file named `values.tfvars.json` under `zenml_stack_recipes/aws_minimal` directory with following contents.

    ```json
        {
            "mlflow-artifact-S3-access-key" : "<fill-me>",
            "mlflow-artifact-S3-secret-key" : "<fill-me>",
            "mlflow-username"               : "<fill-me>",
            "mlflow-password"               : "<fill-me>",
            "zenml-version"                 : "0.20.5"
        }
    ```

     - Add  `aws_access_key_id` to `mlflow-artifact-S3-access-key`.
     - Add `aws_secret_access_key` to  `mlflow-artifact-S3-secret-key`.
     - Add `mlflow-username` and `mlflow-password` as well.

    > **Note**
    > These secrets can be accessed by runnning `cat ~/.aws/credentials`.

3. :rocket: Deploy the recipe with this simple command.

    Navigate to directory containing terraform files.

    ```bash
    cd zenml_stack_recipes/aws_minimal
    ```

    ```bash
    terraform init
    terraform plan
    terraform apply
    terraform output
    ```

4. :hammer: Upon successfully provisioning all resources and making sure kubectl configured to eks cluster. `kubectl get namespaces` should contain `ingress-nginx`.

    To deploy ZenServer, create a file named `zen_server.tfvars.json` and fill in the content

    ```json
        {
            "name": "dobble",
            "provider": "aws",
            "username": "<fill-me>",
            "password": "<fill-me>",
            "create_ingress_controller": "false",
            "ingress_controller_hostname": "<fill-me>",
            "zenmlserver_image_tag": "0.20.5"
        }
    ```

    To get value of `ingress_controller_hostname`, run following

    ```bash
    # copy the LoadBalancer Ingress from the output of command
    kubectl describe svc nginx-controller-ingress-nginx-controller -n ingress-nginx
    ```
    
    Deploy the ZenServer using the command below.
    
     ```bash
    zenml deploy --config zen_server.tfvars.json
    ```
    
    After the server is created, you can visit the output url and login with the credentials supplied above to access ZenServer dashboard.

5. :page_with_curl: A ZenML stack configuration file (ex: `aws_minimal_stack_<something>.yaml`) gets created after the previous command executes :exploding_head:! This YAML file can be imported as a ZenML stack manually by running the following command.

    ```bash
    zenml stack import -f <path-to-the-created-stack-config-yaml> <stack-name>
    ```

    Run zenml pipelines using the above created stack, once it is set as active. Update the configuration files `.yaml` to use appropriate experiment_tracker, etc.

    ```bash
    zenml stack set <stack-name>
    ```

    After the stack is set active, we can run zenml pipelines using this stack.

6. :bomb: Delete the provisioned resources.

    ```bash
    terraform destroy
    ```

7. :sparkler: Destory ZenServer.

    ```bash
    zenml destroy
    ```

8. :broom: You can also remove all the downloaded recipe files from the pull execution by using the clean command.

    ```bash
    zenml stack recipe clean
    ```
