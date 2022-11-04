# ZenML Recipe

The repo contains terraform configuration for resources for running ZenML pipelines on AWS

## What have we modified?
In this repository, we have used this fanatastic repo : [Open Source MLOps Stack Recipes](https://github.com/zenml-io/mlops-stacks) by ZenML. We have taken [aws-minimal](https://github.com/zenml-io/mlops-stacks/tree/main/aws-minimal) recipe as a starting point, and modified to fit our needs. The modification steps are listed below:

1. :cloud: Pull `aws-minimal` recipe that we will use for running ZenML pipelines on AWS.

    ~~~bash
    zenml stack recipe pull aws-minimal
    ~~~

    By default, this recipe will create following resources on AWS

    * An EKS cluster that can act as an [orchestrator](https://docs.zenml.io/mlops-stacks/orchestrators) for your workloads.
    * An S3 bucket as an [artifact store](https://docs.zenml.io/mlops-stacks/artifact-stores), which can be used to store all your ML artifacts like the model, checkpoints, etc.
    * An MLflow tracking server as an [experiment tracker](https://docs.zenml.io/mlops-stacks/experiment-trackers) which can be used for logging data while running your applications. It also has a beautiful UI that you can use to view everything in one place.
    * A Seldon Core deployment as a [model deployer](https://docs.zenml.io/mlops-stacks/model-deployers) to have your trained model deployed on a Kubernetes cluster to run inference on.
    * A [secrets manager](https://docs.zenml.io/mlops-stacks/secrets-managers) enabled for storing your secrets.

    This command will create a folder `zenml_stack_recipes/aws-minimal` that contains bunch of terraform files for above mentioned resources.

2. :scissors: We want to modify this stack to not use Seldon as model deployer. So, we delete Seldon Core specific resources. The files to be modified or deleted are

    * Delete `seldon` folder.
    * Delete `seldon.tf` file.
    * Inside `output_file.tf`, remove lines from 35 to 39.
    * Inside `outpus.tf`, remove line 46 to 57 that output seldon specific information.
    * Inside `get_URIs.tf`, remove line 12 to 21.
    * Inside `locals.ts`, remove line 17 to 21.

3. :art: Customize the deployment by editing the default values in the `locals.tf` file.

    * Change `prefix` from `prefix` to `dobble`.
    * Change `region` from `eu-west-1` to `eu-west-2`.

4. :tada: The resulting terraform configuration is committed to the Git repository.

## Setup

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

