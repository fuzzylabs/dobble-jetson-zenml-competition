# ZenML recipe

We have used this fanatastic repo : [Open Source MLOps Stack Recipes](https://github.com/zenml-io/mlops-stacks) by ZenML. We have taken [aws-minimal](https://github.com/zenml-io/mlops-stacks/tree/main/aws-minimal) recipe as a starting point, and modified to fit our needs. The modification steps are listed below:

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
