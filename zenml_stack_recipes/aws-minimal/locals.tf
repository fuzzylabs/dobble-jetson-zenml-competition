# config values to use across the module
locals {
  prefix = "dobble"
  region = "eu-west-2"
  eks = {
    cluster_name = "zenml-terraform-cluster"
    # important to use 1.22 or above due to a bug with Istio in older versions
    cluster_version = "1.22"
  }
  vpc = {
    name = "zenml-vpc"
  }

  s3 = {
    name = "zenml-artifact-store"
  }

  mlflow = {
    artifact_Proxied_Access = "false"
    artifact_S3             = "true"
    # if not set, the bucket created as part of the deployment will be used
    artifact_S3_Bucket = ""
  }

  ecr = {
    name                      = "zenml-kubernetes"
    enable_container_registry = true
  }

  tags = {
    "managedBy"   = "terraform"
    "application" = local.prefix
  }
}
