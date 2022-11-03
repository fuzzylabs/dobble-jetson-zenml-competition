# get URI for  MLflow tracking server
data "kubernetes_service" "mlflow_tracking" {
  metadata {
    name      = "${module.mlflow.ingress-controller-name}-ingress-nginx-controller"
    namespace = module.mlflow.ingress-controller-namespace
  }
  depends_on = [
    module.mlflow
  ]
}
