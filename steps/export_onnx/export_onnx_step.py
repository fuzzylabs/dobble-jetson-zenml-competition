"""Export PyTorch Model to ONNX."""
import mlflow

import torch
from torch import nn
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class ExportParameters(BaseParameters):
    """Export parameters."""

    # path to save onnx model
    onnx_model_path: str

    # image size to use for training (300 for fasterrcnn and 320 for ssdlite)
    image_size: int


@step
def export_onnx(params: ExportParameters, model: nn.Module) -> Output():
    """Export pytorch model to onnx step.

    Args:
        params (ExportParameters): parameters for exporting to onnx
        model (nn.Module) : PyTorch trained model
    """
    # export to ONNX
    input_names = ["input_0"]
    output_names = ["scores", "boxes"]
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, params.image_size, params.image_size).to(device)  # fmt: skip
    logger.info("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        params.onnx_model_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    
    # Log image size to use for training 
    mlflow.log_param("Image size", params.image_size)

    logger.info(f"Model exported to:  {params.onnx_model_path}")
