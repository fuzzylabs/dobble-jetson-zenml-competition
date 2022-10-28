"""Training pipeline."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def training_pipeline(
    create_data_loader,
    trainer,
    # evaluator,
    # validate_data,
    # validate_model,
    export_onnx,
):
    """Training pipeline.

    Steps
    1. create_data_loader: This step create a dataloaders for train, val and test datasets.
    2. trainer: This step trains a pytorch model using the datasets.
    3. evaluation:
    4. validate_data:
    5. validate_model:
    6. export_onnx: Export trained pytorch model to onnx

    Args:
        create_data_loader: This step create a dataloaders for train, val and test datasets.
        trainer: This step trains a pytorch model using the datasets
        export_onnx : Export trained pytorch model to onnx
    """
    # Create train, val and test dataloaders
    train_loader, val_loader, test_loader, classes = create_data_loader()

    # Train the model
    model = trainer(
        train_loader=train_loader, val_loader=val_loader, classes=classes
    )

    # Evaluate the model
    # evaluator(model=model, test_loader=test_loader)

    #
    # validate_data()

    #
    # validate_model()

    # export trained pytorch model to onnx
    onnx_bytes = export_onnx(model=model)
