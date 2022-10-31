"""Training pipeline."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def training_pipeline(
    download_data,
    create_data_loader,
    trainer,
    # evaluator,
    # validate_data,
    # validate_model,
    export_onnx,
):
    """Training pipeline.

    Steps
    1. download_data: This step downloads the data from the S3 bucket, bringing it into the pipeline
    2. create_data_loader: This step create a dataloaders for train, val and test datasets.
    3. trainer: This step trains a pytorch model using the datasets.
    4. evaluation:
    5. validate_data:
    6. validate_model:
    7. export_onnx: Export trained pytorch model to onnx

    Args:
        download_data: This step downloads the data from the S3 bucket
        create_data_loader: This step create a dataloaders for train, val and test datasets.
        trainer: This step trains a pytorch model using the datasets
        export_onnx : Export trained pytorch model to onnx
    """
    # specify the order - we need the data to be downloaded before creating
    # dataloaders from it
    create_data_loader.after(download_data)

    # download the data from the S3 bucket
    download_data()

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
    export_onnx(model=model)
