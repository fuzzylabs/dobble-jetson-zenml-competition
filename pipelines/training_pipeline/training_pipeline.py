"""Training pipeline."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def training_pipeline(
    download_data,
    create_data_loader,
    validate_data,
    trainer,
    # evaluator,
    validate_data_model,
    export_onnx,
):
    """Training pipeline.

    Steps
    1. download_data: This step downloads the data from the S3 bucket, bringing it into the pipeline
    2. create_data_loader: This step create a dataloaders for train, val and test datasets.
    3. validate_data: This step performs data integrity check on train, val and test datasets.
    4. trainer: This step trains a pytorch model using the datasets.
    5. evaluation:
    6. validate_data_model: This step performs data and model validation using trained model.
    7. export_onnx: Export trained pytorch model to onnx

    Args:
        download_data: This step downloads the data from the S3 bucket
        create_data_loader: This step create a dataloaders for train, val and test datasets
        validate_data: This step performs data integrity check on train, val and test datasets
        trainer: This step trains a pytorch model using the datasets
        validate_data_model: This step performs data and model validation using trained model
        export_onnx : Export trained pytorch model to onnx
    """
    # Specify the order - we need the data to be downloaded before creating
    # dataloaders from it
    create_data_loader.after(download_data)
    trainer.after(validate_data)
    validate_data_model.after(trainer)

    # Download the data from the S3 bucket
    download_data()

    # Create train, val and test dataloaders
    train_loader, val_loader, test_loader, classes = create_data_loader()

    # Run deepchecks on the datasets
    validate_data(train_loader, val_loader, test_loader, classes)

    # Train the model
    model = trainer(
        train_loader=train_loader, val_loader=val_loader, classes=classes
    )

    # Evaluate the model
    # evaluator(model=model, test_loader=test_loader)

    # Validate data and model
    validate_data_model(train_loader, test_loader, model, classes)

    # Export trained pytorch model to onnx
    export_onnx(model=model)
