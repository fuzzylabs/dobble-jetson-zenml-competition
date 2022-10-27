"""Training pipeline."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def training_pipeline(
    create_data_loader,
    # train,
    # evaluation,
    # validate_data,
    # validate_model,
    # convert_to_onnx
):
    """Training pipeline.

    Steps
    1. create_data_loader: This step create a dataloaders for train, val and test datasets.
    2. train: This step trains a Pytorch model using the train dataset.
    3. evaluation:
    4. validate_data:
    5. validate_model:
    6. convert_to_onnx:

    Args:
        create_data_loader: This step create a dataloaders for train, val and test datasets.
    """
    # Create train, val and test dataloaders
    train_loader, val_loader, test_loader = create_data_loader()

    # Train the model
    # train()

    # Model evaluation
    # evaliation()

    #
    # validate_data()

    #
    # validate_model()

    #
    # convert_to_onnx()
