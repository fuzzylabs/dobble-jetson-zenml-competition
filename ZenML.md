# ZenML

Overall Diagram of ZenML pipelines here

## Data Pipeline

Detailed configuration of all steps in the data pipeline in [config_data_pipeline.yaml](pipelines/data_pipeline/config_data_pipeline.yaml).

1. `ingest_data` step : This step collects the data from Labelbox using the Labelbox API and requires 1 parameter.
    * image_base_dir: Path to image directory containing images

    You must also export the following variables:
    ```bash
    # Labelbox API key
    export LABELBOX_API_KEY=""
    # Labelbox project ID
    export LABELBOX_PROJECT_ID=""
    ```

    As this is sensitive information it must be filled out with your own API key and project ID.

2. `labelbox_tov_voc` step : This step requires 2 parameters to convert the labels from labelbox json format to VOC format.
    * image_base_dir : The base directory of the images
    * labels_base_dir : The base directory to store the labels in VOC format

3. `split_data` step: This step requires 3 parameters to split the dataset into train-val and test dataset.
    * image_base_dir : The base directory of the images
    * labels_base_dir : The base directory storing the labels in VOC format
    * train_test_split_ratio: A float in range [0, 1] specifying the split ratio to split dataset into train-val and test dataset

4. `upload_data` step: This step requires 3 parameters to upload everything in the data folder to a S3 bucket.
    * data_base_dir: Path to the data folder containing all outputs from the data pipeline
    * service_name: Name of the aws service (S3)
    * bucket: Name of the S3 bucket

    You must also export the following variables:
    ```bash
    # AWS access key id
    export ACCESS_KEY_ID=""
    # AWS secret access key
    export SECRET_ACCESS_KEY=""
    ```

## Training Pipeline

Detailed configuration of all steps in the training pipeline in [config_training_pipeline.yaml](pipelines/training_pipeline/config_training_pipeline.yaml).

1. `create_data_loader` step: This step requires 5 parameters to create 3 dataloaders for loading the training, validation and testing datasets.
    * dataset_base_dir : The base directory of images stored in VOC format and annotations
    * batch_size : How many samples per batch to load per iteration
    * use_aug : Whether to perform image augmentation on images to artificially expand the data-set
    * image_size : The size (width and height) of the image that will be resized to
    * num_worker : Number of workers for multi-process data loading
