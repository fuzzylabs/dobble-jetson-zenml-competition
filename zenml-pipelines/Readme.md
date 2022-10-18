# ZenML

Overall Diagram of ZenML pipelines here

## Data Pipeline

Detailed configuration of all steps in the data pipeline in [config_data_pipeline.yaml](config_data_pipeline.yaml)

1. `prepare_labels` step : This step requires 3 parameters to convert the labels from labelbox json format to VOC format.
    * image_base_dir : The base directory of the images
    * labels_base_dir : The base directory to store the labels in VOC format
    * labelbox_export_path: The path to the labelbox export file
