# ZenML

Overall Diagram of ZenML pipelines here

## Data Pipeline

Detailed configuration of all steps in the data pipeline in [config_data_pipeline.yaml](config_data_pipeline.yaml)

1. `ingest_data` step : This step collects the data from Labelbox using the Labelbox API and requires the following parameters:
    * image_base_dir: Path to image directory containing images
    * labelbox_export_path: Path to labelbox export json file

You must also export the following variables:

```bash
# Labelbox API key
export LABELBOX_API_KEY=""
# Labelbox project ID
export LABELBOX_PROJECT_ID=""
```
As this is sensitive information it must be filled out with your own API key and project ID.

2. `prepare_labels` step : This step requires 3 parameters to convert the labels from labelbox json format to VOC format.
    * image_base_dir : The base directory of the images
    * labels_base_dir : The base directory to store the labels in VOC format
    * labelbox_export_path: The path to the labelbox export file
