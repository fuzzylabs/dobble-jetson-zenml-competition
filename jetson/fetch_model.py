from zenml.post_execution import get_pipeline
from zenml.enums import ExecutionStatus
import onnx


def fetch_onnx_model() -> bytes:
    print("Fetching the model bytes from the ZenML server...")
    pipeline = get_pipeline(pipeline="training_pipeline")
    completed_runs = [run for run in pipeline.runs if run.status == ExecutionStatus.COMPLETED]
    if len(completed_runs) == 0:
        raise Exception("No completed runs found on ZenML server")
    latest_run = completed_runs[-1]
    onnx_step = latest_run.get_step(step="export_onnx")
    onnx_bytes = onnx_step.output.read()
    return onnx_bytes


def load_model(onnx_bytes: bytes) -> onnx.ModelProto:
    print("Checking ONNX model...")
    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(e) # TODO proper error handling
        raise e

    return onnx_model


def fetch_onnx_from_zenml():
    onnx_bytes = fetch_onnx_model()
    onnx_model = load_model(onnx_bytes)

    print("Saving ONNX model to a file...")
    onnx.save_model(onnx_model, "model/dobble_model.onnx")
