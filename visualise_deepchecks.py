from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from zenml.post_execution import get_pipeline

def visualize_results(pipeline_name: str, step_name: str) -> None:
    pipeline = get_pipeline(pipeline=pipeline_name)
    last_run = pipeline.runs[-1]
    step = last_run.get_step(step=step_name)
    DeepchecksVisualizer().visualize(step)

if __name__ == "__main__":
    visualize_results("training_pipeline", "validate_data_model")