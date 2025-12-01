import os
import sys
import json
from typing import Any
from dotenv import load_dotenv
from coolname import generate_slug
from io import StringIO
import traceback


class MLflowTracker:
    """
    MLflow experiment tracking helper for OpenTSLM curriculum learning.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        rank: int = 0,
        enabled: bool = True,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            rank: Process rank for distributed training (only rank 0 logs)
            enabled: Enable/disable tracking (useful for debugging)
        """
        self.experiment_name = experiment_name
        self.rank = rank
        self.enabled = enabled and (rank == 0)
        self.active_run = None

        # Stdout/stderr capture
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self.original_stdout = None
        self.original_stderr = None
        self.capturing = False

        # Only import and initialize MLflow on rank 0
        if self.enabled:
            try:
                import mlflow
                self.mlflow = mlflow

                # Load environment variables from .env file
                load_dotenv()

                # Set tracking URI (priority: parameter > env variable > default)
                if tracking_uri:
                    self.mlflow.set_tracking_uri(tracking_uri)
                elif os.getenv("MLFLOW_URL"):
                    self.mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
                else:
                    # Default to ./mlruns in project directory
                    self.mlflow.set_tracking_uri("file:./mlruns")

                # Set or create experiment
                self.mlflow.set_experiment(experiment_name)

                self.mlflow.enable_system_metrics_logging()

                print("✅ MLflow tracking initialized")
                print(f"   Experiment: {experiment_name}")
                print(f"   Tracking URI: {self.mlflow.get_tracking_uri()}")

            except ImportError:
                print("⚠️  MLflow not installed. Install with: pip install mlflow")
                print("   Tracking disabled.")
                self.enabled = False
                self.mlflow = None
        else:
            self.mlflow = None

    def start_stage_run(
        self,
        stage_name: str,
        params: dict[str, Any],
    ):
        """
        Start a run for a specific training stage.

        Args:
            stage_name: Name of the stage (e.g., "stage1_mcq")
            params: Dictionary of stage-specific parameters
        """
        if not self.enabled:
            return

        # End previous run if exists
        if self.active_run is not None:
            self.mlflow.end_run()

        # Generate slug and create run name
        slug = generate_slug(2)
        run_name = f"{stage_name}-{slug}"

        # Start new independent run for this stage
        self.active_run = self.mlflow.start_run(run_name=run_name)

        # Log all parameters
        self.mlflow.log_params(self._flatten_params(params))

        # Tag the stage
        self.mlflow.set_tag("stage", stage_name)

        print(f"📊 Started MLflow run: {run_name}")

    def log_params(self, params: dict[str, Any]):
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.enabled or self.active_run is None:
            return

        self.mlflow.log_params(self._flatten_params(params))

    def log_metric(self, key: str, value: float, step: int | None = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Training step/epoch (optional)
        """
        if not self.enabled or self.active_run is None:
            return

        # Verify we're logging to the correct run
        current_run = self.mlflow.active_run()
        if current_run and self.active_run:
            if current_run.info.run_id != self.active_run.info.run_id:
                print(f"[WARNING] Run ID mismatch! Expected {self.active_run.info.run_id}, but active run is {current_run.info.run_id}")

        self.mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch (optional)
        """
        if not self.enabled or self.active_run is None:
            return

        # Verify we're logging to the correct run
        current_run = self.mlflow.active_run()
        if current_run and self.active_run:
            if current_run.info.run_id != self.active_run.info.run_id:
                print(f"[WARNING] Run ID mismatch! Expected {self.active_run.info.run_id}, but active run is {current_run.info.run_id}")

        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """
        Log a single artifact file.

        Args:
            local_path: Path to the file to log
            artifact_path: Subdirectory in MLflow artifacts (optional)
        """
        if not self.enabled or self.active_run is None:
            return

        if os.path.exists(local_path):
            self.mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"⚠️  Artifact not found: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None):
        """
        Log all artifacts in a directory.

        Args:
            local_dir: Directory containing artifacts
            artifact_path: Subdirectory in MLflow artifacts (optional)
        """
        if not self.enabled or self.active_run is None:
            return

        if os.path.exists(local_dir):
            self.mlflow.log_artifacts(local_dir, artifact_path)
        else:
            print(f"⚠️  Artifact directory not found: {local_dir}")

    def log_stage_artifacts(self, stage_name: str, results_dir: str):
        """
        Log all standard artifacts for a training stage.

        Args:
            stage_name: Name of the stage
            results_dir: Base results directory (e.g., results/Llama3_2_1B/OpenTSLMSP)
        """
        if not self.enabled or self.active_run is None:
            return

        stage_dir = os.path.join(results_dir, stage_name)

        # Log checkpoint (best_model.pt)
        checkpoint_path = os.path.join(stage_dir, "checkpoints", "best_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"   Logging checkpoint: {checkpoint_path}")
            self.mlflow.log_artifact(checkpoint_path, "checkpoints")

        # Log loss history
        loss_history_path = os.path.join(stage_dir, "checkpoints", "loss_history.txt")
        if os.path.exists(loss_history_path):
            print(f"   Logging loss history: {loss_history_path}")
            self.mlflow.log_artifact(loss_history_path, "training")

        # Log test predictions
        predictions_path = os.path.join(stage_dir, "results", "test_predictions.jsonl")
        if os.path.exists(predictions_path):
            print(f"   Logging test predictions: {predictions_path}")
            self.mlflow.log_artifact(predictions_path, "results")

        # Log metrics JSON
        metrics_path = os.path.join(stage_dir, "results", "metrics.json")
        if os.path.exists(metrics_path):
            print(f"   Logging metrics: {metrics_path}")
            self.mlflow.log_artifact(metrics_path, "results")

            # Also log metrics as MLflow metrics for easy comparison
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.mlflow.log_metric(f"final_{key}", value)
            except Exception as e:
                print(f"⚠️  Could not log metrics from JSON: {e}")

    def log_model_summary(self, model):
        """
        Log model architecture summary.

        Args:
            model: PyTorch model
            input_sample: Sample input for model signature (optional)
        """
        if not self.enabled or self.active_run is None:
            return

        try:
            # Log model parameters count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.mlflow.log_param("total_parameters", total_params)
            self.mlflow.log_param("trainable_parameters", trainable_params)
            self.mlflow.log_param("frozen_parameters", total_params - trainable_params)

            print(f"   Logged model parameters: {trainable_params:,} trainable / {total_params:,} total")

        except Exception as e:
            print(f"⚠️  Could not log model summary: {e}")

    def start_stdout_capture(self):
        """Start capturing stdout and stderr."""
        if not self.enabled or self.capturing:
            return

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create tee-like objects that write to both original and buffer
        class TeeWriter:
            def __init__(self, original, buffer):
                self.original = original
                self.buffer = buffer

            def write(self, data):
                self.original.write(data)
                self.buffer.write(data)
                self.original.flush()

            def flush(self):
                self.original.flush()

        sys.stdout = TeeWriter(self.original_stdout, self.stdout_buffer)
        sys.stderr = TeeWriter(self.original_stderr, self.stderr_buffer)
        self.capturing = True

        print("Started capturing stdout/stderr for MLflow logging")

    def stop_stdout_capture(self):
        """Stop capturing stdout and stderr."""
        if not self.enabled or not self.capturing:
            return

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.capturing = False

        print("Stopped capturing stdout/stderr")

    def save_and_log_stdout(self, stage_name: str, results_dir: str):
        """
        Save captured stdout/stderr to files and log as artifacts.

        Args:
            stage_name: Name of the stage
            results_dir: Base results directory
        """
        if not self.enabled or self.active_run is None:
            return

        try:
            # Create logs directory
            stage_dir = os.path.join(results_dir, stage_name)
            logs_dir = os.path.join(stage_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            # Save stdout
            stdout_content = self.stdout_buffer.getvalue()
            if stdout_content:
                stdout_path = os.path.join(logs_dir, "stdout.log")
                with open(stdout_path, "w", encoding="utf-8") as f:
                    f.write(stdout_content)
                self.mlflow.log_artifact(stdout_path, "logs")
                print(f"   Logged stdout: {stdout_path}")

            # Save stderr
            stderr_content = self.stderr_buffer.getvalue()
            if stderr_content:
                stderr_path = os.path.join(logs_dir, "stderr.log")
                with open(stderr_path, "w", encoding="utf-8") as f:
                    f.write(stderr_content)
                self.mlflow.log_artifact(stderr_path, "logs")
                print(f"   Logged stderr: {stderr_path}")

            # Create combined log
            if stdout_content or stderr_content:
                combined_path = os.path.join(logs_dir, "combined.log")
                with open(combined_path, "w", encoding="utf-8") as f:
                    if stdout_content:
                        f.write("=" * 80 + "\n")
                        f.write("STDOUT\n")
                        f.write("=" * 80 + "\n")
                        f.write(stdout_content)
                        f.write("\n\n")
                    if stderr_content:
                        f.write("=" * 80 + "\n")
                        f.write("STDERR\n")
                        f.write("=" * 80 + "\n")
                        f.write(stderr_content)
                self.mlflow.log_artifact(combined_path, "logs")
                print(f"   Logged combined log: {combined_path}")

        except Exception as e:
            print(f"WARNING: Could not save stdout/stderr logs: {e}")
            traceback.print_exc()

    def end_stage_run(self, status: str = "FINISHED", stage_name: str = None, results_dir: str = None):
        """
        End the current stage run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
            stage_name: Name of the stage (for logging stdout)
            results_dir: Base results directory (for logging stdout)
        """
        if not self.enabled or self.active_run is None:
            return

        # Save and log stdout/stderr before ending the run
        if stage_name and results_dir:
            self.save_and_log_stdout(stage_name, results_dir)

        self.mlflow.end_run(status=status)
        print(f"📊 Ended MLflow run with status: {status}")
        self.active_run = None

        # Clear buffers for next run
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()


    def _flatten_params(self, params: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """
        Flatten nested parameter dictionaries for MLflow.

        Args:
            params: Dictionary of parameters (possibly nested)
            parent_key: Parent key for nested parameters

        Returns:
            Flattened dictionary with string keys and string/numeric values
        """
        flattened = {}

        for key, value in params.items():
            new_key = f"{parent_key}.{key}" if parent_key else key

            # Skip None values
            if value is None:
                continue

            # Recursively flatten dictionaries
            if isinstance(value, dict):
                flattened.update(self._flatten_params(value, new_key))
            # Convert lists to strings
            elif isinstance(value, (list, tuple)):
                flattened[new_key] = str(value)
            # Keep primitives as-is
            elif isinstance(value, (str, int, float, bool)):
                flattened[new_key] = value
            # Convert other types to strings
            else:
                flattened[new_key] = str(value)

        return flattened

    def set_tags(self, tags: dict[str, str]):
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tag name -> value
        """
        if not self.enabled or self.active_run is None:
            return

        self.mlflow.set_tags(tags)

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename to save the figure as
        """
        if not self.enabled or self.active_run is None:
            return

        self.mlflow.log_figure(figure, artifact_file)

    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self.enabled

    def get_run_id(self) -> str | None:
        """Get the current run ID."""
        if self.active_run:
            return self.active_run.info.run_id
        return None

    def get_experiment_id(self) -> str | None:
        """Get the current experiment ID."""
        if not self.enabled:
            return None

        experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id if experiment else None

    @staticmethod
    def create_from_trainer(
        trainer,
        tracking_uri: str | None = None,
        enabled: bool = True
    ):
        """
        Factory method to create tracker from CurriculumTrainer instance.

        Args:
            trainer: CurriculumTrainer instance
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            enabled: Enable/disable tracking

        Returns:
            MLflowTracker instance
        """
        experiment_name = f"OpenTSLM-{trainer.llm_id_safe}-{trainer.model_type}"

        return MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            rank=trainer.rank,
            enabled=enabled
        )
