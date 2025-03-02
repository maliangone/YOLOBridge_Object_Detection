import os
import logging
from datetime import datetime
import requests
import yaml
import zipfile
import shutil
import hashlib
from pathlib import Path
import json
from logging.handlers import RotatingFileHandler

from label_studio_ml.model import LabelStudioMLBase, SimpleJobManager
from label_studio_ml.utils import get_env, get_image_size, get_single_tag_keys
from label_studio_tools.core.utils.io import get_local_path
from ultralytics import YOLO
import ultralytics
from yolo_config import yolo_model_version, model_predict_config, model_train_config, fine_tune_start_from_original

# Configure logging
log_filename = f'yolo_model_{datetime.now().strftime("%Y%m%d")}.log'
max_bytes = 100 * 1024 * 1024  # 100MB
backup_count = 20  # Number of backup files to keep

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            filename=log_filename,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    ]
)

logger = logging.getLogger(__name__)

# Set up root directory
ROOT = os.path.join(os.path.dirname(__file__))
logger.info(f'ROOT directory set to: {ROOT}')

# Setup model registry directory
MODEL_REGISTRY_DIR = os.path.join(ROOT, 'model_registry')
if not os.path.exists(MODEL_REGISTRY_DIR):
    os.makedirs(MODEL_REGISTRY_DIR)
    logger.info(f'Created model registry directory: {MODEL_REGISTRY_DIR}')

# Model registry JSON file
MODEL_REGISTRY_FILE = os.path.join(MODEL_REGISTRY_DIR, 'model_registry.json')

# YOLO default training output path
YOLO_DEFAULT_OUTPUT_PATH = os.path.join(ROOT, 'runs/detect/train/weights/best.pt')

# Get Label Studio connection details from environment
HOSTNAME = get_env('LABEL_STUDIO_HOST') 
API_KEY = get_env('LABEL_STUDIO_API_KEY')

# Log connection info
logger.info(f'Label Studio hostname: {HOSTNAME}')
if not API_KEY:
    logger.warning('API_KEY is not set')

# Model status constants
MODEL_STATUS = {
    'INIT': 'initializing',
    'TRAINING': 'training_in_progress',
    'COMPLETE': 'training_completed',
    'FAILED': 'training_failed',
    'ARCHIVED': 'archived'
}

class ModelRegistry:
    """
    Manages model versions, metadata, and artifacts.
    
    This class provides a central registry for tracking all model versions,
    their training status, performance metrics, and file locations.
    It ensures consistent versioning and easy access to model information.
    """
    
    def __init__(self, registry_file=MODEL_REGISTRY_FILE):
        """Initialize the model registry"""
        self.registry_file = registry_file
        self._load_registry()
        
    def _load_registry(self):
        """Load the registry from file or create a new one"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": [],
                "current_model": None,
                "latest_version": 0
            }
            self._save_registry()
            
    def _save_registry(self):
        """Save the registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    def create_new_version(self, model_type=yolo_model_version):
        """Create a new model version entry"""
        self.registry["latest_version"] += 1
        version_id = self.registry["latest_version"]
        version_name = f"{model_type}_V{version_id}"
        
        # Create version directory
        version_dir = os.path.join(MODEL_REGISTRY_DIR, version_name)
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
            
        # Create model entry
        model_entry = {
            "id": version_id,
            "name": version_name,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "status": MODEL_STATUS['INIT'],
            "weights_file": os.path.join(version_dir, f"{version_name}_best.pt"),
            "metrics": {},
            "training_config": model_train_config.copy(),
            "last_updated": datetime.now().isoformat()
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        return model_entry
        
    def get_model_by_name(self, model_name):
        """Get a model entry by its name"""
        for model in self.registry["models"]:
            if model["name"] == model_name:
                return model
        return None
        
    def get_latest_completed_model(self):
        """Get the latest model with completed training"""
        for model in reversed(self.registry["models"]):
            if model["status"] == MODEL_STATUS['COMPLETE']:
                return model
        return None
        
    def update_model_status(self, model_name, status):
        """Update a model's status"""
        model = self.get_model_by_name(model_name)
        if model:
            model["status"] = status
            model["last_updated"] = datetime.now().isoformat()
            self._save_registry()
            return True
        return False
        
    def update_model_metrics(self, model_name, metrics):
        """Update a model's performance metrics"""
        model = self.get_model_by_name(model_name)
        if model:
            model["metrics"].update(metrics)
            model["last_updated"] = datetime.now().isoformat()
            self._save_registry()
            return True
        return False
        
    def set_current_model(self, model_name):
        """Set the current model for prediction"""
        model = self.get_model_by_name(model_name)
        if model and model["status"] == MODEL_STATUS['COMPLETE']:
            self.registry["current_model"] = model_name
            self._save_registry()
            return True
        return False
        
    def get_current_model(self):
        """Get the current model for prediction"""
        if self.registry["current_model"]:
            return self.get_model_by_name(self.registry["current_model"])
        # If no current model is set, use the latest completed model
        return self.get_latest_completed_model()

class MyModel(LabelStudioMLBase):
    """
    YOLO-based object detection model for Label Studio integration.
    
    This class implements a machine learning backend that handles object detection tasks
    using the YOLO (You Only Look Once) model. It provides functionality for both
    training (fit) and inference (predict) operations.

    The model supports:
    - Version control and status tracking
    - Training on labeled data from Label Studio
    - Object detection with bounding boxes and class labels
    - Automatic train/validation split
    - Model checkpointing and versioning

    Attributes:
        from_name (str): Source control tag from Label Studio config
        to_name (str): Target control tag from Label Studio config
        value (str): Value from Label Studio config
        labels_in_config (set): Set of label names configured in Label Studio
    """

    def __init__(self, **kwargs):
        """
        Initializes an instance of the MyModel class.

        Attributes:
            labels_in_config (set): The set of labels configured in the frontend.
        """
        super(MyModel, self).__init__(**kwargs)
        logger.info('Initializing MyModel instance')
        # Initialize model registry
        self.model_registry = ModelRegistry()
        
        # Get frontend configs
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        # Frontend label list
        self.labels_in_config = set(self.labels_in_config)
        logger.debug(f'Labels configured in frontend: {self.labels_in_config}')

    def get_model_version_for_fit(self):
        """
        Get or create a model version for training.
        
        Returns:
            tuple: (model_entry, status_message, can_train)
                - model_entry: Dictionary with model information
                - status_message: User-friendly status message
                - can_train: Boolean indicating if training can proceed
        """
        logger.info('Checking model version for training')
        
        # Check if any model is currently training
        for model in self.model_registry.registry["models"]:
            if model["status"] == MODEL_STATUS['TRAINING']:
                logger.info(f"Model {model['name']} is currently training")
                return model, f"Model {model['name']} is already training", False
        
        # Create a new model version
        model_entry = self.model_registry.create_new_version()
        logger.info(f"Created new model version: {model_entry['name']}")
        
        return model_entry, f"Beginning training for new model: {model_entry['name']}", True

    def get_model_for_predict(self):
        """
        Get the appropriate model for prediction.
        
        Returns:
            tuple: (model_entry, status_message, can_predict)
                - model_entry: Dictionary with model information or None
                - status_message: User-friendly status message
                - can_predict: Boolean indicating if prediction can proceed
        """
        logger.info('Checking model version for prediction')
        
        # Get current model from registry
        model_entry = self.model_registry.get_current_model()
        
        if not model_entry:
            logger.error('No trained model found')
            return None, "No trained model available. Please train a model first.", False
            
        if model_entry["status"] == MODEL_STATUS['TRAINING']:
            logger.warning(f"Model {model_entry['name']} is still training")
            return model_entry, f"Model {model_entry['name']} is still training. Please wait.", False
            
        if model_entry["status"] == MODEL_STATUS['COMPLETE']:
            logger.info(f"Using model {model_entry['name']} for prediction")
            return model_entry, f"Using model {model_entry['name']}", True
            
        logger.error(f"Model {model_entry['name']} has unexpected status: {model_entry['status']}")
        return None, "No suitable model found. Please train a new model.", False

    def predict(self, tasks, **kwargs):
        """
        Performs object detection on provided images using the trained YOLO model.

        This method:
        1. Verifies model availability and readiness
        2. Loads the trained YOLO model
        3. Processes each image for object detection
        4. Formats predictions according to Label Studio requirements

        Args:
            tasks (list): List of tasks from Label Studio, each containing image data
            **kwargs: Additional arguments passed from Label Studio

        Returns:
            list: List of predictions for each image, formatted as:
                {
                    "result": List of detected objects with coordinates and labels,
                    "score": Minimum confidence score of detections,
                    "model_version": Current model version identifier
                }

        Raises:
            Exception: If model is not yet trained or is currently being trained
        """
        logger.info('Starting prediction')
        
        # Get appropriate model for prediction
        model_entry, status_message, can_predict = self.get_model_for_predict()
        
        if not can_predict:
            logger.error(f'Cannot perform prediction: {status_message}')
            raise Exception(status_message)
            
        logger.info(f"Model info: {model_entry['name']}, Status: {model_entry['status']}")

        # Load the model
        model_path = model_entry["weights_file"]
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise Exception(f"Model file not found: {model_path}")
            
        self.detector = YOLO(model_path)
        logger.debug('YOLO model loaded successfully')
        
        # Get model labels
        self.labels = self.detector.names
        logger.debug(f'Model labels: {self.labels}')

        # Get images to annotate
        images = [get_local_path(
            task['data'][self.value], hostname=HOSTNAME, access_token=API_KEY) for task in tasks]
        logger.info(f'Processing {len(images)} images')

        # Getting prediction using model
        all_images_predictions_raw_results = self.detector.predict(
            images, **model_predict_config)
        logger.debug('Raw predictions obtained')
        all_images_predictions_reformated_results = []

        for image, result in zip(images, all_images_predictions_raw_results):
            # Getting boxes from model prediction
            single_predictions = []
            score = []
            original_width, original_height = get_image_size(image)
            logger.debug(f'Processing image of size {original_width}x{original_height}')

            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                # Appending prediction to list
                single_predictions.append({
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / original_width * 100,
                        "y": xyxy[1] / original_height * 100,
                        "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                        "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                        "rectanglelabels": [self.labels[int(prediction.cls.item())]]
                    }
                })
                # Appending score to list
                score.append(prediction.conf.item())

            # Dict with final dicts with predictions
            final_prediction = {
                "result": single_predictions,
                "score": min(score) if score else 0.0,  # 最小置信度
                "model_version": model_entry["name"]
            }
            all_images_predictions_reformated_results.append(
                final_prediction)
            logger.debug(f'Processed predictions for image with {len(single_predictions)} detections')

        logger.info('Prediction completed successfully')
        return all_images_predictions_reformated_results

    def fit(self, completions, workdir=None, **kwargs):
        """
        Trains or fine-tunes the YOLO model using annotated data from Label Studio.

        This method handles the complete training pipeline including:
        1. Version control management
        2. Training data generation from Label Studio annotations
        3. Model initialization (new model or continued training)
        4. Model training with specified configuration
        5. Model status tracking and versioning

        The training process can either:
        - Start from a pre-trained YOLO model (for first version or when fine_tune_start_from_original=True)
        - Continue training from the previous best model

        Args:
            completions (list): List of completed annotations from Label Studio
            workdir (str, optional): Working directory for storing model artifacts
            **kwargs: Additional arguments from Label Studio, including:
                - data.project.id: Project ID for fetching annotations

        Returns:
            dict: Contains the path to the trained model file
                {
                    'model_file': str  # Path to the trained model weights
                }

        Raises:
            Exception: If model is already being fitted or if project_id cannot be found
        """
        logger.info('Starting model fitting process')
        
        # Get or create model version for training
        model_entry, status_message, can_train = self.get_model_version_for_fit()
        if not can_train:
            logger.error(f'Cannot train model: {status_message}')
            raise Exception(status_message)
        
        logger.info(f'{status_message}, Model version: {model_entry["name"]}')
        
        # Update model status to training
        self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['TRAINING'])

        # Get project ID for data download
        if completions:
            for completion in completions:
                project_id = completion['project']
                break
        elif kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
        else:
            logger.error('No project_id found in completions or kwargs')
            self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['FAILED'])
            raise Exception('No project_id found')

        # Generate training data
        if not self.gen_train_data(project_id):
            logger.error('Training data generation failed')
            self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['FAILED'])
            raise Exception('Failed to generate training data')
            
        logger.info('Starting model training')
        logger.debug(f'Ultralytics checks: {ultralytics.checks()}')
        
        start_time = datetime.now()
        
        # Determine starting point for training
        if model_entry["id"] == 1 or fine_tune_start_from_original:
            # First model version or explicitly starting from original
            logger.info(f'Starting training from pre-trained model: {yolo_model_version}')
            model = YOLO(yolo_model_version)
        else:
            # Continue from previous best model if available
            previous_model = None
            for m in self.model_registry.registry["models"]:
                if m["id"] == model_entry["id"] - 1 and m["status"] == MODEL_STATUS['COMPLETE']:
                    previous_model = m
                    break
                    
            if previous_model and os.path.exists(previous_model["weights_file"]):
                logger.info(f'Continuing training from previous model: {previous_model["name"]}')
                model = YOLO(previous_model["weights_file"])
            else:
                logger.info(f'No valid previous model found, starting from pre-trained model: {yolo_model_version}')
                model = YOLO(yolo_model_version)

        # Verify data.yaml exists
        if not os.path.exists(model_train_config['data']):
            logger.error('data.yaml not found')
            self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['FAILED'])
            raise Exception('data.yaml not found')

        # Redirect Ultralytics stdout to logger
        import sys
        from io import StringIO

        class LoggerWriter:
            def __init__(self, logger_func):
                self.logger_func = logger_func
                self.buf = []

            def write(self, message):
                if message.strip():  # Only log non-empty messages
                    self.logger_func(message.strip())

            def flush(self):
                pass

        # Save original stdout
        original_stdout = sys.stdout
        # Redirect stdout to our logger
        sys.stdout = LoggerWriter(logger.info)

        try:
            # Run model training
            results = model.train(**model_train_config)
            
            # Store metrics
            if hasattr(results, 'results_dict'):
                metrics = {
                    "map": results.results_dict.get('metrics/mAP50-95(B)', 0),
                    "map50": results.results_dict.get('metrics/mAP50(B)', 0),
                    "precision": results.results_dict.get('metrics/precision(B)', 0),
                    "recall": results.results_dict.get('metrics/recall(B)', 0),
                }
                self.model_registry.update_model_metrics(model_entry["name"], metrics)
                
            # Check if the model was trained successfully
            if os.path.exists(YOLO_DEFAULT_OUTPUT_PATH):
                # Move the trained model to the dedicated version directory
                shutil.copy(YOLO_DEFAULT_OUTPUT_PATH, model_entry["weights_file"])
                logger.info(f'Saved model weights to: {model_entry["weights_file"]}')
                
                # Set as current model
                self.model_registry.set_current_model(model_entry["name"])
                
                # Update status to completed
                self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['COMPLETE'])
                logger.info(f'Model {model_entry["name"]} training completed successfully')
            else:
                logger.error('Model training failed - no weights file produced')
                self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['FAILED'])
                raise Exception('Model training failed - no weights file produced')
                
        except Exception as e:
            logger.error(f'Model training error: {str(e)}', exc_info=True)
            self.model_registry.update_model_status(model_entry["name"], MODEL_STATUS['FAILED'])
            raise Exception(f'Model training error: {str(e)}')
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        logger.info(f'Training completed in {training_time} seconds')

        return {'model_file': model_entry["weights_file"]}

    def gen_train_data(self, project_id):
        """
        Generates training data from Label Studio annotations in YOLO format.

        This method:
        1. Downloads annotated data from Label Studio
        2. Creates train/validation split (80/20) using consistent hash-based splitting
        3. Organizes data in YOLO-compatible directory structure
        4. Generates data.yaml configuration file

        Args:
            project_id (int): Label Studio project identifier

        Returns:
            bool: True if data generation was successful, False otherwise

        The generated dataset structure:
        datasets/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── data.yaml
        """
        logger.info(f'Generating training data for project {project_id}')
        try:
            start_time = datetime.now()
            download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export?export_type=YOLO&download_all_tasks=false&download_resources=true'
            response = requests.get(download_url, headers={
                                    'Authorization': f'Token {API_KEY}'})
            
            if response.status_code != 200:
                logger.error(f'Failed to download data: HTTP {response.status_code} - {response.text}')
                return False
                
            if not response.content:
                logger.error('Downloaded empty data archive')
                return False
            
            datasets_dir = os.path.join(ROOT, 'datasets')
            if not os.path.exists(datasets_dir):
                logger.info('Creating datasets directory')
                os.makedirs(datasets_dir)
                
            zip_path = os.path.join(datasets_dir, 'train.zip')
            train_path = os.path.join(datasets_dir, 'train')

            logger.debug('Saving downloaded data')
            with open(zip_path, 'wb') as file:
                file.write(response.content)
                file.flush()
                
            # Validate zip file
            if not zipfile.is_zipfile(zip_path):
                logger.error('Downloaded file is not a valid zip archive')
                os.remove(zip_path)
                return False
                
            logger.debug('Extracting zip file')
            try:
                with zipfile.ZipFile(zip_path) as f:
                    f.extractall(train_path)
                os.remove(zip_path)
            except zipfile.BadZipFile:
                logger.error('Bad zip file received from Label Studio')
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return False
            except Exception as e:
                logger.error(f'Error extracting zip file: {str(e)}')
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return False

            train_path = Path(train_path)
            
            # Check if the extracted data contains the expected structure
            if not (train_path / 'images').exists() or not (train_path / 'labels').exists():
                logger.error('Downloaded data does not contain expected images/labels directories')
                return False
                
            val_path = train_path.parent / 'valid'
            val_path.mkdir(parents=True, exist_ok=True)
            (val_path / 'images').mkdir(exist_ok=True)
            (val_path / 'labels').mkdir(exist_ok=True)

            image_files = list((train_path / 'images').glob('*'))
            
            if not image_files:
                logger.error('No images found in downloaded data')
                return False
                
            logger.info(f'Found {len(image_files)} images in training set')

            val_images = []
            val_labels = []

            for image_path in image_files:
                filename = image_path.name
                hasher = hashlib.sha256()
                hasher.update(filename.encode())
                hash_digest = hasher.hexdigest()
                hash_value = int(hash_digest[:10], base=16) % 100

                if hash_value < 20:
                    val_images.append(image_path)
                    label_path = train_path / 'labels' / image_path.with_suffix('.txt').name
                    if label_path.exists():
                        val_labels.append(label_path)
                    else:
                        logger.warning(f'Label file not found for {image_path.name}')

            if len(val_images) == 0:
                logger.warning('No validation images selected by hash, using last image')
                num_val_images = max(1, min(int(len(image_files) * 0.2), 10))  # 20% or max 10 images
                val_images = image_files[-num_val_images:]
                for image_path in val_images:
                    label_path = train_path / 'labels' / image_path.with_suffix('.txt').name
                    if label_path.exists():
                        val_labels.append(label_path)
                    else:
                        logger.warning(f'Label file not found for {image_path.name}')

            logger.info(f'Moving {len(val_images)} images and {len(val_labels)} labels to validation set')
            for image_path in val_images:
                shutil.move(str(image_path), str(val_path / 'images' / image_path.name))

            for label_path in val_labels:
                shutil.move(str(label_path), str(val_path / 'labels' / label_path.name))

            # Check for notes.json file
            if not (train_path / 'notes.json').exists():
                logger.error('notes.json file not found in downloaded data')
                return False
                
            try:
                notes = json.loads((train_path / 'notes.json').read_text())
                nc = len(notes['categories'])
                names = [category['name'] for category in notes['categories']]
                
                if nc == 0:
                    logger.error('No categories found in notes.json')
                    return False
                    
                logger.debug(f'Found {nc} categories: {names}')

                data_yaml = dict(
                    train=f'../train/images',
                    val=f'../valid/images',
                    test=f'../valid/images',
                    nc=nc,
                    names=names
                )

                yaml_path = train_path.parent / 'data.yaml'
                yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False))
                logger.info(f'Created data.yaml at {yaml_path}')
            except json.JSONDecodeError:
                logger.error('Invalid notes.json file')
                return False
            except Exception as e:
                logger.error(f'Error processing notes.json: {str(e)}')
                return False

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f'Data generation completed in {processing_time} seconds')

            return True
        except Exception as e:
            logger.error(f'Error occurred while generating training data: {str(e)}', exc_info=True)
            return False
