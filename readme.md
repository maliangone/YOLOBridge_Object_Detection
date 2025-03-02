# YOLOBridge: Intelligent Object Detection Backend for Label Studio

## Description

YOLOBridge is a sophisticated machine learning backend that connects YOLO's powerful object detection capabilities with Label Studio's annotation platform. It creates an intelligent feedback loop between annotation and model training, enabling semi-automated labeling workflows and continuous model improvement.

The system provides a complete pipeline for training, versioning, and deploying YOLO-based object detection models directly within the Label Studio ecosystem.

## Key Features

- 🧠 **Smart Model Registry** - Automatic version control with comprehensive tracking of model lineage, metrics, and artifacts
- 🔄 **Continuous Learning Loop** - Seamless integration between user annotations and model training
- 🎯 **State-of-the-Art Detection** - Leveraging the latest YOLOv8 architectures for accurate object detection
- 📊 **Intelligent Data Management** - Automated train/validation splits with consistent hashing for reproducibility
- 🚀 **Production-Ready Deployment** - Docker support for easy scaling in production environments
- 📝 **Extensive Logging** - Comprehensive logging system for debugging and performance monitoring
- ⚙️ **Flexible Configuration** - Easily customizable settings for both inference and training
- 🔍 **Real-Time Inference** - Fast prediction capabilities for interactive labeling sessions

## Requirements

- Python 3.8+
- Label Studio (Now only support 1.10.x)
- Docker (optional, but recommended)
- CUDA-compatible GPU (recommended for training)

## Installation

### Using Docker (Recommended)

```bash
docker-compose up -d
```

### Manual Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export LABEL_STUDIO_HOST=<your-label-studio-host>
export LABEL_STUDIO_API_KEY=<your-api-key>
```

## Project Structure

```
├── yolo.py              # Main implementation with ML Backend classes
├── yolo_config.py       # Configuration settings for model and training
├── _wsgi.py             # WSGI entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
└── docker-compose.yml   # Docker Compose configuration
```

## Configuration

The system can be easily configured through `yolo_config.py`. Key settings include:

### Model Configuration
```python
# YOLO model version selection (yolov8s, yolov8m, yolov8l, yolov8x)
yolo_model_version = 'yolov8x'

# Whether to retrain from scratch or continue from previous version
fine_tune_start_from_original = False
```

### Prediction Configuration
```python
model_predict_config = dict(
    conf=0.3,  # Confidence threshold
    iou=0.1    # IoU threshold
)
```

### Training Configuration
```python
model_train_config = dict(
    data='datasets/data.yaml',
    epochs=1000,
    patience=20,
    plots=True,
    exist_ok=True,
    verbose=True
)
```

## Usage

1. Start Label Studio and create a project with object detection configuration

2. Configure the ML Backend in Label Studio settings:
   - Add ML Backend URL (e.g., `http://localhost:9090`)
   - Enable "Use for interactive preannotations"
   - Enable "Use for autoannotation"

3. Start annotating and experience the intelligent workflow:
   - The system automatically trains on your annotations
   - Provides predictions for new images
   - Tracks model versions and performance metrics

## API Integration

YOLOBridge implements the Label Studio ML Backend API with three core endpoints:

- `/predict` - For real-time model predictions
- `/train` - For model training and fine-tuning
- `/health` - For health checks and system status

## Model Registry

The system maintains a comprehensive model registry that tracks:

- Model versions and lineage
- Training configurations
- Performance metrics (mAP, precision, recall)
- Model status (initializing, training, completed, failed)
- Weights file locations

## License

MIT

## Support

For issues and feature requests, please create an issue in the repository.
