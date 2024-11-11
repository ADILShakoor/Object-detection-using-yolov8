# Object-detection-using-yolov8
A fruit detection model from image using yolov8 model
Here's a `README.md` template based on the code you've shared for an object detection project using YOLOv8 in Google Colab. This file will include sections describing the setup, usage, and code explanations.

---

# YOLOv8 Object Detection Project

This project demonstrates object detection using the YOLOv8 model. The notebook leverages Google Colab and Google Drive to train and test a YOLOv8 model on custom data.

## Project Structure

- `notebooks/`: Contains the Colab notebook for object detection with YOLOv8.
- `data/`: Dataset with training images and annotations.
- `results/`: Directory for storing training results and output predictions.

## Requirements

- Python 3.x
- YOLOv8 (installed via `ultralytics` package)
- Google Colab and Google Drive for storage

## Installation

1. Clone the repository and navigate to the project folder:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. In the Google Colab notebook, install the required packages:
    ```python
    !pip install ultralytics
    ```

## Usage

The following steps explain how to use the Colab notebook to train and test the model.

1. **Set Up Google Drive**: Mount Google Drive to access and save files.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install YOLOv8**:
   ```python
   !pip install ultralytics
   from ultralytics import YOLO
   ```

3. **Train the Model**:
   - Define the path to the YOLOv8 model and data configuration file.
   - Start the training process with specified epochs and image size.
   ```python
   model = YOLO("yolov8m.pt")
   ROOT_DIR = '/content/drive/MyDrive/yolov_classi04'
   results = model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=20, imgsz=640)
   ```

4. **Save Results to Google Drive**:
   - Copy the results to Google Drive to retain training outputs.
   ```python
   import shutil
   destination = '/content/drive/MyDrive/yolov8_003_results'
   if os.path.exists(destination):
       shutil.rmtree(destination)
   shutil.copytree('/content/runs', destination)
   ```

5. **Make Predictions**:
   - Use the trained model to make predictions on new images.
   ```python
   model = YOLO("/content/drive/MyDrive/yolov_classi04_results/detect/train2/weights/best.pt")
   results = model.predict("/content/drive/MyDrive/yolov_classi04/test01.jpg")
   ```

6. **Extract Prediction Details**:
   - Retrieve bounding boxes, object class, and confidence scores for each prediction.
   ```python
   result = results[0]
   for box in result.boxes:
       class_id = result.names[box.cls[0].item()]
       cords = [round(x) for x in box.xyxy[0].tolist()]
       conf = round(box.conf[0].item(), 2)
       print("Object type:", class_id)
       print("Coordinates:", cords)
       print("Probability:", conf)
       print("--")
   ```

## Example Output

An example output showing object type, bounding box coordinates, and probability:

```
Object type: cat
Coordinates: [50, 30, 200, 180]
Probability: 0.85
--
Object type: dog
Coordinates: [120, 60, 300, 220]
Probability: 0.90
--
```

## Results

- **Model Accuracy**: [Your metric details, if available]
- **Example Images**: [Screenshots or results from model predictions]

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
