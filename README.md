
# YOLOv8 Object Detection Project

This project demonstrates object detection using the YOLOv8 model. The notebook leverages Google Colab and Google Drive to train and test a YOLOv8 model on custom data.
A fruit detection model from image using yolov8 model
Here's a `README.md` template based on the code you've shared for an object detection project using YOLOv8 in Google Colab. This file will include sections describing the setup, usage, and code explanations.
## Dataset 
dataset have fallowing type of image each object in image is annotated by bounding box
- Apple
- Banana
- Grape
- Orange
- Pineapple
- Watermelon


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
    git clone : https://github.com/ADILShakoor/Object-detection-using-yolov8.git
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
   ROOT_DIR = '/content/drive/MyDrive/path current dir'
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
   load the model which you trained
   model = YOLO("weights/best.pt")
   results = model.predict("test01.jpg")
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

- **Model Accuracy**:
- ![results](https://github.com/user-attachments/assets/54b49960-b614-4ae3-934f-20d47e682429)
![R_curve](https://github.com/user-attachments/assets/270eba9c-6fd3-4339-902e-aea9aa04bb7d)
![PR_curve](https://github.com/user-attachments/assets/f3cd886d-a9b6-45e4-b3f5-9099db23e728)
![P_curve](https://github.com/user-attachments/assets/3df6fa2d-2930-401b-8fb0-808b21a412c8)
![labels_correlogram](https://github.com/user-attachments/assets/657b80dc-f302-40f2-ac2f-45dde3ac7003)
![labels](https://github.com/user-attachments/assets/6322ffc0-e66f-41f3-9db5-7ae8ff7a063a)
![F1_curve](https://github.com/user-attachments/assets/8c647de0-fd18-4243-8b71-a24d70de8c6b)
![confusion_matrix_normalized](https://github.com/user-attachments/assets/0d697a5b-4cd6-4971-aa49-5b1ec5a957ce)
![confusion_matrix](https://github.com/user-attachments/assets/0d1b0654-372c-49c3-9d81-a3b82409b164)

- **results Images**:
![val_batch2_labels](https://github.com/user-attachments/assets/5e1c94b9-cc2d-425e-b27b-89909fa01266)
![val_batch1_pred](https://github.com/user-attachments/assets/181aa6c3-480f-4c7c-b013-7d7968c4ae8d)
![val_batch1_labels](https://github.com/user-attachments/assets/f0077798-975c-49e9-b968-8ebb53dee3a4)
![val_batch0_pred](https://github.com/user-attachments/assets/6118736f-2515-4053-a988-c8813d6240ed)
![val_batch0_labels](https://github.com/user-attachments/assets/d5ad63c0-bf55-465f-92bc-8295077f7309)
![train_batch4502](https://github.com/user-attachments/assets/5c700cf9-ef13-4643-96d9-67e5ee9ac464)![val_batch2_pred](https://github.com/user-attachments/assets/095ed727-0cf2-4b44-a559-77bcb14e0459)

![train_batch4501](https://github.com/user-attachments/assets/2086cc4f-7f91-4bd0-85c3-9cf0984d6ec1)
![train_batch4500](https://github.com/user-attachments/assets/ab6c3c6e-ee21-4381-9f44-93db6f416c58)
![train_batch2](https://github.com/user-attachments/assets/4961e25f-56be-4c85-979e-632453d96999)
![train_batch1](https://github.com/user-attachments/assets/59a7767d-527b-4bd4-b4fe-49f36dab33cb)
![train_batch0](https://github.com/user-attachments/assets/0a403e4f-fe92-447f-a3d0-7283124b135f)


- - **Prediction Images**:
  - ![test03](https://github.com/user-attachments/assets/8c56e201-8565-4a7b-8366-d9205a530f09)
![test02](https://github.com/user-attachments/assets/7c785fb2-0474-421e-bf04-66d6063922cd)
![test01](https://github.com/user-attachments/assets/278fefc2-3005-4bc9-99df-251522a716d1)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
