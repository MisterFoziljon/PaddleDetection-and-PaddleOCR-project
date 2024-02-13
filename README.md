## Vehicle and Licence Plate Detection + Licence Plate Recognition with PaddlePaddle.

### 1. Vehicle Detection.
#### Inference:
```python
import cv2
from VehicleDetector import Detector

model_path = '/home/admin/paddle_vehicle/models/vehicle_detection/'
model = Detector(model_path)

image = cv2.imread("test.jpg")
result = model(image)
```


### 2. Licence Plate Detection.
#### Inference:
```python
import cv2
from LicencePlateDetector import PlateDetector

model_path = '/home/admin/paddle_vehicle/models/LP_detection/'
model = PlateDetector(model_path)

image = cv2.imread("test.jpg")
result = model(image)
```

## 3. Number Plate Recognition.
### Inference:
```python
import cv2
from LicencePlateRecognizer import Recognizer

model_path = '/home/admin/paddle_vehicle/models/LP_recognition/'
characters_dict = '/home/admin/paddle_vehicle/characters_dict.txt'
model = Recognizer(model_path, characters_dict)

image = cv2.imread("test.jpg")
result = model(image)
```
