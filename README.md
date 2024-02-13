# Vehicle and Licence Plate Detection + Licence Plate Recognition with PaddlePaddle.

## 1. Vehicle Detection.
```python
import cv2
from VehicleDetector import Detector

model_path = '/home/admin/paddle_vehicle/models/vehicle_detection/'
model = Detector(model_path)

image = cv2.imread("test.jpg")
result = model(image)
```
