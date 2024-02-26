## Vehicle and Licence Plate Detection + Licence Plate Recognition with PaddlePaddle.
![image](https://github.com/MisterFoziljon/PaddleDetection-and-PaddleOCR-project/blob/main/output.png)
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

### 3. Number Plate Recognition.
#### Inference:
```python
import cv2
from LicencePlateRecognizer import Recognizer

model_path = '/home/admin/paddle_vehicle/models/LP_recognition/'
characters_dict = '/home/admin/paddle_vehicle/characters_dict.txt'
model = Recognizer(model_path, characters_dict)

image = cv2.imread("test.jpg")
result = model(image)
```

### Deploy with PaddlePaddle:
```cmd
python deploy.py
```

### Models
You can download models from this [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/deploy/pipeline/docs/tutorials/PPVehicle_QUICK_STARTED_en.md)


### Problems.
1. PP-vehicle detection modeli yaxshi. FPS taxminan 25.
2. PP-plate detection modeli yaxshi emas. Paddle o'zini modeli. Faqat chet el moshinalariga o'qitilgan. Deteksiyada oynadigi akslaniyam oladi. Number plate XX qismini yoki 123ABC qismini alohida kesib oladi. Asosiy adashadigan qismi. O'zimizani nomerla bilan FineTuning qilish kere. Tezligi 10-15 FPS butun frame uchun.
3. OCR uchun PP-plate rec umuman ishlamidi. PaddleOCR ni o'zini modelini ishlatganda yaqin keganda yaxshi aniqlidi. Tezligi birinchi marta ishlaganda: 2 FPS, keyingi ishlashiga: 25-50 FPS atrofida ishladi.
4. PaddlePaddle modeli dinamik bo'lgani uchun TensorRT da ishlatib bo'lmadi. Static bo'lishi kerak model.
5. PaddlePaddle modelini o'zidan onnx, tensorrt da ishlatib ko'rish kerak.
