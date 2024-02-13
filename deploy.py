from VehicleDetector import Detector
from LicencePlateDetector import PlateDetector
from LicencePlateRecognizer import Recognizer
import cv2
import time

path = "/home/foziljon/PADDLE_VEHICLE/models/"

vehicle_model_path = path + "Paddle_vehicle_model/"
lpd_model_path = path + "Paddle_NPD_model/"
lpr_model_path = path + "LPR/"
lpr_dictionary_path = "/home/foziljon/PADDLE_VEHICLE/ppocr/utils/en_dict.txt"

vehicle_model = Detector(vehicle_model_path)
lpd_model = PlateDetector(lpd_model_path)
lpr_model = Recognizer(lpr_model_path, lpr_dictionary_path)

video = cv2.VideoCapture("8.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

while video.isOpened():
    
    ret, frame = video.read()
    if not ret:
        break
    start = time.time()
    frame = cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)
    
    vehicle_boxes = vehicle_model([frame])
    
    for vehicle_bbox in vehicle_boxes:
        
        xmin, ymin, xmax, ymax = vehicle_bbox
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (17, 212, 105), 2)
        
        vehicle = frame[ymin:ymax,xmin:xmax]
        try:
            number_plate_boxes = lpd_model(vehicle)
                
            xpmin, ypmin, xpmax, ypmax = number_plate_boxes
            cv2.rectangle(frame, (xmin+xpmin-3,ymin+ypmin-3), (xmin+xpmax+3,ymin+ypmax+3), (17, 212, 105), 2)
            
            plate = vehicle[ypmin-3:ypmax+3, xpmin-3:xpmax+3]
            text = lpr_model(plate)[0]  
            (label_width, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            rect_position = (xmax - label_width, ymax - label_height)
    
            cv2.rectangle(frame, (rect_position[0]-10, rect_position[1]-10), (rect_position[0] + label_width, rect_position[1] + label_height), (17, 212, 105), -1)
            cv2.putText(frame, text, (xmax-label_width-5,ymax-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)
        except:
            continue
        
        
    end = time.time()
    print(1./(end-start))
    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video.release()
out.release()
cv2.destroyAllWindows()      