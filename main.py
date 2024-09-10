from ultralytics import YOLO
import cv2
from sort import *

model = YOLO('../YOLO Models/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
vcount = []
lineLimits = [0,230,720,230]
cap =  cv2.VideoCapture('demo/v2.mp4')
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

while True:
    
    status , frame = cap.read()
    frame = cv2.resize(frame, (720,400))
    
    results = model(frame, stream=True)
    detections = np.empty((0,5)) 
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            x1,y1 , x2,y2 = box.xyxy[0]
            x1,y1 , x2,y2 = int(x1),int(y1) , int(x2),int(y2)
            #cv2.rectangle(frame,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            
            conf = int((box.conf[0] * 100)) / 100
            #cv2.putText(frame,org=(max(0,x1),max(0,y1-5)),text=str(conf),color=(0,0,255),
            #        thickness=1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1) 
        
            cls_id = int(box.cls[0])
            current_class = classNames[cls_id]
            #cv2.putText(frame,org=(max(0,x2),max(0,y1-5)),text=current_class,color=(0,255,0),
            #            thickness=1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1)
            
            cv2.line(frame,pt1=(lineLimits[0],lineLimits[1]),pt2=(lineLimits[2],lineLimits[3]),color=(255,0,0),thickness=2)
            
            if (current_class == 'car') and (conf > 0.3):
                
                currentArray = np.array([x1,y1,x2,y2,conf])    
                detections = np.vstack((detections,currentArray))     
    
    trackerResults = tracker.update(detections)
    for results in trackerResults:
       
        x1 , y1 , x2, y2 ,id = results
        x1 , y1 , x2, y2 ,id = int(x1) , int(y1) , int(x2), int(y2) , int(id)
        
        cv2.rectangle(frame,pt1=(x1,y1),pt2=(x2,y2),color=(0,0,255),thickness=1)
        
        cv2.putText(frame,org=(max(0,x2),max(0,y2+5)),text=str(id),color=(0,255,0),
                        thickness=1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1)
        
        w ,h = x2-x1 , y2-y1 
        cx , cy = x1+w//2 , y1+h//2
        cv2.circle(frame,center=(cx,cy),thickness=-1,color=(255,255,0),radius=2)
        
        if lineLimits[0] < cx < lineLimits[2] and lineLimits[1]-5 < cy < lineLimits[2]+5:
            if vcount.count(id) == 0:
                vcount.append(id)
        
        cv2.putText(frame,text=str(len(vcount)),color=(0,255,255),org=(500,100),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=3,thickness=2)
        
    cv2.imshow('Car Counter',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cv2.destroyAllWindows()
cap.release()