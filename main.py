from ultralytics import YOLO
import cv2

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
count=0
cap =  cv2.VideoCapture('demo/v2.mp4')
while True:
    
    status , frame = cap.read()
    frame = cv2.resize(frame, (720,400))
    
    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            x1,y1 , x2,y2 = box.xyxy[0]
            x1,y1 , x2,y2 = int(x1),int(y1) , int(x2),int(y2)
            cv2.rectangle(frame,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,255),thickness=1)
            
            conf = int((box.conf[0] * 100)) / 100
            cv2.putText(frame,org=(max(0,x1),max(0,y1-5)),text=str(conf),color=(0,0,255),
                        thickness=1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1)    
        
            cls_id = int(box.cls[0])
            current_class = classNames[cls_id]
            cv2.putText(frame,org=(max(0,x2),max(0,y1-5)),text=classNames[cls_id],color=(0,255,0),
                        thickness=1,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1)
            
            cv2.line(frame,pt1=(0,250),pt2 = (720,260),color=(0,0,0),thickness=1)
            lp1 = (0,250)
            lp2 = (720,260)
            mdlp = (360,255)
            boxpt1 = (x1,y1)
            boxpt2 =  (x2,y2)
            
            if (current_class == 'car') and (conf > 0.3) and y2 > 250 :
                print(y2)
                count+=1    
            
            cv2.putText(frame,text=str(count),org=((700-40),20),fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2,thickness=1,color=(255,0,0)) 
    
    cv2.imshow('Car Counter',frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    
cv2.destroyAllWindows()
cap.release()