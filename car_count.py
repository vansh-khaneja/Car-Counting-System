from ultralytics import YOLO
import cv2
from sort import *
model = YOLO('yolov8n.pt')

class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

#image = cv2.imread("C:/Users/VANSH KHANEJA/Downloads/carview.png")
cap = cv2.VideoCapture("C:/Users/VANSH KHANEJA/Downloads/5473765-uhd_3840_2160_24fps.mp4")

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [160,400,530,400]
totalCount = []
while True:

    success,image = cap.read()

    resize = cv2.resize(image,(700,500))



    mask = cv2.imread('mask.png')

    imgRegion = cv2.bitwise_and(resize,mask)

    results = model(imgRegion,stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            currentClass = class_list[int(box.cls[0])]
            currentConf = box.conf[0]
            if currentClass == "car" and currentConf > 0.3:
                #cv2.rectangle(resize,(x1,y1),(x2,y2),(255,0,255),2)
                #cv2.putText(resize,str(class_list[int(box.cls[0])]), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255,2, cv2.LINE_AA, False)
                currentArray = np.array([x1,y1,x2,y2,currentConf])
                detections = np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(resize,(160,400),(530,400),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        w,h = x2-x1,y2-y1
        cx,cy = int(x1+w//2),int(y1+h//2)
        cv2.rectangle(resize,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)
        cv2.putText(resize,str(int(Id)), (int(x1),int(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255,2, cv2.LINE_AA, False)
        cv2.circle(resize,(cx,cy),5,(0,255,0),-1)

        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
            if totalCount.count(Id)==0:
                totalCount.append(Id)

        cv2.putText(resize,str(len(totalCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255,2, cv2.LINE_AA, False)
   
        
    cv2.imshow("Image",resize)

    cv2.waitKey(1)
