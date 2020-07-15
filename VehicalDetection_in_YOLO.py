import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression

# ------------ Vehicals Detection and Counting ------------

# Load Yolo
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
classes = []
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("input/trafficMany.jpg")
img = cv2.resize(img, None, fx=1, fy=1)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
count=0
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[25]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if label == "bicycle" or label == "car" or label == "motorbike" or label == "bus" or label == "truck":
            count += 1

# ------------ Vehicals Detection and Counting End------------



#Machine Learning Algorithm to calculate Estimated Time needed by vehicals to cross the traffic signal
#------------- start -----------------------------------------

data = pd.read_csv(r'dataset\vehical_time_dataset.csv')

real_x = data.iloc[:,0].values
real_y = data.iloc[:,1].values
real_x = real_x.reshape(-1,1)
real_y = real_y.reshape(-1,1)

lin = LinearRegression()
lin.fit(real_x,real_y)

pred_y = lin.predict([[count]])
#print(pred_y)




#-------------- End --------------------------------------------



cv2.putText(img, "Total Detectable Vehicals : " + str(count), (30,30), cv2.FONT_HERSHEY_PLAIN, fontScale=2,color=(255,255,255),thickness=2)
cv2.putText(img, "Total Estimated Time : " + str(int(pred_y)) + "sec", (30,60), cv2.FONT_HERSHEY_PLAIN, fontScale=2,color=(255,255,255),thickness=2)

# cv2.putText(img, "Total Detectable Vehicals : " + str(count), (30,30), cv2.FONT_HERSHEY_PLAIN, fontScale=2,color=(15,15,155),thickness=2)
# cv2.putText(img, "Total Estimated Time : " + str(int(pred_y)) + "sec", (30,60), cv2.FONT_HERSHEY_PLAIN, fontScale=2,color=(15,15,155),thickness=2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()