#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2
import time
import numpy as np
from datetime import datetime
import requests , json
import pyimgur
from firebase import firebase
import math
import matplotlib.pyplot as plt


# In[19]:


global right_line
global left_line
right_line = np.array([0,0,0,0])
left_line = np.array([0,0,0,0])

net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg','yolov4-tiny-custom_best.weights')
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
classes = []
with open('obj.names','r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
objects_to_calc_distance = ['car','pedestrian','truck','biker']
frame_id = 0
to_calc_distance = []
CLIENT_ID="0e404320dbebf87"
FBConn = firebase.FirebaseApplication('https://dbd-app-92b5f-default-rtdb.firebaseio.com/',None)
path = r"F:/College/Senior_2_semester_2/driver/sent to database/l.jpg"


# In[41]:


classes


# In[20]:


def send_to_app(img,what_happened):
    now = datetime.now()
    cv2.imwrite(path, img)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    im=pyimgur.Imgur(CLIENT_ID)
    uploaded_image=im.upload_image(path,title="test")
    print('Uploaded link:',uploaded_image.link)
    result = FBConn.post('/serialnum/456xyz',"AT "+dt_string+" speed limit" + "\n" +uploaded_image.link)
    send_link = FBConn.post('/serialnum/456xyz',uploaded_image.link)


# In[21]:


def calc_slope(line):
    return (line[3]-line[1])/(line[2]-line[1])

def most_common(lst):
    return max(set(lst), key=lst.count)

def get_intersection(l1,l2):
    x1 = l1[0]
    y1 = l1[1]
    x2 = l1[2]
    y2 = l1[3]
    x3 = l2[0]
    y3 = l2[1]
    x4 = l2[2]
    y4 = l2[3]
    D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    #print("d = ",D)
    try:
        px = int(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/D)
        py = int(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/D)
        return (px,py)
    except:
        return (int(480/2),int(480/2))

def get_vanishing(canny):
    taken_lines = []
    intersections = []
    intersectionsx = []
    intersectionsy = []
    distances = []
    dist_warning = False
    lines = cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength = 20,maxLineGap = 5)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            m = calc_slope(l)
            if((l[0]>canny.shape[1]/2) and (l[2]>canny.shape[1]/2)):
                theta = np.arctan(m)
                theta = abs(np.rad2deg(theta))
                if(math.isnan(theta)):
                    theta = 0
                if(int(theta)>10 and int(theta)<80):
                    taken_lines.append(l)
            
            elif((l[0]<frame.shape[1]/2) and (l[2]<frame.shape[1]/2)):
                theta = np.arctan(m)
                theta = abs(np.rad2deg(theta))
                if(math.isnan(theta)):
                    theta = 0
                if(int(abs(theta))>10 and int(abs(theta))<80):
                    taken_lines.append(l)

    #for line in taken_lines:
        #cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,255,255), 2, cv2.LINE_AA)
    
    for i, line1 in enumerate(taken_lines[:-1]):
        for line2 in taken_lines[i+1:]:
            intersectionx,intersectiony = get_intersection(line1,line2)
            intersectionsx.append(intersectionx)
            intersectionsy.append(intersectiony)
            intersections.append((intersectionx,intersectiony))
            
    try:
        vanish_x = most_common(intersectionsx)
        vanish_y = most_common(intersectionsy)
    except:
        vanish_x = int(480/2)
        vanish_y = int(480/2)
    #cropped = ROI(canny,vanish_x,vanish_y)
    #cv2.imshow("cropped", cropped)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    return [vanish_x,vanish_y,len(intersections)]


# In[22]:


def Canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny


# In[23]:


def calc_dist(obj_dist,vanish_y,set_d = 30):
    rate = set_d/(480-vanish_y)
    DD = round(rate * (480-obj_dist),1)
    return DD


# In[45]:


def crop_vanish(frame,vanish_x,vanish_y):
    height = frame.shape[0]
    width = frame.shape[1]
    polygons = np.array([
        [(int(width/7),height-int(height/8)),(int(vanish_x),int(vanish_y)),(int((6*width)/7),height-int(height/7.2))]
    ])
    
    #polygons = np.array([
    #    [(0,360),(240,150),(480,360)]
    #])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask,polygons,255)
    cropped = cv2.bitwise_and(frame,mask)
    return cropped


# In[46]:


def get_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            line = line.reshape(4)
            x1,y1,x2,y2 = line
            try:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),6)
            except:
                return line_image
    return line_image

def get_coordinates (image, params):
    
    slope, intercept = params 
    y1 = image.shape[0]
    y2 = int(y1 * (7/10))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope) 
    
    return np.array([x1, y1, x2, y2])


def avg_lines(image, lines): 
    left_fit = [] 
    right_fit = [] 
    right_line_pos = 0
    left_line_pos = 0
    global right_line
    global left_line
    try:
        for line in lines: 
            line = line.reshape(4)
            x1, y1, x2, y2 = line
            params = np.polyfit((x1, x2), (y1, y2), 1)  
            slope = params[0] 
            y_intercept = params[1] 

            if slope < 0: 
                left_fit.append((slope, y_intercept)) #Negative slope = left lane
            else: 
                right_fit.append((slope, y_intercept)) #Positive slope = right lane
    except:
        print("no lane detected")
    
    if left_fit:
        left_avg = np.average(left_fit, axis = 0) 
        left_line = get_coordinates(image, left_avg)
        left_line_pos = left_line[0]
    
    if right_fit:
        right_avg = np.average(right_fit, axis = 0) 
        right_line = get_coordinates(image, right_avg)
        right_line_pos = right_line[0]
    
    return np.array([left_line, right_line]),left_line_pos,right_line_pos


def get_offset(img_center,left_line_pos,right_line_pos):
    #print("left",left_line_pos)
    #print("right",right_line_pos)
    lane_center = ((right_line_pos - left_line_pos)/2) + left_line_pos
    #print("lane center",lane_center)
    #print("image center",img_center)
    offset = lane_center-img_center
    return offset/101.78437

def ROI(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([
        [(int(width/7),height-int(height/8)),(170,300),(330,300),(int((6*width)/7),height-int(height/7.2))]
    ])
    
    #polygons = np.array([
    #    [(0,360),(240,150),(480,360)]
    #])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    cropped = cv2.bitwise_and(img,mask)
    return cropped


def lanes(frame,canny):
    offset = 0
    cropped_image = ROI(canny)
    #cv2.imshow("crop",cropped_image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=10,maxLineGap=5)
    #print(lines)
    averaged_lines,left_line_pos,right_line_pos = avg_lines(frame,lines)
    line_image = get_lines(frame,averaged_lines)
    frame = cv2.addWeighted(line_image,0.8,frame,1,1)
    
    if(right_line_pos & left_line_pos != 0):
        offset = get_offset(frame.shape[1]/2,left_line_pos,right_line_pos)
    
    offset = round(offset, 3)
    cv2.putText(frame, 'Offset: '+str(offset), (10, 100), cv2.FONT_HERSHEY_PLAIN, 
                   2, (0,0,0), 2)
    return offset,frame


# In[47]:


cap = cv2.VideoCapture("F:/College/Senior_2_semester_2/driver/project_video.mp4")

starting_time = time.time()
frame_id = 0
frames_to_calc_vanish = 0
to_calc_distance = []
vanishes_x = []
vanishes_y = []
intersections = 0
intersections_list = []
sent_offset_flag = False
sent_nostop_flag = False
sent_distance_violation_flag = False

while cap.isOpened():
    _,frame = cap.read()
    try:
        frame = cv2.resize(frame,(480,480))
        frame_id += 1
    except:
        break
    
    canny = Canny(frame)
    if (frame_id == 1):
        vanish_x,vanish_y,_ = get_vanishing(canny)
        frames_to_calc_vanish += 1
    
    elif(frames_to_calc_vanish == 40):
        max_intersections = max(intersections_list)
        index = intersections_list.index(max_intersections)
        vanish_x = vanishes_x[index]
        vanish_y = vanishes_y[index]
        frames_to_calc_vanish = 0
    
    else:
        vanish_x_temp,vanish_y_temp,intersections = get_vanishing(canny)
        intersections_list.append(intersections)
        vanishes_x.append(vanish_x_temp)
        vanishes_y.append(vanish_y_temp)
        frames_to_calc_vanish += 1
    
    #cv2.imshow("canny",canny)
    #cv2.waitKey()
    #cv2.destroyAllWindows
    
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB = True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    speed_limit = 120
    current_speed = 20
    
    if(frame_id>20):
        current_speed = 120
        
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            confidence = confidences[i]
            if((label == "no-stopping") and (current_speed == 0) and not sent_stop_flag):   
                send_to_app(frame,"no stopping sign violation")
                sent_nostop_flag = True
            if(label in objects_to_calc_distance):   
                distance = calc_dist(y+h,vanish_y)
                if ((distance<4) and not sent_distance_violation_flag):
                    send_to_app(frame,"drivig too close to other vehicles")
                    sent_distance_violation_flag = True
                if ((distance>4) and sent_distance_violation_flag):
                    sent_distance_violation_flag = False
                
            if("speed" in label):
                speed_limit = int(label.split("-")[-1])
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x+10, y + 30), font, 2, color, 2)
            cv2.putText(frame," d = " + str(distance), (x, y), font, 1, color, 1)
    #cropped_vanish = crop_vanish(frame,vanish_x,vanish_y)
    #cv2.imshow("cropped vanish", cropped_vanish)
    offset,frame  =  lanes(frame,canny)
    
    if (current_speed>speed_limit):
        send_to_app(frame,"speed limit exceeded")
    
    if((offset>0.7) and not sent_offset_flag):
        send_to_app(frame,"offcenter")
        sent_offset_flag = True
    
    elif((offset<0.7) and sent_offset_flag):
        sent_offset_flag = False
    
    
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    #cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow("vid", frame)
    #saveFrame = 'vid4/frame'+str(time.time())+'.jpg'
    #cv2.imwrite(saveFrame, frame)
    #plt.imshow(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
  
cap.release()
cv2.destroyAllWindows()


# ## 

# In[ ]:




