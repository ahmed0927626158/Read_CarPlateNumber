import cv2
import easyocr
import os,re
import numpy as np
import pymongo
from datetime import datetime
def enrtanceCamera():
    client = pymongo.MongoClient('localhost', 27017)
    db = client.npr
    
    if not os.path.exists('plateImage'):
        os.makedirs('plateImage')

    path = os.path.join('model', 'npr.xml')
    classifier=cv2.CascadeClassifier(path)
    
    config_file=os.path.join('model','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    frozen_model=os.path.join('model','frozen_inference_graph.pb')

    model=cv2.dnn_DetectionModel(frozen_model,config_file)

    ClassLabels=[]
    Label_file=os.path.join('model','Labels.txt')

    with open(Label_file,'rt') as fl:
        ClassLabels=fl.read().rstrip('\n').split('\n')

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)

    global plate
   
    global car_region
    global time

    path = os.path.join('static','video', 'C0035.MP4')
    readPlate=cv2.VideoCapture(path)

    while True:

        if readPlate.get(cv2.CAP_PROP_POS_FRAMES)==readPlate.get(cv2.CAP_PROP_FRAME_COUNT):
            readPlate.set(cv2.CAP_PROP_POS_FRAMES,0)
        success,frame = readPlate.read()
        if not success:
            break
        else:
            number_Of_Space=0

            gray_fram=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            blur_frame=cv2.GaussianBlur(gray_fram,(3,3),1)
            threshold_frame=cv2.adaptiveThreshold(blur_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
            median_frame=cv2.medianBlur(threshold_frame,5)
            kernel=np.ones((3,3),np.uint8)
            dilate_frame=cv2.dilate(median_frame,kernel,iterations=1)
           # countt
            num_Pixels=cv2.countNonZero(dilate_frame)
          #Image=frame[y:y+height,x:x+width]
            font_size=3
            font=cv2.FONT_HERSHEY_PLAIN

            Index_Label,confidence,bbox=model.detect(frame,confThreshold=0.55)

            if(len(Index_Label)!=0):
                    for index,conf,box in  zip(Index_Label.flatten(),confidence.flatten(),bbox):
                         if(index<=80):
                            cv2.rectangle(frame,box,(255,0,0),2)
                            cv2.putText(frame,ClassLabels[index-1],(box[0]+10,box[1]+40),font,fontScale=font_size,color=(0,255,0),thickness=3)
                            if ClassLabels[index-1]=="car" or ClassLabels[index-1]=="truck" or ClassLabels[index-1]=="bus":
                                color=(0,0,255)
                                thickness=2
                                global read
                                global car_code
                                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                nplate=classifier.detectMultiScale(gray,1.1,4)
                                region_detect=False
                                if nplate!=():
                                    for (X,Y,w,h) in nplate:
                                        a,b=(int(0.02*frame.shape[0]),int(0.025*frame.shape[1]))
                                        plate=frame[Y+a:Y+h-a,X+b:X+w-b,:]
                                        plate_img=frame[Y+a:Y+h-a,X+b:X+w-b,:]
                                        cv2.rectangle(frame,(X,Y),(X+w,Y+h),(0,255,0),3)
                                        cv2.rectangle(frame,(X-200,Y-150),(X+w+300,Y),(0,255,0),5)
                                        image_name=datetime.now().strftime("%y%m%d-%H%M%S")
                                        copy_plate=np.copy(plate_img)
                                        # image_name=now.strftime("%y-%m-%d %H፡ %M ፡%S")
                                        # cv2.imwrite('./plateImage/plate'+image_name+'.jpg',copy_plate)
                                        # image_name=now.strftime("%y-%m-%d %H፡ %M ፡%S")



                                        cv2.imwrite('./plateImage/plate'+image_name+'.jpg',plate)
                                        cascade_adddis=cv2.CascadeClassifier(os.path.join('model', 'region_prediction.xml'))
                                        plate_region=frame[Y+a:Y+h-a,X+b:X+w-b,:]
                                        plate=cv2.imread('./plateImage/plate'+image_name+'.jpg')

                                        try:
                                            addis=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                            region_addis=cascade_adddis.detectMultiScale(addis,1.1,4)
                                            if region_addis!=():
                                                car_region="A A"
                                                cv2.putText(frame,"A A",(X-100,Y-10),font,fontScale=2,color=(0,0,255),thickness=2)
                                            
                                            #grayRegion=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                                            
                                           
                                            # cv2.imwrite('./plateImage/plate'+str(1)+'.jpg',img)

                                            # plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))
                                            reader=easyocr.Reader(['en'])  
                                         
                                            font_size=3
                                            font=cv2.FONT_HERSHEY_PLAIN

                                            # Identify the car code using HSV color
                                            # Check if the car code is blue 
                                            hsv=cv2.cvtColor(plate,cv2.COLOR_BGR2HSV) #change the image to HSV color
                                            # lower_blue = np.array([96,100,0]) #finding the lower bound of blue color HSV threshold value
                                            # upper_blue=np.array([136,255,255]) #finding the uper bound of blue color HSV threshold value
                                            # mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)#masking the other color and extracting the
                                            
                                            # im_mask=mask_blue>0
                                            # blue=np.zeros_like(plate,np.uint8)
                                            # blue[im_mask]=plate[im_mask]

                                            # car_code=reader.readtext(blue)
                                            # if car_code is not None:
                                            #     car_code="0 2"
                                        # Read Car code
                                            lower_red = np.array([160,100,100])
                                            upper_red = np.array([179,255,255])
                                            
                                            mask_red=cv2.inRange(hsv,lower_red,upper_red)
                                            text_red=reader.readtext(mask_red)
                                            if len(text_red)!=0:
                                                car_code='0 1'
                                                print(car_code)
                                            # Blue car code
                                            if len(text_red)==0:
                                                lower_blue = np.array([96,100,0])
                                                upper_blue = np.array([136,255,255])
                                                mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
                                                text_blue=reader.readtext(mask_blue)
                                                if len(text_blue)!=0:
                                                    car_code='0 2'
                                                    print(car_code)

                                                # Green car code
                                                if len(text_blue)==0:
                                                    lower_green = np.array([36,10,0])
                                                    upper_green = np.array([86,255,255])
                                                    mask_green=cv2.inRange(hsv,lower_green,upper_green)
                                                    text_green=reader.readtext(mask_green)
                                                    if len(text_green)!=0:
                                                        car_code='0 3'
                                                        print(car_code)

                                                if len(text_blue)==0:
                                                    lower_black = np.array([0,0,0])
                                                    upper_black = np.array([350,55,100])
                                                    mask_black=cv2.inRange(hsv,lower_black,upper_black)
                                                    text_black=reader.readtext(mask_black)
                                                    if len(text_black)!=0:
                                                        car_cod='0 4'
                                                        print(car_code)


                                            

                                            text=reader.readtext(gray)
                            
                                            if(len(text)>0):
                                                writeT=text[0][-2]
                                                plate_number_final=None
                                                font=cv2.FONT_HERSHEY_SIMPLEX
                                                plate_list=text[1] 
                                                plate_abc=plate_list[-2]
                                                
                                                if len(plate_list[-2])==5:
                                                    regex = r'([0-9]{5})'
                                                    if re.fullmatch(regex,plate_list[-2]):
                                                        print("plate number is "+plate_list[-2])
                                                        plate_number_final=plate_list[-2]
                                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                                        plate=plate_number_final
                                                        
                                                        cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)


                                                        
                                                if len(plate_list[-2])==6:
                                                    plate_number=plate_list[-2]
                                                    regex = r'([a-cA-C]{1}[0-9]{5})'
                                                    if re.fullmatch(regex,plate_list[-2]):
                                                        plate_number_final=plate_list[-2]
                                                        print("plate number is "+plate_list[-2])
                                                        plate_number_final=plate_list[-2]
                                                        plate=plate_number_final
                                                        
                                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                                        cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)

                                                    
                                                    plate_number=plate_list[-2]
                                                    regex = r'([0-9]{6})'
                                                    if re.fullmatch(regex,plate_list[-2]):
                                                        plate_number=plate_number[0]+plate_number[1]+plate_number[2]+plate_number[3]+plate_number[4]
                                                        print("plate number is "+plate_number)
                                                        plate_number_final=plate_number
                                                        plate=plate_number_final
                                                    
                                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                                        cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)

                                                        


                                                if len(plate_list[-2])>6:
                                                    plate_abc=plate_list[-2]
                                                    regex_full=r'([0-9]{6,})'
                                                    if re.fullmatch(regex_full,plate_abc):
                                                        plate_abc=plate_abc[0]+plate_abc[1]+plate_abc[2]+plate_abc[3]+plate_abc[4]
                                                        print("plate number is "+plate_abc)   
                                                        plate_number_final=plate_abc
                                                        plate=plate_number_final
                                                    
                                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                                        cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)

                                                        

                                                    first_index=plate_abc[0]
                                                    regex_first = r'([^a-cA-C0-9])'
                                                    if re.fullmatch(regex_first,first_index):
                                                        plate_abc=plate_abc.replace(plate_abc[0], '')
                                                        #print(len(plate_abc))
                                                        if len(plate_abc)==6:
                                                            regex_after_drop = r'([a-cA-C]{1}[0-9]{5})'
                                                            first_indexx=plate_abc
                                                            if re.fullmatch(regex_after_drop,first_indexx):
                                                                #plate_abc=plate_abc.replace(plate_abc[0], '')
                                                                print("plate number is "+plate_abc)
                                                                plate_number_final=plate_abc
                                                                plate=plate_number_final
                                                                
                                                                font=cv2.FONT_HERSHEY_SIMPLEX
                                                                cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)


                                                                
                                                        if len(plate_abc)>6:
                                                            first_after_drop=plate_abc[0]
                                                            regex_first_check= r'([a-cA-C])'
                                                            if re.fullmatch(regex_first_check,first_after_drop):
                                                                if len(plate_abc)>6:
                                                                    regex_second=r'([a-cA-C]{1}[0-9]{5})'
                                                                    plate_abc=plate_abc[0]+plate_abc[1]+plate_abc[2]+plate_abc[3]+plate_abc[4]+plate_abc[5]
                                                                    #r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+   [a-z0-9|A-Z0-9~`!@#$%^&*()-_+={}[]\/:;\"\'<>,.?]'
                                                                    if re.fullmatch(regex_second,plate_abc):
                                                                        print("plate number is "+plate_abc)
                                                                        plate_number_final=plate_abc
                                                                        plate=plate_number_final
                                                                       
                                                                        font=cv2.FONT_HERSHEY_SIMPLEX
                                                                        cv2.putText(frame,car_code+plate_abc,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)
                                                                        



                                                    regex_first_check= r'([a-cA-C])' 
                                                    if re.fullmatch(regex_first_check,first_index):
                                                        if len(plate_abc)>6:
                                                            regex_second=r'([a-cA-C]{1}[0-9]{5})'
                                                            plate_abc=plate_abc[0]+plate_abc[1]+plate_abc[2]+plate_abc[3]+plate_abc[4]+plate_abc[5]
                                                            #r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+   [a-z0-9|A-Z0-9~`!@#$%^&*()-_+={}[]\/:;\"\'<>,.?]'
                                                            if re.fullmatch(regex_second,plate_abc):
                                                                print("plate number is "+plate_abc)
                                                                plate_number_final=plate_abc
                                                                plate=plate_number_final
                                                              
                                                            
                                                font=cv2.FONT_HERSHEY_SIMPLEX
                                                cv2.putText(frame,car_code+plate_number_final,(X,Y-10),font,fontScale=3,color=(0,0,255),thickness=2)
                                                if(plate!=None):
                                                  
                                                #   Start timer of the car
                                                    now=datetime.now()
                                                    dt_string = now.strftime("%d/%m/%Y")
                                                    hours = str(now.hour)
                                                    minutes=str(now.minute)
                                                    timeEntry=hours+":"+minutes
                                                    status="pending"
                                                    check_status=db.reservations.find_one({"car_region":car_region,"car_code":car_code,"car_plate":plate,"status":status})
                                                    if check_status is None:
                                                        db.reservations.insert_one({"car_region":car_region,"car_code":car_code,"car_plate":plate,"entryDate":dt_string,"entryTime":timeEntry,"exitDate":"-","exitTime":"-","totalPrice":"0.0","status":status})
                                                   
                                                    # Find the number of free slots and subtract one from the free slot
                                                    # slots=db.numberOfSlots.find_one({})
                                                    # oldFreeSlot=slots["freeSlots"]
                                                    # updateFreeSlot=int(oldFreeSlot)-1
                                                    
                                                    # # Update the freeSlot in the database
                                                    # db.numberOfSlots.update_one({'freeSlots':oldFreeSlot},{"$set":{'freeSlots':updateFreeSlot}})
                                                 
                                                


                                        except:
                                            pass
                                        global counter
                                        counter=0
     
            ret,buffer=cv2.imencode('.jpg',gray_fram)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n'+gray_fram+b'r\n')