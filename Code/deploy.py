# Importing required libraries
import torch
import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt
import imutils

# Initializing EasyOCR
EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

# Initializing counter
ct=1

### -------------------------------------- Function for detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ Function to plot Bounding-Box and results --------------------------------------------------------
def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    # Looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            # Getting Bounding Box co-ordinates 
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            text_d = classes[int(labels[i])]
            
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            print(f"Plate No. = {plate_num}")

            # Plotting rectangle & displaying the detected number plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.rectangle(frame, (x1, y2), (x1+200, y2-30), (255,0,0), -1) 
            cv2.putText(frame, f"{plate_num}", (x1, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)

    return frame

### ---------------------------- Function to recognize license plate --------------------------------------
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    global ct

    # Separate coordinates from box
    xmin, ymin, xmax, ymax = coords

    # Get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    # Cropping the number plate from the whole image
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] 
    
    ocr_result = reader.readtext(nplate)

    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
        # Saving each frame
        # cv2.imwrite("Output"+str(ct)+".jpg", nplate) 
        # ct+=1
        
    return text

# Function to filter out wrong detections
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = []
    var = ""
    print(ocr_result)

    for result in ocr_result:
        var = result[1]

    plate.append(var)    
    return plate

### ---------------------------------------------- Main function -----------------------------------------------------
def npr(img_path=None, vid_path=None, vid_out = None, livestream = None):
    print("[INFO] Loading model... ")
    # Loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## If you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='yolov5s.pt', force_reload=True) # The repo is stored locally

    # Class names in string format
    classes = model.names 

    ### --------------- For detection on image --------------------
    if img_path != None:
        # Getting the image
        img = cv2.imread(img_path)

        # Converting the input image to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Converting BGR to RGB values
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

        # Applying bi-lateral filter (Smoothening image and reducing noise, while preserving edges)
        bfilter = cv2.bilateralFilter(gray,11,17,17)

        # Detect edges in image
        edged = cv2.Canny(bfilter,30,200)

        plt.imshow(cv2.cvtColor(edged,cv2.COLOR_BGR2RGB))

        # Extracting contours(line joining all pts along boundary of img that have same intensity) from image
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours and sorting them
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]


        location = None
        for contour in contours:
            # Approximation of shape of contour
            approx = cv2.approxPolyDP(contour,10,True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[location],0,255,-1)
        new_image = cv2.bitwise_and(img,img,mask=mask)
        plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))


        (x,y) = np.where(mask==255)
        (x1,y1)=(np.min(x),np.min(y))
        (x2,y2)=(np.max(x),np.max(y))
        cropped_image = gray[x1:x2+1,y1:y2+1]
        # plt.show()

        # Extracting the text from image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)

        # Checking if vehicle any text is extracted
        if len(result) != 0:
            # Saving the output
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.savefig("Output.jpg")

            str=result[-1][-2]
            str=str.upper()

            s=""
            for a in str:
                if a!=' ' and a!='\'' and a!='.' and a!='-' and a!='*' and a!='`' and a!='"' and a!='[' and a!=']' and a!='{' and a!='}' and a!='(' and a!=')':
                    s+=a

            s = s.upper()
            print(s)

        else:
            print("No vehicle detected!")    

    ### --------------- For detection on video --------------------
    elif vid_path != None:
        # Defining video capture object
        cap = cv2.VideoCapture(vid_path)

        # If user wants to save the output video
        if vid_out: 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Getting the width of input video
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Getting the height of input video
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Getting the frame rate of inout video
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # Getting the codec of input video
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))  # Initializing video writer 

        # assert cap.isOpened()
        frame_no = 0

        # Displaying window
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)

        while True:
            # cap.read() returns 2 tuples:
            #   1. Boolean value ---> Whether a frame is grabbed or not.
            #   2. Next video frame
            ret, frame = cap.read()

            # If no next frame is found ---> End of video
            if frame is None:
                break

            # If frame is grabbed
            if ret  and frame_no % 25 == 0:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)

                if vid_out:
                    # Write each frame to output video
                    out.write(frame)

                # waitKey ---> Allow users to display window for given milliseconds
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            frame_no += 1
        
        # Releasing the video writer
        out.release()
        
        # Closing all windows
        cv2.destroyAllWindows()

    ### --------------- For detection on livestream from camera --------------------
    elif livestream !=None:
        cap = cv2.VideoCapture(livestream)

        if vid_out: 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)

        while True:
            ret, cropu1 = cap.read()

            if ret  and frame_no % 1 == 0:
                cropu1 = cv2.cvtColor(cropu1,cv2.COLOR_BGR2RGB)
                # frame = cropu1[450:700,700:1200]
                frame = cropu1
                # frame = cv2.resize(frame, None, None, fx=1, fy=1)

                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                cv2.imshow("vid_out", frame)

                if vid_out:
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_no += 1
        
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()

### -------------------  Calling the main function  -------------------------------

# npr(vid_path="./test_videos/footage5.mp4", vid_out="vid_5.mp4") ### for custom video

# npr(vid_path=0,vid_out="webcam_facemask_result.mp4") ### for webcam

npr(img_path="./test_images/c5.jpg") ### for image
             
# npr(None,None,None,'rtsp://student:student123@192.168.3.12') ### for livestream            