import flask
from flask import Flask,render_template,request,session,redirect,url_for,g
from flask.wrappers import Response #For hosting webpages on local machine
import cv2
import imutils
import pytesseract
from pytesseract import Output
import sqlite3
import numpy as np
import pandas as pd
import flask
app=Flask(__name__) #initialize flask object
app.secret_key = "UnifromDetection"

    
#Main Code
import cv2
import imutils
import pytesseract
from pytesseract import Output
import sqlite3
import numpy as np
import pandas as pd
def uniform():
    haar_upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_upperbody.xml")


    video_capture = cv2.VideoCapture(0)


    count = 0
    while True:
        ret,frames = video_capture.read()


        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 

        upper_body = haar_upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.01,
            minNeighbors = 5,
            minSize = (100, 200), 
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in upper_body:
            var1=cv2.rectangle(frames, (x, y-100), (x + w, y + h+300), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
            cv2.putText(frames, "Person", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
            crop = frames[y-50:y+h+100,x:x+w]
            if crop.any():
                        

                        gray,img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
                        gray = cv2.bitwise_not(img_bin)
                        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
                        kernel =np.ones((1,1),np.uint8)
                        img =cv2.erode(gray,kernel,iterations=1)
                        img = cv2.dilate(img,kernel,iterations=1)
                        configs = '--oem 3 --psm 11/6'
                        result = pytesseract.image_to_data(img, output_type=Output.DICT,config=configs, lang='eng')


                        boxes = len(result['text'])
                        for sequence_number in range(boxes):
                            if int(result['conf'][sequence_number]) > 30: 
                                (x, y, w, h) = (result['left'][sequence_number], result['top'][sequence_number],
                                                result['width'][sequence_number], result['height'][sequence_number])


                                cv2.rectangle(img, (x, y-50), (x + w, y + h+100), (128, 255, 128), 3)
                        for angle in np.arange(0, 180, 30):
                            rotated = imutils.rotate_bound(crop, angle)
                            label = pytesseract.image_to_string(rotated,config=configs)
                        
                            var = label.lower()
                            
                            conn = sqlite3.connect("company.db")
                            cur = conn.cursor()
                            cur.execute("select * from uniform;")
                            results = cur.fetchall()
                            df = pd.read_sql_query("SELECT * FROM uniform", conn)
                            df
                            keys=[]
                            my_dict = dict(zip(df.Name, df.AdditionalLabel))
                       

                            for key,value in my_dict.items():





                                if key in var:

                                    keys.append(key)
                                    print('name of the company is',keys)
                                    count=0

                                elif value in var:

                                    keys.append(key)
                                    print('name of the company is',keys)
                                    count=0

                                else:
                                    if count < 1:
                                        print("No delivery guy detected.")
                                        count+=1




            cv2.imshow('Video', frames) 


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        cv2.imshow('Video', frames)

    video_capture.release()
    cv2.destroyAllWindows()
uniform() 
 
 

    
    
    
    
'''def createFramestoBlob():
    
    while True:
        frame=uniform()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')'''


    



@app.route('/index') #create index page
@app.route('/') #index page works with both / and /index
def index():
	return render_template('index.html') #render index.html

'''@app.route('/video_feed')
def videoStream():
    return Response(createFramestoBlob(),mimetype='multipart/x-mixed-replace; boundary=frame')

#recieve video stream from post request
@app.route('/predict', methods=['POST'])
def RecieveStreamBlobObject():
    if request.method == 'POST':
        data = request.get_json()
        #json to video
        data = json.loads(data)
        data = data['data']
        data = base64.b64decode(data)
        np_data = np.fromstring(data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        #convert images to video stream
        ret, jpeg = cv2.imencode('.jpg', img)
        return jsonify(result=jpeg.tobytes())'''

    

    

if __name__ == '__main__':
    app.run(port = 8000, debug = True)




