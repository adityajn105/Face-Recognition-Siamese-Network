import numpy as np
import cv2
import os
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

from model import SiameseNet
from train import test_oneshot

import matplotlib.pyplot as plt

#haarcascade_frontalface_default.xml is saved model for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def giveAllFaces(image,BGR_input=True,BGR_output=False):
    """
      return GRAY cropped_face,x,y,w,h 
    """
    gray = image.copy()
    if BGR_input:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if BGR_output:
        for (x, y, w, h) in faces:
            yield image[y:y+h,x:x+w,:],x,y,w,h
    else:
        for (x, y, w, h) in faces:
            yield gray[y:y+h,x:x+w],x,y,w,h

def test(path="sample/tbbt.jpg"):
    image = cv2.imread(path)
    faces= [ cv2.resize(face,(100,100),interpolation = cv2.INTER_AREA) for face,_,_,_,_ in giveAllFaces(image,BGR_output=True)]
    print("Total Faces Detected: {}".format(len(faces)))
    t = math.ceil(len(faces)/2)
    i,one = 0,[]
    while i<t:
        one.append(faces[i]);i+=1
    two = one.copy()
    while i<len(faces):
        two[i-t] = faces[i];i+=1
    plt.imshow(np.vstack([np.hstack(one),np.hstack(two)]))

def putBoxText(image,x,y,w,h,text="unknown"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image,text, (x,y-6), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def putCharacters(image,db="database"):
    dbs = os.listdir(db)
    right = np.array([ np.expand_dims(cv2.imread(os.path.join(db,x),0),-1) for x in dbs ])
    names = [ os.path.splitext(x)[0] for x in dbs ]
    for face,x,y,w,h in giveAllFaces(image):
        face = cv2.resize(face,(100,100),interpolation = cv2.INTER_AREA)
        face = np.expand_dims(face,-1)
        left = np.array([face for _ in range(len(dbs))])
        probs = np.squeeze(SiameseNet.predict([left,right]))
        index = np.argmax(probs)
        prob = probs[index]
        name = "Unknown"
        if prob>0.5:
            name = names[index]
        putBoxText(image,x,y,w,h,text=name+"({:.2f})".format(prob))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--database", dest = "database", help="Saved Image Database", required=True)
    parser.add_argument("-m", "--model", dest="model", help="Saved Model", required=True)
    parser.add_argument("-i", "--image",  nargs='+', dest="images", help="Image Paths", required=True)
    args = parser.parse_args()
    print("############ Please wait while model is loading. ###############")
    val_acc = None
    while val_acc==None: 
        try:
            SiameseNet.load_weights(args.model)
            val_acc = test_oneshot(SiameseNet,1000,verbose=0,path="../eval")
            print("Model loaded with Accuracy: {}".format(val_acc))
        except:
            print("Exception Occured: Ignoring")

    for image in args.images:
        im = cv2.imread(image,1)
        putCharacters(im,db=args.database,)
        plt.imshow(im)
        plt.show()