import numpy as np
import os
import random
import cv2
from model import SiameseNet

def getMiniBatch(batch_size=32,prob=0.5,path = "train"):
    persons = os.listdir(path)
    left = [];right = []
    target = []
    for _ in range(batch_size):
        res = np.random.choice([0,1],p=[1-prob,prob])
        if res==0:
            p1,p2 = tuple(np.random.choice(persons,size=2,replace=False))
            while len(os.listdir(os.path.join(path,p1)))<1 or len(os.listdir(os.path.join(path,p2)))<1:
                p1,p2 = tuple(np.random.choice(persons,size=2,replace=False))
            p1 = os.path.join(path,p1,random.choice(os.listdir(os.path.join(path,p1))))
            p2 = os.path.join(path,p2,random.choice(os.listdir(os.path.join(path,p2))))
            p1,p2 = np.expand_dims(cv2.imread(p1,0),-1),np.expand_dims(cv2.imread(p2,0),-1)
            left.append(p1);right.append(p2)
            target.append(0)
        else:
            p = np.random.choice(persons)
            while len(os.listdir(os.path.join(path,p)))<2:
                p = np.random.choice(persons)
            p1,p2 = tuple(np.random.choice( os.listdir(os.path.join(path,p)), size=2, replace=False ))
            p1,p2 = os.path.join(path,p,p1),os.path.join(path,p,p2)
            p1,p2 = np.expand_dims(cv2.imread(p1,0),-1),np.expand_dims(cv2.imread(p2,0),-1)
            left.append(p1);right.append(p2)
            target.append(1)
    return [np.array(left),np.array(right)],np.array(target)

def test_oneshot(model,N,verbose=0,path="eval"):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    inputs, targets = getMiniBatch(N,path=path)
    probs = model.predict(inputs)
    output = (np.squeeze(probs)>0.5)*1
    percent_correct = (output==targets).sum()*100/N
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
    return percent_correct


if __name__ == "__main__":
	evaluate_every = 7000
	loss_every = 500
	batch_size = 32
	N = 1000
	best = 0
	loss_history = []
	for i in range(0,200000):
	    (inputs,targets)= getMiniBatch(batch_size,path="train")
	    loss=SiameseNet.train_on_batch(inputs,targets)
	    loss_history.append(loss)
	    if i % loss_every == 0:
	        vloss = SiameseNet.test_on_batch(*getMiniBatch(batch_size,path="eval"))
	        print("iteration {}, training loss: {:.7f}, validation loss : {:.7f}".format(i,np.mean(loss_history),vloss))
	        loss_history.clear()
	        val_acc = test_oneshot(SiameseNet,N,verbose=True)
	        if val_acc >= best:
	            print("saving")
	            SiameseNet.save('saved_best')
	            best=val_acc