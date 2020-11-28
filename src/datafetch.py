import numpy as np
import pandas as pd
import cv2
import os
import threading
from urllib.request import urlopen

def getData(url,dirname="data",img_shape=(100,100)):
    data = pd.read_csv(url,sep="\t",skiprows=2,header=None,names=['Name','imagenum','url','rect','md5'])
    print(data.shape)
    totalrows=data.shape[0]
    total_personalities = data.Name.nunique()
    current = 0
    if not os.path.exists(dirname): os.mkdir(dirname)
    j=0
    for i in range(data.shape[0]):
        if not os.path.exists(os.path.join(dirname,data.iloc[i].Name)):
            os.mkdir(os.path.join(dirname,data.iloc[i].Name))
            current+=1
            print("{} : {}/{} {:.2f}% done".format(dirname,current,total_personalities,i*100/totalrows))
            j=0
        try:
            resp = urlopen(data.iloc[i].url,timeout=1)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
            p1,p2,p3,p4 = tuple(map(int,data.iloc[i].rect.split(',')))
            image = image[p2:p4,p1:p3]
            image = cv2.resize(image,img_shape,interpolation = cv2.INTER_AREA)
            plt.imsave(os.path.join(dirname,data.iloc[i].Name,str(j)+'.jpg'),image)
            j+=1
        except:
            pass


if __name__ == '__main__':
	data_e = threading.Thread(target = getData, 
	                           args = ('http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt', 'eval'))
	data_e.start()
	data_d = threading.Thread(target = getData, 
                           args = ('http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt', 'train'))
	data_d.start()

	data_d.join()
	data_e.join()
