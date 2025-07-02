import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def getstuff(path, limit=2000):
    X = []
    Y = []
    d = {'cats':0,'dogs':1}
    for j in d:
        full = os.path.join(path, j)
        allimg = os.listdir(full)[:limit]
        for one in allimg:
            try:
                imgg = Image.open(full + "\\" + one)
                imgg = imgg.convert('RGB')
                imgg = imgg.resize((150,150))
                arr = np.array(imgg)
                gray = rgb2gray(arr)
                feat = hog(gray,9,(8,8),(2,2),'L2-Hys')
                X.append(feat)
                Y.append(d[j])
            except:
                pass
    return np.array(X), np.array(Y)

p = r'C:\Users\johnd\OneDrive\Desktop\internship\task3\archive\train'
a1, b1 = getstuff(p)

print(a1.shape)
print(b1.shape)
print(np.bincount(b1))

split = train_test_split(a1, b1, test_size=0.2, stratify=b1, random_state=42)
trainx = split[0]
testx = split[1]
trainy = split[2]
testy = split[3]

sc = StandardScaler()
trainx = sc.fit_transform(trainx)
testx = sc.transform(testx)

model = SVC(C=10, kernel='rbf')
model.fit(trainx, trainy)

pred = model.predict(testx)
acc = accuracy_score(testy, pred)

print("acc is", acc*100)
print("report:")
print( classification_report(testy , pred , target_names = ['Cat' , 'Dog'] ) )
