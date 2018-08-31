import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import svm,metrics

user=40
sample=8

train_data=np.zeros((user*sample,10304))
train_target=np.zeros((user*sample))

test_data=np.zeros((user*(10-sample),10304))
test_target=np.zeros((user*(10-sample)))

plt.figure(1)
for i in range(1,40):
    path='C:/Users/rajasharma/Desktop/aedifico/supervised learning/orl_face/u%d/1.png'%(i)
    im=mimg.imread(path)
    
    plt.subplot(5,8,i)
    plt.imshow(im,cmap='gray')
    plt.axis('off')
    
plt.show()

c=0
for p in range(1,user+1):
    for q in range(1,sample+1):
        path='C:/Users/rajasharma/Desktop/aedifico/supervised learning/orl_face/u%d/%d.png'%(p,q)
        v=mimg.imread(path)
        features=v.reshape(1,-1)
        train_data[c,:]=features
        train_target[c]=p
        c=c+1
        
c=0
for p in range(1,user+1):
    for q in range(sample+1,11):
        path='C:/Users/rajasharma/Desktop/aedifico/supervised learning/orl_face/u%d/%d.png'%(p,q)
        v=mimg.imread(path)
        features=v.reshape(1,-1)
        test_data[c,:]=features
        test_target[c]=p
        c=c+1
        

svm_model=svm.SVC(kernel='linear')
svm_model=svm_model.fit(train_data,train_target)

output=svm_model.predict(test_data)

acc=metrics.accuracy_score(test_target,output)
conf_mat=metrics.confusion_matrix(test_target,output)

print("accuracy: ",acc*100)
print("confusion matrix:",conf_mat)
report=metrics.classification_report(test_target,output)
print(report)