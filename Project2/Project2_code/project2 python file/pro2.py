# 2. (Local Descriptor) Based on the descriptor designed in FL class, design the followings.
import sklearn.datasets as dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import math 

# data load
mnist = dataset.fetch_openml('mnist_784')
mnist_data = mnist.data.to_numpy()
mnist_target = mnist.target.to_numpy()
print('mnist data shape : ',mnist_data.shape)
print('mnist target shape : ',mnist_target.shape)

def select(data, target, number=20000):
    subset_idx = np.sort( np.random.choice(len(data), number,replace=False) )
    
    subset_data = data[subset_idx]
    subset_target = target[subset_idx]
    return subset_data, subset_target

subset_data, subset_target  = select(mnist_data, mnist_target, number=30000)

initial_centroid = []
initial_centroid.append(subset_data[np.where(subset_target=='0')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='1')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='2')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='3')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='4')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='5')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='6')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='7')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='8')[0][3]])
initial_centroid.append(subset_data[np.where(subset_target=='9')[0][3]])
initial_centroid = np.array(initial_centroid)
print(initial_centroid.shape)

# 1) Apply k-means (k=10) algorithm to MNIST test dataset, where input dimension is 28^2=784

k =10 
def kmeans(data, k, feature_size = 784, feature_scale = 256, centroid=None):
    data = data.copy()
    repeat = 1 
    if centroid == None:
        centroid = np.random.uniform(0,feature_scale,(k,feature_size)) # random initialization
    print('start')
    
    while True:
        repeat+=1
        
        # groups, ndexes
        groups, indexes = [], []
    
        # cluster 개수에 맞게 리스트 추가
        for d in range(k):
            groups.append([])
            indexes.append([])
            
        # 거리에 따라 group에 각 feature 값과 index 추가
        for i in range(len(data)):
            distances = np.array([])
            for d in range(k):
                distances = np.append(distances, np.sum((data[i] - centroid[d])**2))
            idx=np.argmin(distances)
            groups[idx].append(data[i])
            indexes[idx].append(i)

        # 각 그룹의 feature 평균값으로 다음 centroid 계산
        next_centroid = np.zeros( (k,feature_size) )
        for d in range(k):
            if len(indexes[d])!=0:
                next_centroid[d] = np.mean(np.array(groups[d]),axis=0)
        if repeat % 20 == 0:
            print('repeat:', repeat)
        
        # 종료조건
        if(np.array_equal(centroid, next_centroid)):
            print('find!')
            break
        centroid = next_centroid
    
    print('repeat:', repeat)
    return centroid, indexes

centroid, group_indexes = kmeans(subset_data, 10, 784, 256)#, initial_centroid)

    # centriod에 target 할당
# print(training_target[group_indexes[0]])
cluster_target = []
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[0]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[1]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[2]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[3]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[4]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[5]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[6]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[7]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[8]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[9]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
print(np.array(cluster_target))

# 2) Measure the clustering accuracy by counting the number of incorrectly grouped images, and analyze the results. 
count=0
for i in range(10):
    count += len(np.where(subset_target[group_indexes[i]] == str(cluster_target[i]) )[0])
print('accuracy:',100 * (count / len(subset_data)), '%')

# 3) Divide the input 28×28 image into four 14×14 sub-blocks, and compute the histogram of orientations for each sub-block as below

def devide(image):
    subMatrix = []
    subMatrix.append(image[0:14,14:28])
    subMatrix.append(image[14:28,0:14])
    subMatrix.append(image[0:14,0:14])
    subMatrix.append(image[14:28,14:28])
    subMatrix = np.array(subMatrix)
    return subMatrix

def filtering(img, kernel):   # convolution
    h, w = img.shape
    fSize = len(kernel)       # filter Size
    pSize = (fSize-1) // 2    # padding Size
    
    img = np.pad(img, ((pSize,pSize),(pSize,pSize)), 'symmetric')
    filteredImg = np.zeros((h,w))
    
    for i in range(pSize,h+pSize):      # operate on ground-truth pixel
        for j in range(pSize,w+pSize):
            product = img[i-pSize:i+pSize+1,j-pSize:j+pSize+1] * kernel
            filteredImg[i-pSize][j-pSize] = product.sum()
            
    return filteredImg

def gaussianFilter(size, std):
    kernel = np.ones((size,size))
    filterOffset = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i][j] = math.exp( -1 * ( (i-filterOffset)**2 + (j-filterOffset)**2 ) / (2*(std**2)) )
    return kernel / kernel.sum()

def orientation_histogram(img):
    #Gaussian filtering 
    Gaussian_kernel = gaussianFilter(3,3)
    img = filtering(img, Gaussian_kernel)
    
    # magnitude, direction Matrix 생성
    h, w = img.shape
    magnitude = np.zeros((h,w))
    direction = np.zeros((h,w))
    img = np.pad(img, ((1,1),(1,1)), 'symmetric')
    for i in range(h):  # y
        for j in range(w): #x
            temp_y = i+1
            temp_x = j+1
            magnitude[i,j] = np.sqrt((img[ temp_y, temp_x+1] - img[temp_y, temp_x-1])**2 + \
                                                                                (img[temp_y+1,temp_x] - img[temp_y-1,temp_x])**2)
            direction[i,j] = np.rad2deg(np.arctan2(img[temp_y+1,temp_x] - img[temp_y-1,temp_x], \
                                                                                img[temp_y, temp_x+1] - img[temp_y, temp_x-1] ))
            
    # histogram 생성
    bins =8
    width = np.rad2deg((np.pi*2 ) / bins)           # 45도
    histogram = np.zeros(bins) 
    temp_mag = magnitude.ravel()                    # 반복문 사용 편리를 위해 펼침
    direction = direction.ravel()         
    direction[ direction < 0 ] += 360               # -180 ~ 180 -> 0 ~ 360으로 변환
    hist_index = (direction // width).astype(int)   # 각 direction값을 45로 나눈 몫을 histogram index로 사용
    hist_index[hist_index==8] = 0                   # 360도 == 0도
    for i in range( len(hist_index) ):
        histogram[hist_index[i]] += temp_mag[i]     # magnitude 누적
    
    return histogram

    # feature Extraction
subset_feature = []
for i in range(len(subset_data)):
    temp = []
    subMatrix = devide(subset_data[i].reshape(28,28))
    for j in range(len(subMatrix)):
        temp.append(orientation_histogram(subMatrix[j]))
    temp = np.array(temp).ravel()
    subset_feature.append(temp)
subset_feature  = np.array(subset_feature)
print('subset_feature shape: ', subset_feature.shape )

# 4) Apply k-means (k=10) algorithm again using feature, where input dimension is 8×4=32.

centriod, group_indexes = kmeans(subset_feature, 10, feature_size = 32, feature_scale = 5000)
cluster_target = []
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[0]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[1]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[2]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[3]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[4]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[5]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[6]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[7]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[8]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
cluster_target.append(np.argmax(np.histogram((subset_target[group_indexes[9]]).astype(int),bins=[0, 1, 2, 3,4,5,6,7,8,9,10])[0]) )
print(cluster_target)
print(centriod.shape)

# 5) Measure the clustering accuracy for feature-based approach, and analyze the results.

count=0
for i in range(10):
    count += len(np.where(subset_target[group_indexes[i]] == str(cluster_target[i]) )[0])
print('accuracy:',100 * (count / len(subset_data)), '%')

