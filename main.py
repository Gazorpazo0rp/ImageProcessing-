import cv2
import sys
import numpy as np
import time
from matplotlib import pyplot as plt

#EDIT THRESHOLDS HERE
t1=10 # the threshold for the frames difference
armWidth=70 
maxArmYidx= maxArmXidx =minArmYidx =0
minArmXidx=1000 #infinity



def extractHog(image):

    winSize = (image.shape[0],image.shape[1]) # Image Size
    cellSize = (4,4) #Size of one cell    
    blockSizeInCells = (2,2)# will be multiplied by No of cells
    blockSize=(blockSizeInCells[1] * cellSize[1], blockSizeInCells[0] * cellSize[0])
    blockStride=(cellSize[1], cellSize[0])
    nbins = 9 #Number of orientation bins
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) # 
    h = hog.compute(image)
    h = h.flatten()
    return h.flatten()

def SiftFeatureExtractors(image):
    # detecting the features
    sift = cv2.xfeatures2d.SIFT_create()
    
    (kps, descs) = sift.detectAndCompute(image, None)
    #print(kps)
    
    
    for k in kps:
        (x,y)=k.pt
        
        x=int(round(x))
        y=int(round(y))
        print(x,y)
        #visualize
        #########cv2.circle(image,(x,y),3,255,5)
        
    
    

    return (kps, descs)


armMask=[[],[]]
armMask[0]=np.array([],dtype = np.uint8)
armMask[1]=np.array([],dtype = np.uint8)

#arm detection here
def detectArm(lr):
    global maxArmYidx,minArmXidx,maxArmXidx,minArmYidx,armWidth,person,armMask
    #reset after each arm
    maxArmYidx= maxArmXidx  =0
    minArmXidx =minArmYidx=1000 #infinity
    if(lr) :
        start=1
        end=width-1
        step=1
    else:
        start=width-1
        end=1
        step=-1
    for j in range (start,end,step):
        armUp=0 #initialized tto max indicating that this line intersects no arms
        armDown=height #initialized tto max indicating that this line intersects no arms
        i=width-j #will be deleted!!!!!!!!!!!!!!!
        for k in range (0,height-1):
            if person[k][i]==255:
                armUp=k
                break
        for k in range (1,height):
            m=height-k
            if person[m][i]==255:
                armDown=m
                break
        if armDown-armUp<armWidth: #in the arm
            
            maxArmXidx=max(maxArmXidx,i)
            minArmXidx=min(minArmXidx,i)
            if armUp<minArmYidx:
                minArmYidx=armUp
            if armDown>maxArmYidx:
                maxArmYidx=armDown
               
            #this will now work if any noise is in the image so we MUST apply min filter
        elif armDown-armUp >300:
            continue
        else:
            
            break
    #maxArmYidx=minArmYidx+armWidth
    if(minArmXidx > maxArmXidx):
        minArmXidx , maxArmXidx = maxArmXidx , minArmXidx 
    #armMask[lr]=person[minArmYidx:maxArmYidx,minArmXidx:maxArmXidx]
    #create the arm 2dArray
    Counter = 0
    Sum = 0;
    for i in range(minArmYidx,maxArmYidx+1):
        for j in range(minArmXidx,maxArmXidx+1):
            if person[i][j]==255:
                Counter+=1
                Sum+=gray2[i][j]
    Avg = Sum//Counter
    for i in range(minArmYidx,maxArmYidx+1):
        for j in range(minArmXidx,maxArmXidx+1):
            if person[i][j]==255:
                armMask[lr]=np.append(armMask[lr],frame2[i][j])
            else:
                armMask[lr]=np.append(armMask[lr],frame2[0][0]-frame2[0][0])
    armMask[lr]=armMask[lr].reshape(abs(maxArmYidx-minArmYidx)+1,abs(maxArmXidx-minArmXidx)+1,3)
        
    print("MinY: ",minArmYidx)
    print("MaxY: ",maxArmYidx)
    print("MinX: ",minArmXidx)
    print("MaxX: ",maxArmXidx)       
def showDetected(person):    
    # the following loop prints the extracted arm mask
    i=j=0
    for i in range (0,tmp.shape[0]):
        for j in range(0,tmp.shape[1]):
            if i>minArmYidx and i<=maxArmYidx and j>minArmXidx and j<=maxArmXidx:
                #we need to flip the graylevel here
                #gray2=np.fliplr(gray2)
                tmp[i][j]=gray2[i][j]
                #print(person[i][j])
           

#camera captures
cap = cv2.VideoCapture(0)
i=0
while i<10:
    ret, frame = cap.read()
    i=i+1
    if not ret:
        continue
gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
height, width = np.shape(gray1)   
cv2.destroyAllWindows()
time.sleep(5)  
i=0
while i<10:
    i=i+1
    ret2, frame2 = cap.read()
    if not ret2:
        continue
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
person = cv2.absdiff(gray2, gray1)
person[person>=t1]=255
person[person<t1]=0
#dilation and erosion to remove noise
kernel = np.ones((8,8),np.uint8)
erodedImage = cv2.erode(person,kernel,iterations = 1)
dilated = cv2.dilate(erodedImage,kernel,iterations = 1)

while True:
    cv2.imshow("img", np.fliplr(dilated))
    cv2.imshow("original", np.fliplr(frame2))
    k = cv2.waitKey(1) & 0x0FF
    if k == ord('q') or k == ord('Q'):
        break

person=dilated


cap.release()
cv2.destroyAllWindows()


# Here starts the vertical restrization to detect the arms
# I'll start with the left arm on the right of the screen
# initialize average arm height as 50px for now we should make it a function in depth later
# left arm detection // right part of the screen



tmp = np.copy(person)

for i in range(0,person.shape[0]):
    for j in range(0,person.shape[1]):
        tmp[i][j] = 0
        
OP = np.ones((3,3))
CL = np.ones((4,4))
person = cv2.morphologyEx(person,cv2.MORPH_OPEN,OP)
person = cv2.morphologyEx(person,cv2.MORPH_CLOSE,CL)


detectArm(0)
showDetected(person)

detectArm(1)
showDetected(person)


leftArmKps,leftArmDes = SiftFeatureExtractors(armMask[0])
rightArmKps,rightArmDes = SiftFeatureExtractors(armMask[1])
#el hog
'''
print(armMask[0].shape)
leftArmFeatures=extractHog(np.array(armMask[0]))
rightArmFeatures=extractHog(np.array(armMask[1]))
print(leftArmFeatures)
'''
cap = cv2.VideoCapture(0)

while True:
    #in each frame we extract the body, then get the body features then match them
    liveRet, liveFrame = cap.read()
    if liveRet:
        

        liveGray= cv2.cvtColor(liveFrame, cv2.COLOR_BGR2GRAY)
        livekps,Livedes = SiftFeatureExtractors(liveGray)
        #create matcher obj
        
        '''
        #Brute force matcher:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(Livedes,leftArmDes, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])
        #print(matches)
        img3 = cv2.drawMatchesKnn(liveFrame,livekps,armMask[0],leftArmKps,good,None,flags=2)
        
        
        plt.imshow(img3),plt.show()
        '''
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(Livedes,leftArmDes,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        
        img3 = cv2.drawMatchesKnn(liveGray,livekps,armMask[0],leftArmKps,matches,None,**draw_params)
        
        plt.imshow(img3,),plt.show()
        
        
        
        
        cv2.imshow("image", np.fliplr(tmp))
        cv2.imshow("left", np.fliplr(armMask[0]))
        cv2.imshow("right", np.fliplr(armMask[1]))
        #cv2.imshow("image with kps", np.fliplr(features))

    k = cv2.waitKey(1) & 0x0FF
    if k == ord('q') or k == ord('Q'):
        break


cap.release()
cv2.destroyAllWindows()








