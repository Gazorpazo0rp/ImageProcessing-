import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

#EDIT THRESHOLDS HERE
t1=30 # the threshold for the frames difference
t2=2 #the arm vertial remoal pixels in the arm detection mask
t3=0.95 # selection of good matches
t4=8 #erosion and dilation kernel 
t5=300 #noise maximum height in arm detection
MIN_MATCH_COUNT=1
armWidth=50 
armMask=[[],[]]
armMask[0]=np.array([],dtype = np.uint8)
armMask[1]=np.array([],dtype = np.uint8)

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

def SiftFeaturesExtractor(image):
    # detecting the features
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(image, None)
    #print(kps)
    for k in kps:
        (x,y)=k.pt
        x=int(round(x))
        y=int(round(y))
        #print(x,y)
        #visualize
        #########cv2.circle(image,(x,y),3,255,5)
    return (kps, descs)

def FastBriefFeaturesExtractor(image):
    # detecting the features
    fast = cv2.FastFeatureDetector_create(threshold=10)
    kps = fast.detect(image,None)
    
    # descriptors with brief
    br = cv2.BRISK_create();
    kps, des = br.compute(image,  kps)
    return (kps,des)


def OrbFeaturesExtractor(image):
    # detecting the features
    orb=cv2.ORB_create()
    
    kps, des = orb.detectAndCompute(image, None)
    kpimg = cv2.drawKeypoints(image, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("LeftArmMatching", np.fliplr(kpimg))
    return (kps,des)

#arm detection here
def detectArm(lr):
    # Here starts the vertical restrization to detect the arms
    global maxArmYidx,minArmXidx,maxArmXidx,minArmYidx,armWidth,person,armMask,width,height
    #reset after each arm
    armMask[lr]=np.array([],dtype = np.uint8)
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
        elif armDown-armUp >t5: #noise
            continue
        else: # arm finished
            break
    #maxArmYidx=minArmYidx+armWidth
    if(minArmXidx > maxArmXidx):
        minArmXidx , maxArmXidx = maxArmXidx , minArmXidx 
    #armMask[lr]=person[minArmYidx:maxArmYidx,minArmXidx:maxArmXidx]
    #create the arm 2dArray
    '''Counter = 0
    Sum = 0;
    for i in range(minArmYidx,maxArmYidx+1):
        for j in range(minArmXidx,maxArmXidx+1):
            if person[i][j]==255:
                Counter+=1
                Sum+=gray2[i][j]
    if Counter!=0:
        Avg = Sum//Counter
    '''
    print("MinY: ",minArmYidx)
    print("MaxY: ",maxArmYidx)
    print("MinX: ",minArmXidx)
    print("MaxX: ",maxArmXidx)    
    for i in range(minArmYidx,maxArmYidx+1):
        for j in range(minArmXidx,maxArmXidx+1):
            
            if person[i][j]==255:
                armMask[lr]=np.append(armMask[lr],gray2[i][j])
            else:
                armMask[lr]=np.append(armMask[lr],gray2[0][0]-gray2[0][0])
    if (abs(maxArmYidx-minArmYidx)+1)*(abs(maxArmXidx-minArmXidx)+1) ==armMask[lr].size:
        armMask[lr]=armMask[lr].reshape(abs(maxArmYidx-minArmYidx)+1,abs(maxArmXidx-minArmXidx)+1)
    else:
        print("error in reshaping arm num",lr ,"  returning")
        return False
    armBorderSum=0
    counter=0
    #the following loop is meant to decrease the features outside the arm 
    for i in range(0,armMask[lr].shape[1]):
        #print(i)
        #the upper part
        for j in range(0,armMask[lr].shape[0]):
            #print(j)
            if armMask[lr][j][i]==0:
                continue
            armBorderSum+=armMask[lr][j+t2][i]
            counter+=1
            for modify in range(j, j+t2):
                #print(i,modify,"up")
                armMask[lr][modify][i]=armMask[lr][j+t2][i]
            break
        #the lower part
        for j in range(armMask[lr].shape[0]-1,0,-1):
            #print(j)
            if armMask[lr][j][i]==0:
                continue
            armBorderSum+=armMask[lr][j-2*t2][i]
            counter+=1
            for modify in range(j-2*t2,j):
                #print(i,modify,"down")
                armMask[lr][modify][i]=armMask[lr][j-2*t2][i]
            break
    borderAvg=armBorderSum//counter
    #this loop replaces the black pixels in each arm mask with the avg of the arm border pixels to eliminate bad features           
    for i in range(0,armMask[lr].shape[1]):  
        for j in range(0,armMask[lr].shape[0]):
            if armMask[lr][j][i]==0:
                armMask[lr][j][i]=borderAvg
    return True
       
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
           
def setup():    
    global gray2,person,frame2,tmp,width,height
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
    diff = cv2.absdiff(frame2, frame)
    person = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    person[person>=t1]=255
    person[person<t1]=0
    #dilation and erosion to remove noise
    kernel = np.ones((t4,t4),np.uint8)
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
    tmp = np.copy(person)
    for i in range(0,person.shape[0]):
        for j in range(0,person.shape[1]):
            tmp[i][j] = 0
            
    OP = np.ones((3,3))
    CL = np.ones((4,4))
    person = cv2.morphologyEx(person,cv2.MORPH_OPEN,OP)
    person = cv2.morphologyEx(person,cv2.MORPH_CLOSE,CL)

def main(): 
    #
    setup()
    detectArm(0)
    showDetected(person)
    print("first arm detected")
    detectArm(1)
    showDetected(person)
    print("sec arm detected")
    
    #calculate the keypoints and the descriptors of each arm
    leftArmKps,leftArmDes = SiftFeaturesExtractor(armMask[1])
    rightArmKps,rightArmDes = SiftFeaturesExtractor(armMask[0])
    # These are some validation for the features list of each arm

    if leftArmDes is None :
        #iterate untill success
        while leftArmDes is None:
            print("left arm features list is empty.. the setup will start again in 5 secs")
            time.sleep(5)
            setup()
            if detectArm(1)==True:
                leftArmKps,leftArmDes = SiftFeaturesExtractor(armMask[1])
    elif len(leftArmDes)<2:
        while leftArmDes is None:
            print("left arm features list has length less than the num of neighbours.. the setup will start again in 5 secs")
            time.sleep(5)
            setup()
            if detectArm(1)==True:
                leftArmKps,leftArmDes = SiftFeaturesExtractor(armMask[1])
    if rightArmDes is None :
        while rightArmDes is None:
            print("right arm features list is empty.. the setup will start again in 5 secs")
            time.sleep(5)
            setup()
            if detectArm(0)==True:
                rightArmKps,rightArmDes = SiftFeaturesExtractor(armMask[0])
    elif len(rightArmDes)<2:
        while rightArmDes is None:
            print("left arm features list has length less than the num of neighbours.. the setup will start again in 5 secs")
            time.sleep(5)
            setup()
            if detectArm(0)==True:
                rightArmKps,rightArmDes = SiftFeaturesExtractor(armMask[0])
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
            livekps,liveDes = SiftFeaturesExtractor(liveGray)
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
            cv2.imshow("image", np.fliplr(tmp))
            cv2.imshow("left", np.fliplr(armMask[1]))
            cv2.imshow("right", np.fliplr(armMask[0]))
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            leftArmMatches = flann.knnMatch(leftArmDes,liveDes,k=2)
            rightArmMatches = flann.knnMatch(rightArmDes,liveDes,k=2)
            #leftArmMatches = sorted(leftArmMatches, key = lambda x:x.distance)
            # Need to draw only good matches, so create a mask
            
            leftmatchesMask = [[0,0] for i in range(len(leftArmMatches))]
            rightmatchesMask = [[0,0] for i in range(len(rightArmMatches))]

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(leftArmMatches):
                if m.distance < t3*n.distance:
                    leftmatchesMask[i]=[1,0]
            for i,(m,n) in enumerate(rightArmMatches):
                if m.distance < t3*n.distance:
                    rightmatchesMask[i]=[1,0]
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = leftmatchesMask,
                               flags = 0)
            matchesLeftVisualization = cv2.drawMatchesKnn(liveGray,livekps,armMask[0],leftArmKps,leftArmMatches,None,**draw_params)
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = rightmatchesMask,
                               flags = 0)
            matchesRightVisualization = cv2.drawMatchesKnn(liveGray,livekps,armMask[1],rightArmKps,rightArmMatches,None,**draw_params)
            #plt.imshow(img3,),plt.show()
            cv2.imshow("LeftArmMatching", np.fliplr(matchesLeftVisualization))
            cv2.imshow("rightArmMatching", np.fliplr(matchesRightVisualization))
            '''
            #matching and homography

            
            # store all the good matches as per Lowe's ratio test.
            goodLeft = []
            for m,n in leftArmMatches:
                if m.distance < 0.7*n.distance:
                    goodLeft.append(m)
            goodRight = []
            for m,n in rightArmMatches:
                if m.distance < t3*n.distance:
                    goodRight.append(m)
            # now the homogram 
            if len(goodLeft)>=MIN_MATCH_COUNT:
                src_pts = np.float32([ leftArmKps[m.queryIdx].pt for m in goodLeft ]).reshape(-1,1,2)
                dst_pts = np.float32([ livekps[m.trainIdx].pt for m in goodLeft ]).reshape(-1,1,2)
            
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if not M:
                    continue
                matchesMaskLeft = mask.ravel().tolist()
            
                h,w = armMask[1].shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
            
                liveGray = cv2.polylines(liveGray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
            else:
                print ("Not enough matches are found - %d/%d" % (len(goodLeft),MIN_MATCH_COUNT))
                matchesMaskLeft = None
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMaskLeft, # draw only inliers
                   flags = 2)

            leftFinal = cv2.drawMatches(armMask[1],leftArmKps,liveGray,livekps,goodLeft,None,**draw_params)
            
            cv2.imshow("LeftArmMatching", np.fliplr(leftFinal))
            '''
            k = cv2.waitKey(1) & 0x0FF
            if k == ord('q') or k == ord('Q'):
                break
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    
    
if __name__== "__main__":
  main()    
    
    
    
