import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import string

img_height = 240
img_width = 320
left_start_x=284
left_start_y=671
right_start_x= 284
right_start_y=671+320


left_gaze_y=955
right_gaze_y=1090
gaze_x=[95,118,141]
gaze_width = 65
gaze_height= 18

name_=[0,1,2,3,4,5]

def ShowVideo(path,savepath):
    if not os.path.exists(os.path.join(savepath,'left')):
        os.makedirs(os.path.join(savepath,'left'))
    if not os.path.exists(os.path.join(savepath,'right')):
        os.makedirs(os.path.join(savepath,'right'))

    if not os.path.exists(os.path.join(savepath+'/left','x')):
        os.makedirs(os.path.join(savepath+'/left','x'))
    if not os.path.exists(os.path.join(savepath+'/left','y')):
        os.makedirs(os.path.join(savepath+'/left','y'))
    if not os.path.exists(os.path.join(savepath+'/left','z')):
        os.makedirs(os.path.join(savepath+'/left','z'))

    if not os.path.exists(os.path.join(savepath+'/right','x')):
        os.makedirs(os.path.join(savepath+'/right','x'))
    if not os.path.exists(os.path.join(savepath+'/right','y')):
        os.makedirs(os.path.join(savepath+'/right','y'))
    if not os.path.exists(os.path.join(savepath+'/right','z')):
        os.makedirs(os.path.join(savepath+'/right','z'))

    cap = cv2.VideoCapture(path)
    count = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(ret!=True):
            break;
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left = gray[left_start_x:left_start_x+img_height,left_start_y:left_start_y+img_width]
        right = gray[right_start_x:right_start_x+img_height,right_start_y:right_start_y+img_width]

        left_gaze_img_x=gray[gaze_x[0]:gaze_x[0]+gaze_height,left_gaze_y:left_gaze_y+gaze_width]
        left_gaze_img_y =gray[gaze_x[1]:gaze_x[1]+gaze_height,left_gaze_y:left_gaze_y+gaze_width]
        left_gaze_img_z =gray[gaze_x[2]:gaze_x[2]+gaze_height,left_gaze_y:left_gaze_y+gaze_width]

        right_gaze_img_x=gray[gaze_x[0]:gaze_x[0]+gaze_height,right_gaze_y:right_gaze_y+gaze_width]
        right_gaze_img_y =gray[gaze_x[1]:gaze_x[1]+gaze_height,right_gaze_y:right_gaze_y+gaze_width]
        right_gaze_img_z =gray[gaze_x[2]:gaze_x[2]+gaze_height,right_gaze_y:right_gaze_y+gaze_width]

        cv2.imwrite(savepath+'/left/'+str(count)+'.png',left)
        cv2.imwrite(savepath+'/right/'+str(count)+'.png',right)

        cv2.imwrite(savepath + '/left/x/' + str(count) + '.png', left_gaze_img_x)
        cv2.imwrite(savepath + '/left/y/' + str(count) + '.png', left_gaze_img_y)
        cv2.imwrite(savepath + '/left/z/' + str(count) + '.png', left_gaze_img_z)

        cv2.imwrite(savepath + '/right/x/' + str(count) + '.png', right_gaze_img_x)
        cv2.imwrite(savepath + '/right/y/' + str(count) + '.png', right_gaze_img_y)
        cv2.imwrite(savepath + '/right/z/' + str(count) + '.png', right_gaze_img_z)
        count+=1
        # Display the resulting frame
        #cv2.imshow('frame', gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('frames done')
def FoveUndistort(path):

    left_mtx = np.zeros((3,3))
    left_mtx[0, 0] = 1565.45007081745
    left_mtx[1, 1] = 1849.89862768801
    left_mtx[2, 2] = 1.0
    left_mtx[0, 2] = 166.001721474211
    left_mtx[1, 2] = 107.786531289313

    left_dist = np.zeros((1, 5))
    left_dist[0, 0] = 18.5810242354530
    left_dist[0, 1] = 4638.76366783197
    left_dist[0, 2] = -0.522551569767859
    left_dist[0, 3] = 0.0809710498045889
    left_dist[0, 4] = - 366327.678391643


    right_mtx = np.zeros((3, 3))
    right_mtx[0, 0] =  987.180646325560
    right_mtx[1, 1] =    1370.27142585013
    right_mtx[2, 2] = 1.0
    right_mtx[0, 2] =     158.462410801839
    right_mtx[1, 2] =     204.764716179780
    right_dist = np.zeros((1, 5))
    right_dist[0, 0] = 6.29819960784428
    right_dist[0, 1] =  1344.56017730721
    right_dist[0, 2] =     0.750591061146256
    right_dist[0, 3] =    0.0210036652309011
    right_dist[0, 4] = - 41383.6903131919
    extra = ['left/','right/']

    for j in range(len(extra)):
        if(extra[j]=='left/'):
            mtx  = left_mtx
            dist = left_dist
        if(extra[j]=='right/'):
            mtx=right_mtx
            dist = right_dist

        list = glob.glob(path + '/' + extra[j] + '/' + '*.png')

        for i in range(len(list)):

            img = cv2.imread(list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            cv2.imwrite(list[i], dst)
    print('undistort done')
def MakeVideo(path,start,end,label):
    LeftMat = FittingEillps(108.137,178.496,89.531,104.031)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('FoveVideo/extraction/'+label+'.avi',fourcc,30.0,(320,240))
    for i in range(start,end):
        frame = cv2.imread(path+'/'+str(i)+'.png')
        frame = Fill(frame,LeftMat)
        # out.write(frame)
        cv2.imwrite(path+'/'+str(i)+'.png',frame)
    # out.release()


def FittingEillps(xc,yc,xl,yl):
    Mat = np.zeros((240,2))
    Mat-=1
    for i in range(240):
        x =i
        det = 1-np.power( x-xc,2 )/np.power(xl,2)
        if(det<=0):
            continue;
        det*=np.power(yl,2)
        det = np.sqrt(det)
        Mat[i,0]=yc-det
        Mat[i,1]=yc+det
    return np.array(Mat,dtype=np.int)

def Fill(Frame,Mat):
    value=254
    for i in range(240):
        if(Mat[i,0]==-1):
            Frame[i,:]=[value,value,value]
            continue;
        Frame[i,0:Mat[i,0]]=[value,value,value]
        Frame[i,Mat[i,1]:320]=[value,value,value]
    return Frame

#MakeVideo('IREye/0109/Right/',190,1500,'Right')

# def thresholding():
#     img = cv2.imread('tmp.png', 0)
#     img = cv2.medianBlur(img, 5)
#     ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                                 cv2.THRESH_BINARY, 11, 2)
#     th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                 cv2.THRESH_BINARY, 11, 2)
#     titles = ['Original Image', 'Global Thresholding (v = 127)',
#               'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#     images = [img, th1, th2, th3]
#     for i in xrange(4):
#         plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
#     plt.show()
# thresholding()

def CheckDarkest():
    path = 'IREye/0109/Darkest_Point/'+'Left.txt'
    tmp = np.loadtxt(path,dtype=np.int)
    region = FittingEillps(108.137,178.496,89.531,104.031)
    legal=[]
    for i in range(len(tmp)):
        if(tmp[i,0]>region[tmp[i,1],0] and tmp[i,0]<region[tmp[i,1],1]):
            legal.append(i)
    np.savetxt('IREye/0109/Darkest_Point/'+'legal_left.txt',np.array(legal))

def flip(path):
    list = glob.glob(path+'/right/'+'*.png')
    for i in range(len(list)):
        tmp= cv2.imread(list[i])
        tmp = np.flip(tmp,1)
        cv2.imwrite(list[i],tmp)
    print('flip done')


def Crop(path):
    extra = ['left/','right/']

    for j in range(len(extra)):

        list = glob.glob(path+'/'+extra[j]+'/'+'*.png')
        if(extra[j]=='left/'):
            xc, yc, xl, yl =108.137, 178.496, 89.531, 104.031
        else:
            xc, yc, xl, yl = 107.126, 149.538, 79.046, 90.721
        Mat = FittingEillps(xc, yc, xl, yl)
        for i in range(len(list)):
            tmp= cv2.imread(list[i])
            frame = Fill(tmp, Mat)
            # out.write(frame)
            cv2.imwrite(list[i], frame[int(xc-xl):int(xc+xl),int(yc-yl):int(yc+yl)])
    print( "Crop done")
def Resize(path):
    list = glob.glob(path + '/left/' + '*.png')
    for i in range(len(list)):
        tmp = cv2.imread(list[i])
        frame = cv2.resize(tmp,(182,158))
        # out.write(frame)
        cv2.imwrite(list[i], frame)
    print("Resize done")

s=6
s=str(s)
save_path='Data/'+s+'/'
ShowVideo('Raw_data/S'+s+'.mp4',save_path)
FoveUndistort(save_path)
flip(save_path)
Crop(save_path)
Resize(save_path)



def debug():
    path = 'IREye/0118/sgx/segmentation/tmp'
    save="IREye/0118/sgx/segmentation/tmp"
    for i in range(3278,4376):
            tmp=cv2.imread(path+'/Label_'+str(i)+'.png',0)
            if(tmp.shape==(158, 182)):
                cv2.imwrite(os.path.join(save,str(i-1)+'.tif'),tmp)



