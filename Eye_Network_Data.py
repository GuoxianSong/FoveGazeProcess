import cv2
import numpy as np
import shutil
import glob

input_width=182
input_height =158
augmentation_time =10

crop_width =  167
crop_height = 151
subject=[3886, 6062, 5276, 5503, 5777, 3391, 5389]
new_crop_width = 148


def ExtractData(path):

    side =['right','left']

    for i in range(len(side)):
        raw_label = np.loadtxt(path+'/'+side[i]+'_ellipse.txt')
        data = np.zeros((len(raw_label),input_height,input_width))
        truth_gaze= np.loadtxt(path+'ground_truth_'+side[i]+'_gaze.txt')
        if(i==1):
            truth_gaze[:,1]=truth_gaze[:,1]*(-1)
        label = np.zeros((len(raw_label),5))
        truth_gaze_save = np.zeros((len(raw_label),2))
        index=0
        for j in range(len(raw_label)):
            if(sum(raw_label[j,:])==0):
                continue;
            if(truth_gaze[j,0]==0):
                continue;
            img = cv2.imread(path+'/'+side[i]+'/'+str(j+1)+'.png',0)
            print (path+'/'+side[i]+'/'+str(j+1)+'.png')
            data[index,:,:] =img
            label[index,:] =raw_label[j,0:5]
            truth_gaze_save[index, 0] = np.arccos(truth_gaze[j,2])/np.pi*180
            if(truth_gaze[j, 0]>0):
                truth_gaze_save[index, 1] = np.arccos(truth_gaze[j, 1]) / np.pi * 180
            else:
                truth_gaze_save[index, 1] = -np.arccos(truth_gaze[j, 1]) / np.pi * 180
            index+=1
        data=data[:index,:,:]
        label = label[:index,:]
        truth_gaze_save=truth_gaze_save[:index,:]
        np.save(path+side[i]+'_data.npy',data)
        np.save(path+side[i]+'_label.npy',label)
        np.save(path + side[i] + '_truth_gaze.npy', truth_gaze_save)

def CropFrame(img, label):
    out = label
    delta_w=np.random.randint(6,size=1)[0]
    delta_h = np.random.randint(4,size=1)[0]

    out[3]=label[3]-delta_w
    out[4]=label[4]-delta_h
    return img[delta_h:delta_h+crop_height,delta_w:delta_w+crop_width],out

def Augmentation_Crop(path):
    side = ['right', 'left']
    for i in range(len(side)):
        data = np.load(path+side[i]+'_data.npy')
        label = np.load(path+side[i]+'_label.npy')
        true_gaze = np.load(path+side[i]+'_truth_gaze.npy')
        augmenta_data = np.zeros((augmentation_time* len(data),crop_height,crop_width))
        augmenta_label= np.zeros((augmentation_time* len(data),5))
        compare_optimaztion= np.zeros((len(data),4))
        for j in range(len(data)):
            img = data[j,:,:]
            for s in range(augmentation_time):
                delta_w = np.random.randint(6, size=1)[0]
                delta_h = np.random.randint(4, size=1)[0]
                augmenta_data[augmentation_time*j+s,:,:]=img[delta_h:delta_h+crop_height,delta_w:delta_w+crop_width]
                augmenta_label[augmentation_time * j + s, :]=label[j,:]
                augmenta_label[augmentation_time * j + s, 3]=augmenta_label[augmentation_time * j + s, 3]-delta_w
                augmenta_label[augmentation_time * j + s, 4] =augmenta_label[augmentation_time * j + s, 4] - delta_h
                augmenta_label[augmentation_time * j + s, :] = NewLabel(augmenta_label[augmentation_time * j + s, :])
                if(s==0):
                    compare_optimaztion[j, 0] = true_gaze[j, 0]
                    compare_optimaztion[j, 1] = true_gaze[j, 1]
                    compare_optimaztion[j, 2] = augmenta_label[augmentation_time * j + s, 0]
                    compare_optimaztion[j, 3] = augmenta_label[augmentation_time * j + s, 1]

                augmenta_label[augmentation_time * j + s,1] = true_gaze[j, 0]
                augmenta_label[augmentation_time * j + s,2] = true_gaze[j, 1]

        np.save(path+'/'+side[i]+'_Adata.npy',np.array(augmenta_data,dtype=np.uint16))
        np.save(path+'/'+side[i]+'_Alabel.npy',augmenta_label)
        np.save(path + '/' + side[i] + '_compare_gaze.npy', compare_optimaztion)
def Package():
    source_path = 'Data/'
    save_path ='NetworkInput/Pickage/'
    first   =True
    side = ['right', 'left']
    data=[]
    label=[]
    for i in range(7):
        for j in side:
            if(i ==6 and j=='right'):
                data_ = np.load(source_path + str(i)+'/'+ j + '_Adata.npy')
                label_ = np.load(source_path + str(i)+'/' + j + '_Alabel.npy')
                np.save(save_path + '0220_Test_data.npy', data_)
                np.save(save_path + '0220_Test_label.npy', label_)
                continue;
            
            if(first):
                data = np.load(source_path+str(i)+'/'+j+'_Adata.npy')
                label = np.load(source_path+str(i)+'/'+j+'_Alabel.npy')
                first = False
            else:
                tmp = np.load(source_path+str(i)+'/'+j+'_Adata.npy')
                tmp_label = np.load(source_path+str(i)+'/'+j+'_Alabel.npy')
                data = np.concatenate((data, tmp), axis=0)
                label =np.concatenate((label, tmp_label), axis=0)
    np.save(save_path+'0220_Train_data.npy',data)
    np.save(save_path + '0220_Train_label.npy', label)

def Check_std(std):
    [m,n]  = std.shape
    small = 0.000001
    for i in range(m):
        for j in range(n):
            if(std[i,j]==0):
                std[i,j] = small
    return std

def Normalize():
    data  = np.concatenate((np.load('Pickage/0220_Train_data.npy'),np.load('Pickage/0220_Test_data.npy')), axis=0)
    data  = data/255.0
    data = np.array(data,dtype=np.float32)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_std = Check_std(data_std)
    np.save('Pickage/'+'data_mean.npy',data_mean)
    np.save('Pickage/'+'data_std.npy',data_std)

    label = np.concatenate((np.load('Pickage/0220_Train_label.npy'),np.load('Pickage/0220_Test_label.npy')), axis=0)
    label_mean = np.mean(label, axis=0)
    label_std = np.std(label, axis=0)

    np.save('Pickage/'+'label_mean.npy',label_mean)
    np.save('Pickage/'+'label_std.npy',label_std)

def CopyFile(filename):
    shutil.copy2('NetworkInput/augmentation/'+filename,'NetworkInput/Pickage/'+filename)

def check_label():
    path = 'NetworkInput/'
    side = ['right', 'left']
    for i in namelist:
        for j in side:
            file = path+i+'_'+j+'_label.npy'
            label = np.load(file)
            print( i)
            print (j)
            print (np.min(label))

def NewLabel(parameter):
    new_parameter= np.zeros((1,5))
    x = parameter[0]
    y = parameter[1]
    z = parameter[3]
    if (x > y):
        new_parameter[0,0]=x
        A = y / x
        B = 0;
    else:
        A = x / y
        new_parameter[0, 0] = y
        if (z < crop_width / 2.0):
            B = 3.14159 / 2.0
        else:
            B = -3.14159 / 2.0
    new_parameter[0, 1] = np.arccos(A)/np.pi*180
    new_parameter[0, 2] = CheckAngle(-B - parameter[2])/np.pi*180
    new_parameter[0,3] = parameter[3]
    new_parameter[0,4] = parameter[4]
    return new_parameter

def CheckAngle(value):
    if(value<-np.pi):
        value+=2*np.pi
    elif(value>np.pi):
        value-=2*np.pi
    return value

def GazeError(filename):
    gaze = np.load(filename)
    truth = gaze[:,:2]
    optimaztion = gaze[:,2:]
    value=0
    for i in range(len(truth)):
        angle_x = truth[i, 0] / 180.0 * np.pi
        angle_z = truth[i, 1] / 180.0 * np.pi
        rotation_x = np.matrix(
            [[1.0, 0, 0], [0.0, np.cos(angle_x), -np.sin(angle_x)], [0.0, np.sin(angle_x), np.cos(angle_x)]])
        rotation_z = np.matrix(
            [[np.cos(angle_z), -np.sin(angle_z), 0.0], [np.sin(angle_z), np.cos(angle_z), 0.0], [0, 0, 1.0]])
        vec0 = np.matrix([0, 0, 1.0] * rotation_x * rotation_z)

        angle_x = optimaztion[i, 0] / 180.0 * np.pi
        angle_z = optimaztion[i, 1] / 180.0 * np.pi
        rotation_x = np.matrix(
            [[1.0, 0, 0], [0.0, np.cos(angle_x), -np.sin(angle_x)], [0.0, np.sin(angle_x), np.cos(angle_x)]])
        rotation_z = np.matrix(
            [[np.cos(angle_z), -np.sin(angle_z), 0.0], [np.sin(angle_z), np.cos(angle_z), 0.0], [0, 0, 1.0]])
        vec1 = np.matrix([0, 0, 1.0] * rotation_x * rotation_z)
        value += np.arccos(
            np.dot([vec0[0, 0], vec0[0, 1], vec0[0, 2]], [vec1[0, 0], vec1[0, 1], vec1[0, 2]])) / np.pi * 180
    print(value / float(len(truth)))
#GazeError()
# for i in range(7):
#     for s in ['right', 'left']:
#         path = "Data/"+str(i)+"/"+s+'_compare_gaze.npy'
#         GazeError(path)
#         print (path)
#Package()
# Normalize()


def New_data_normalization_corner():
    data_path = 'Data/'
    eye_corner= np.loadtxt('Eye_corner.txt')

    test_path ='gaze_sample/'
    list = glob.glob(test_path + '*.png')
    for i in range(len(list)):
        img = cv2.imread(list[i],0)
        if(i%2==0):
            corner = eye_corner[int(i/2),0:4]
        else:
            corner = eye_corner[int(i/2),4:8]

        scale = new_crop_width/( corner[2]-corner[0] )
        res = cv2.resize(img,None,fx=scale,fy=scale)

        base_x =int((corner[0]+corner[2])/2* scale)
        base_y =int((corner[1]+corner[3])/2* scale)
        tmp = res[base_y- 75:base_y+15 ,base_x-74:base_x+74]
        cv2.imwrite('tmp/'+str(i)+'.png',tmp)

New_data_normalization_corner()