import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import cv2
from plyfile import PlyData, PlyElement
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree

BASE_DIR = './utils'
NUM_CLASS = 8

colour_code = [[204,0,0],[255,128,0],[1,255,255],[1,1,255],[102,0,204],[255,0,255],[204,204,0],[1,1,255],[128,128,128],[40,40,255]]
nan= [51,255,51]

def listdirs(rootdir):
    _list = []
    for it in os.scandir(rootdir):
        if it.is_dir():
            #print(it.path)
            _list.append(it.path)
    return _list

IMG_ROOT_PATH='./data/robotiz3d/IMG_SEG/GrayScaleImg'
ANNOT_ROOT_PATH='./data/robotiz3d/IMG_SEG/Label/poor_labelled'
TRAIN = 'train/case3'
VALID = 'valid'
TEST= 'test'

#Make a list of Chunk file
robotiz3d_path = './rdd_data/labelled/'
segmentation_map = 'segmentation_map.npy'
point_cloud = '.ply'
_dir_list = listdirs(robotiz3d_path)

def read_ply(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']

        # compute normals
        xyz = np.array([[x, y, z] for x, y, z in plydata["vertex"].data])
        #nxnynz = compute_normal(xyz, face)
        #vertices[:,6:] = nxnynz
        #cloud = PyntCloud.from_file(filename)
        #cloud.plot()
    return xyz

def read_npy(filename):
    _npy = np.load(filename)
    return _npy

def read_bin(filename):
    _bin = open(filename,"rb")
    return _bin

def xyz2rgb(xyz,_min,_max):
    g_scale = int((xyz[2]- _min) / (_max - _min) * 255)
    rgb = [g_scale,g_scale,g_scale]
    return rgb


def pc_to_gs(ply_data):
    _z_value = ply_data[:,2]
    xyz_min = np.amin(ply_data, axis=0)[0:3]
    xyz_max = np.amax(ply_data,axis=0)[0:3]
    ply_data[:, 0:3] = (ply_data[:, 0:3]-xyz_min)/(xyz_max-xyz_min)

    _x_uniq = np.unique(ply_data[:,0]) #number of x-channel
    _y_uniq = np.unique(ply_data[:,1]) #number of y-channel
    _num_x_channel = len(_x_uniq)
    _num_y_channel = len(_y_uniq)
    plate = np.zeros((_num_y_channel+1,_num_x_channel+1))
    plate[(_num_y_channel*(ply_data[:,1])).astype(int),(_num_x_channel*(ply_data[:,0])).astype(int)] = 255*_z_value
    return plate

def value_norm(data):
    _mean = np.mean(data)
    _std  =np.std(data)
    _data  = (data-_mean)/_std
    _min = np.min(_data)
    _max = np.max(_data)
    _data = (_data - _min) / (_max - _min)
    _data = 255 * _data
    _data = _data.astype(int)
    return _data,np.min(_data[np.nonzero(_data)])

def pc_to_gs_annot(ply_data,_npy_data):
    _z_value,_z_min = value_norm(ply_data[:,2])
    xyz_min = np.amin(ply_data, axis=0)[0:3]
    xyz_max = np.amax(ply_data,axis=0)[0:3]
    _x_uniq = np.unique(ply_data[:,0]) #number of x-channel
    _y_uniq = np.unique(ply_data[:,1]) #number of y-channel
    ply_data[:, 0:3] = (ply_data[:, 0:3] - xyz_min) / (xyz_max - xyz_min)
    _num_x_channel = len(_x_uniq)
    _num_y_channel = len(_y_uniq)
    plate = -1*np.ones((_num_y_channel+1,_num_x_channel+1))
    annot_plate = -1*np.ones((_num_y_channel + 1, _num_x_channel + 1,3))
    annot_npy = -1*np.ones((_num_y_channel + 1, _num_x_channel + 1))
    plate[(_num_y_channel*(ply_data[:,1])).astype(int),(_num_x_channel*(ply_data[:,0])).astype(int)] = _z_value
    plate[np.where(plate<=0)]=_z_min
    annot_plate[(_num_y_channel*(ply_data[:,1])).astype(int),(_num_x_channel*(ply_data[:,0])).astype(int),0] = _npy_data
    annot_npy[(_num_y_channel*(ply_data[:,1])).astype(int),(_num_x_channel*(ply_data[:,0])).astype(int)] = _npy_data
    #pdb.set_trace()
    # for i in range(2,8):
    #     _t = np.where(annot_plate[:,:,0]!=i)
    #     print(len(np.where(annot_plate[:,:,0]!=i)))
    # annot_plate[np.where(annot_plate[:,:,0]!=1)]=colour_code[0] #Pothole
    # annot_plate[np.where(annot_plate[:,:,0]==2)]=colour_code[1] #Manhole
    # annot_plate[np.where(annot_plate[:,:,0]==3)]=colour_code[2] #Longitudinal crack
    # annot_plate[np.where(annot_plate[:,:,0]==4)]=colour_code[3] #Transverse crack
    # annot_plate[np.where(annot_plate[:,:,0]==5)]=colour_code[4] #Joint Crack
    # annot_plate[np.where(annot_plate[:,:,0]==6)]=colour_code[5] #Wheel crack
    # annot_plate[np.where(annot_plate[:,:,0]==7)]=colour_code[6] #Alligator crack
    # annot_plate[np.where(annot_plate[:,:,0]==8)]=colour_code[7] #Block crack
    # annot_plate[np.where(annot_plate[:,:,0]==9)]=colour_code[8] #other crack
    # annot_plate[np.where(annot_plate[:,:,0]==-1)]=nan #No points - Empy space in the point cloude
    annot_plate[np.where(annot_plate[:, :, 0]>=1)] = [255,255,255]  # Pothole
    annot_plate[np.where(annot_plate[:, :, 0]==-1)] = [0, 0, 0]  # No points - Empy space in the point cloude
    annot_plate[np.where(annot_plate[:, :, 0]==0)] = [0, 0, 0] #Road class
    annot_npy[np.where(annot_npy==-1)]=255
    return plate,annot_npy,annot_plate



_point_num = np.zeros(NUM_CLASS)
try:
    #Filtering valid directory
    listdirs(_dir_list[0])
    _f_cnt = 0
    _ssibal = 0
    _sb_case=[]
    _sb_num=[]
    _sb_howmany=[]
    for _path in _dir_list:
        print(_path)
        _tmp_list = listdirs(_path)
        for _t_path in _tmp_list:
            _sub_path = os.path.join(_t_path,'profiler_data')
            _npy_file = None
            _ply_file = None
            for _ft in os.listdir(_sub_path):
                if _ft.endswith(".npy"):
                    #print(os.path.join(_sub_path, _ft))
                    _npy_file = os.path.join(_sub_path, _ft)
                if _ft.endswith(".ply"):
                    #print(os.path.join(_sub_path, _ft))
                    _ply_file = os.path.join(_sub_path, _ft)
            if _npy_file!=None and _ply_file!=None:
                _ply_data = read_ply(_ply_file)
                print(np.min(_ply_data[:,2]),np.max(_ply_data[:,2]))
                _npy_data = read_npy(_npy_file)
                # print(np.min(_npy_data), np.max(_npy_data))
                # if np.max(_npy_data)>=8:
                #     _ssibal +=1
                #     _sb_case.append(_npy_file)
                #     _sb_num.append(np.max(_npy_data))
                #     print(np.sum(_npy_data>=8))
                #     _sb_howmany.append(np.sum(_npy_data>=8))
                if len(_ply_data)!=len(_npy_data):
                    continue
                else:
                    _filename = os.path.split(_ply_file)[1].split('.')[0]
                    if np.sum(_npy_data==0)==len(_npy_data): #if a sample is only composed of road point.
                        print('[Found] - Road point only - '+_ply_file)
                        # _image,_annot, _annot_image = pc_to_gs_annot(_ply_data.copy(),_npy_data.copy())
                        # _image= cv2.medianBlur(_image.astype(np.uint8),3)
                        # _annot_image =cv2.medianBlur(_annot_image.astype(np.uint8),3)
                        # _int_z = _ply_data[:,2]
                        # #print(np.min(_npy_data),np.max(_npy_data))
                        # #print(np.min(_int_z),np.max(_int_z))
                        # _total_plate =  np.hstack((cv2.merge([_image,_image,_image]), _annot_image))
                        # #_img_file_path  = os.path.join(IMG_ROOT_PATH,'road_only',_filename+'.png')
                        # #_annot_file_path = os.path.join(ANNOT_ROOT_PATH,'road_only',_filename+'npy')
                        # cv2.imwrite('./image/v4/road_img_and_annot_%s.png' % (_filename), _total_plate)
                        # #cv2.imwrite(_img_file_path, _image)
                        # #np.save(_annot_file_path,_annot)
                        # #_road_only_file.writelines(_ply_file+'\n')
                        # # plt.subplot(1, 2, 1)
                        # # plt.imshow(_image, cmap='gray')
                        # # plt.title('image')
                        # # plt.subplot(1, 2, 2)
                        # # plt.hist(_int_z, 30, [np.min(_int_z), np.max(_int_z)])
                        # # plt.title('histogram')
                        # # plt.savefig('./image/histogram/image_and_d_hist_%s.png' % (_filename))
                        # # plt.close()


                    else:
                        #_with_defect_file.writelines(_ply_file+'\n')
                        print('[Found] - With defect - '+_ply_file)
                        _image,_annot, _annot_image= pc_to_gs_annot(_ply_data.copy(),_npy_data.copy())
                        _image= cv2.medianBlur(_image.astype(np.uint8),3)
                        _annot_image =cv2.medianBlur(_annot_image.astype(np.uint8),3)
                        _total_plate =  np.hstack((cv2.merge([_image,_image,_image]), _annot_image))
                        #_int_z = _ply_data.astype(int)
                        _int_z = _ply_data[np.where(_npy_data==0), 2]
                        _int_z_abnormal = _ply_data[np.where(_npy_data > 0), 2]
                        _img_file_path  = os.path.join(IMG_ROOT_PATH,TRAIN,_filename+'.png')
                        #_annot_file_path = os.path.join(ANNOT_ROOT_PATH,TRAIN,_filename+'.npy')
                        #print(np.min(_npy_data),np.max(_npy_data))
                        cv2.imwrite('./image/img/%s_img.png' % (_filename), _image)
                        cv2.imwrite(_img_file_path, _image)
                        cv2.imwrite('./image/annot/%s_annot.png' % (_filename), _annot_image)
                        cv2.imwrite(_img_file_path, _image)

                        #np.save(_annot_file_path,_annot)
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(_image, cmap='gray')
                        # plt.title('image')
                        # plt.subplot(1, 2, 2)
                        # plt.hist(_int_z[0], 30, [np.min(_ply_data[:,2]), np.max(_ply_data[:,2])],label=['road'])
                        # plt.hist(_int_z_abnormal[0], 30,[np.min(_ply_data[:, 2]), np.max(_ply_data[:, 2])], label=['defect'])
                        # plt.legend(loc='upper right')
                        # plt.title('histogram')
                        # plt.savefig('./image/histogram/image_and_d_hist_%s.png' % (_filename))
                        # plt.close()
                _f_cnt += 1
    # print('total ssibal case=%d'%(_ssibal))
    # for i,_path in enumerate(_sb_case):
    #     print(_path)
    #     print(_sb_num[i])
    #     print(_sb_howmany[i])
except:
    print('out')