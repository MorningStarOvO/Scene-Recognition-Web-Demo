# -*- coding: utf-8 -*
"""
    本代码用于: 运行 unified 的「场景识别 demo」
    创建时间: 2022 年 02 月 08 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 02 月 08 日
    参考代码: https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
    具体功能: 预测 scene category、attribute、class activation map
"""
# ==================== 导入必要的包 ==================== #
# ----- 减少版本影响 ----- # 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ----- 系统操作相关的包 ----- #
import time
import sys
import os 

# ----- 数据处理相关的包 ----- #
import numpy as np 

# ----- 模型创建相关的包 ----- # 
import torch 
from torch.autograd import Variable as V 
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

# ----- 处理图像相关的包 ----- # 
import cv2
from PIL import Image



# ==================== 设置常量参数 ==================== #
DATA_DIR = "data"
MODEL_PATH = "checkpoint"
DEMO_CAM_PATH = "static/CAM"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==================== 函数实现 ==================== #
# ---------- hacky way to deal with the Pytorch 1.0 update ---------- #
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

# ---------- prepare all the labels ---------- #
def load_labels():
    # scene category relevant
    file_name_category = os.path.join(DATA_DIR, 'categories_places365.txt')
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget -P ' + DATA_DIR + ' '  + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = os.path.join(DATA_DIR, 'IO_places365.txt')
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget -P ' + DATA_DIR + ' '  + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = os.path.join(DATA_DIR, 'labels_sunattribute.txt')
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget -P ' + DATA_DIR + ' '  + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = os.path.join(DATA_DIR, 'W_sceneattribute_wideresnet18.npy')
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget -P ' + DATA_DIR + ' '  + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = os.path.join(MODEL_PATH, 'wideresnet18_places365.pth.tar')
    if not os.access(model_file, os.W_OK):
        os.system('wget -P ' + MODEL_PATH + ' ' + 'http://places2.csail.mit.edu/models_places365/' + 'wideresnet18_places365.pth.tar')
        # os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

def Scene_Recognition(img_path):
    # ----- 开始计时 ----- #
    T_Start = time.time()

    # ----- load the labels ----- #
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # ----- load the model ----- #
    global features_blobs
    features_blobs = []
    model = load_model()

    # ----- load the transformer ----- # 
    tf = returnTF() # image transformer

    # ----- get the softmax weight ----- #
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    # ----- load the test image ----- #
    # img_url = 'http://places.csail.mit.edu/demo/12.jpg'
    # img_str = img_url.split('/')[-1].split('.')[0]
    # os.system('wget %s -q -O test.jpg' % img_url)
    img = Image.open(img_path)
    input_img = V(tf(img).unsqueeze(0))

    # ----- forward pass ----- #
    logit = model.forward(input_img)
    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # ----- output the IO prediction ----- #
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        environment = "indoor"
    else:
        environment = "outdoor"

    # ----- output the prediction of scene category ----- #
    # print('--SCENE CATEGORIES:')
    categories = ""
    for i in range(0, 5):
        # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        categories += '{}({:.3f}), '.format(classes[idx[i]], probs[i])

    # ----- output the scene attributes ----- #
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    # print('--SCENE ATTRIBUTES:')
    # print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))
    scene_attributes = ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)])

    # ----- generate class activation mapping ----- #
    # print('Class activation map is saved as cam.jpg')
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # ----- render the CAM and output ----- # 
    img = cv2.imread(img_path)
    img_str = img_path.split("/")[-1]
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite(os.path.join(DEMO_CAM_PATH, img_str + '-cam.jpg'), result)
    img_CAM = os.path.join(DEMO_CAM_PATH, img_str + '-cam.jpg')

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)

    return environment, categories, scene_attributes, img_CAM, T_Second


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 ! \n")
    print("系统运行环境: ", sys.executable)
    print("")

    environment, categories, scene_attributes, img_CAM, T_Second = Scene_Recognition("test.jpg")
    print(environment, categories, scene_attributes, img_CAM, T_Second)

    # # ----- load the labels ----- #
    # classes, labels_IO, labels_attribute, W_attribute = load_labels()
    # # print("classes: \n", classes, '\n')
    # # print("labels_IO: \n", labels_IO, '\n')
    # # print("labels_attribute: \n", labels_attribute, '\n')
    # # print("W_attribute: \n", W_attribute, '\n')

    # # ----- load the model ----- #
    # features_blobs = []
    # model = load_model()

    # # ----- load the transformer ----- # 
    # tf = returnTF() # image transformer

    # # ----- get the softmax weight ----- #
    # params = list(model.parameters())
    # weight_softmax = params[-2].data.numpy()
    # weight_softmax[weight_softmax<0] = 0

    # # ----- load the test image ----- #
    # img_url = 'http://places.csail.mit.edu/demo/12.jpg'
    # img_str = img_url.split('/')[-1].split('.')[0]
    # os.system('wget %s -q -O test.jpg' % img_url)
    # img = Image.open('test.jpg')
    # input_img = V(tf(img).unsqueeze(0))

    # # ----- forward pass ----- #
    # logit = model.forward(input_img)
    # h_x = F.softmax(logit).data.squeeze()
    # probs, idx = h_x.sort(0, True)
    # probs = probs.numpy()
    # idx = idx.numpy()

    # print('RESULT ON ' + img_url)

    # # ----- output the IO prediction ----- #
    # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    # if io_image < 0.5:
    #     print('--TYPE OF ENVIRONMENT: indoor')
    # else:
    #     print('--TYPE OF ENVIRONMENT: outdoor')

    # # ----- output the prediction of scene category ----- #
    # print('--SCENE CATEGORIES:')
    # for i in range(0, 5):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # # ----- output the scene attributes ----- #
    # responses_attribute = W_attribute.dot(features_blobs[1])
    # idx_a = np.argsort(responses_attribute)
    # print('--SCENE ATTRIBUTES:')
    # print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))


    # # ----- generate class activation mapping ----- #
    # print('Class activation map is saved as cam.jpg')
    # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # # ----- render the CAM and output ----- # 
    # img = cv2.imread('test.jpg')
    # height, width, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    # result = heatmap * 0.4 + img * 0.5
    # cv2.imwrite(os.path.join(DEMO_CAM_PATH, img_str + '-cam.jpg'), result)

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 !")