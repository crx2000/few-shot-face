import json
import os
from glob import glob
import random
import shutil
import cv2
# 单张图片，镜像生成一个
def imgpre(img_root_path,out_mpath,out_withoutmpath):
    namedirs = os.listdir(img_root_path)

    for dir in namedirs:
        curdir_path = os.path.join(rootpath, dir)
        # 新建
        without_mask_path = os.path.join(out_withoutmpath, dir)
        mask_path = os.path.join(out_mpath, dir)
        os.mkdir(without_mask_path)
        os.mkdir(mask_path)

        # 找到unmasked文件，其余复制到另外一个
        imgs = os.listdir(curdir_path)
        for img in imgs:
            if (img.split(".")[0] == "unmasked"):
                orgimg_path = os.path.join(curdir_path, img)
                target_path = os.path.join(without_mask_path, img)
                shutil.copy(orgimg_path, target_path)
            else:
                orgimg_path = os.path.join(curdir_path, img)
                target_path = os.path.join(mask_path, img)
                shutil.copy(orgimg_path, target_path)

def check_dir(dirs_path):
    dirs = os.listdir(dirs_path)
    out_list = []
    for dir in dirs:
        curdir_path = os.path.join(dirs_path, dir)
        imgs = os.listdir(curdir_path)
        if(len(imgs) == 1):
            imgpath = os.path.join(curdir_path,imgs[0])
            img = cv2.imread(imgpath)
            imgF = cv2.flip(img,1)
            new_basename = os.path.splitext(imgs[0])[0] + '_flop' +os.path.splitext(imgs[0])[1]
            flop_path = os.path.join(curdir_path,new_basename)
            # cv2.imshow(str(os.path.basename(curdir_path)),imgF)
            # cv2.waitKey(0)
            cv2.imwrite(flop_path,imgF)
            out_list.append(os.path.basename(curdir_path))
    return out_list

def gen_easyset_json(datadir,outputdir):
    pass




if __name__ == "__main__":
    data_dir = '/home/crx/150gdata/20230421_facedata/experiment_img/val'
    jsonpath = os.path.join(data_dir,"easy_set_val.json")
    file = open(jsonpath,'w')
    file.close()
    # 初始化json和dir
    class_names_list = []
    class_roots_list= []

    # 遍历获得数据
    dirs = os.listdir(data_dir)
    for dir in dirs:
        class_names_list.append(dir)
        curdir_path = os.path.join(data_dir, dir)
        class_roots_list.append(curdir_path)
    json_dic = {"class_names": class_names_list, "class_roots": class_roots_list}
    with open(jsonpath, "w") as f:
        json.dump(json_dic,f)



