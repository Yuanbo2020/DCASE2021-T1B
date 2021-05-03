import os

import zipfile
import cv2

k = 2
def mkdir(path):
    # 新建文件夹
    if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径



def unzip_file(zip_src, dist_path):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dist_path)
    else:
        print('This is not zip')

def unzip_all_file():
    # 修改路径
    # zip_dir: 压缩文件存放的路径 如:/Users/ymn/Documents/
    # dist_dir: 解压到的路径
    zip_dir = "/home/share/tyz"
    dist_dir = "/home/share/tyz"
    mkdir(dist_dir)
    for root, dirs, files in os.walk(zip_dir):
        for file in files:
            if file.endswith(".zip"):
                # print(file)
                unzip_file(os.path.join(root, file),dist_dir)

def save_key_frame(video_path, image_path,file_name):
    '''
    Save the key frame image
    :return:
    '''

    print("save_key_frame")
    cap = cv2.VideoCapture(video_path)  # 打开视频文件

    fps = round(cap.get(5))
    timeF = round(fps / k)
    frm = 0
    j = 0
    while True:
        frm = frm + 1
        success, data = cap.read()
        if not success:
            break
        if (frm % timeF == 0):
            j = j + 1
            cv2.imwrite(os.path.join(image_path, file_name + "_" + str(j) + ".jpg"), data)
def extract_frame(video_path="/home/share/tyz/video",image_path="/home/share/tyz/dataset_2fps/video"):

    cls_name = ["bus", "metro", "metro_station", "street_traffic",
                "street_pedestrian", "tram", "park", "shopping_mall", "public_square", "airport"]
    files = os.listdir(video_path)
    num = 0
    for file in files:
        first_name = file.split('-')[0]
        # 如果以十种类别名开头, 则复制到新目录
        num += 1

        if first_name in cls_name:
            # 复制文件
            orig_path = os.path.join(video_path, file)

            to_path = os.path.join(image_path, first_name, os.path.splitext(file)[0])
            if not os.path.exists(to_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(to_path)

            else:
                num_file = len(os.listdir(to_path))
                if num_file >= 10*k:
                    continue
            print(num)
            save_key_frame(orig_path, to_path, os.path.splitext(file)[0])
        else:
            print(first_name)



if __name__ == '__main__':
    # unzip_all_file()
    extract_frame()
