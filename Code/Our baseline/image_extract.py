import os
from queue import Queue
import time
import threading

# 创建队列实例， 用于存储任务
queue = Queue()

# 获取文件名 不带文件名
def get_file_name_noExt(file_dir): 
    L = []
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录

        # 只提取mp4
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.splitext(file)[0])

        # # 提取所有
        # L.append)(os.path.splitext(file)[0])
    return L

# 获取文件绝对路径        
def get_file_path(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        # 只提取mp4
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, file))
            
        # # 提取所有
        # for file in files:
        #         L.append(os.path.join(root, file))
    return L

# 获取文件绝对路径 不带文件名       
def get_file_path_noExt(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        # 只提取mp4
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                L.append(os.path.join(root, os.path.splitext(file)[0]))
            
        # # 提取所有
        # for file in files:
        #         L.append(os.path.join(root, os.path.splitext(file)[0]))
    return L

# 拼接相对路径
def get_relpath(file_path,raw_path,out_path):
    L =[]
    for i in file_path:
        rel = os.path.relpath(i,raw_path)
        L.append(out_path+"\\"+rel)

    return L

# 创建多级文件夹    
def mkdir_multi(path):
    # 判断路径是否存在
    isExists=os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path) 
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

# 剪辑视频
def cutting_video():
    while True:
        L = queue.get()

        # excute_path,abs_path,save_path
        new_cmd = cmd.format(L[0],L[1],L[2])

        f = os.popen(new_cmd)
        f.read()
        queue.task_done()


if __name__ == '__main__':
    # 源文件目录
    raw_dir = r'F:\视频2\video2'
    # 输出截取之后的文件目录
    out_dir = r'G:\V2'

    # ffmpeg.exe路径
    excute_path = r'D:\tanyaya\use\ffmpeg\bin\ffmpeg.exe'
    # ffmpeg指令   1s截1帧   {ffmpeg.exe路径} {视频路径} {保存路径}
    cmd = cmd =r"{} -i {} -f image2 -r 10 {}-%03d.jpg"

    # 文件名 无后缀
    file_name_noExt = get_file_name_noExt(raw_dir)
    # 文件绝对路径
    file_abs_path = get_file_path(raw_dir)
    # 文件绝对路径无后缀
    file_abs_path_noExt = get_file_path_noExt(raw_dir)
    # 输出截图目录
    out_path = get_relpath(file_abs_path_noExt,raw_dir,out_dir)

    # print(file_name_noExt)
    # print(file_abs_path)
    # print(file_abs_path_noExt)
    # print(out_path)
    
    # 创建对应文件夹
    for path in out_path:
        mkdir_multi(path)

    # 创建包括5个线程的线程池
    for i in range(5):
        t = threading.Thread(target=cutting_video)
        t.daemon=True # 设置线程daemon  主线程退出，daemon线程也会退出
        t.start()

    # 执行任务
    for abs_path,out,name in zip(file_abs_path,out_path,file_name_noExt):
        save_path = out+"\\"+name
        queue.put([excute_path,abs_path,save_path])
    
    # 阻塞线程
    queue.join()


    print("\n\n======================")
    print("执行完毕！")




