# coding = utf-8
from PIL import Image
import os

def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid

path=os.getcwd()+'/src_img/'
file_list=os.listdir(path=path)
with open('log.txt','w') as log:
    for file in file_list:
        print('检测{0}'.format(file))
        if is_valid(path+file):
            1 + 1
        else:
            log.write(file)
