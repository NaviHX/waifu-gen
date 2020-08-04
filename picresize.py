from PIL import Image
import os
import sys, getopt
import tensorflow

path = os.getcwd() + '/src_img/'
save_path = os.getcwd() + '/resized_img'
file_list = os.listdir(path=path)
total=0
for file in file_list:
    try:
        pic = path + file
        img = Image.open(pic)
        img = img.resize((32, 32))
        total+=1
        img.save('{0}/{1}.jpg'.format(save_path,str(total)))
        print('{1}:{0} SAVED'.format(file,str(total)))
    except Exception as e:
        print('{0} ERROR {1}'.format(file,e))
        exit()
