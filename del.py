import os

path='./resized_img/'
file_list=open('log.txt','r').read().splitlines()
for i in file_list:
    os.remove(path+i)
    print("delete "+path+i)
