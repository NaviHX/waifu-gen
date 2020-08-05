from PIL import Image
import os

log = open('log.txt', 'w')
file = os.listdir('./resized_img')
total = 0
for i in file:
    try:
        Image.open('./resized_img/' + i)
    except Exception as e:
        print('{} fail'.format(i))
        log.write(i+'\n')
        total += 1
print(total)