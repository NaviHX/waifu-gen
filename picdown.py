# coding = utf-8
import requests
import re
from urllib import request
import threading
import random


def download(url_list, id):
    for pic_url in url_list:
        try:
            pic = requests.get(pic_url, timeout=15)
            filename = ''.join(
                random.sample('abcdefghijklmnopqrstuvwxyz!@#$%^&()', 10))
            save_path = path + filename + '.jpg'
            with open(save_path, 'wb') as file:
                file.write(pic.content)
            print('ID : {0} Dowload: {1}'.format(id, filename))
        except Exception as e:
            print(e)
            print(pic_url, 'download ERROR')
    return


class downloadThread(threading.Thread):
    def __init__(self, id, url_list):
        threading.Thread.__init__(self)
        self.list = url_list
        self.id = id

    def run(self):
        download(self.list, self.id)


keyword = input('keyword: ')  # 下载关键字
max_count = int(input('max count: '))  # 下载张数
download_count = 0  # 当前张数
max_page = int(max_count / 80) + 1  # 遍历页数
print('将遍历至搜索结果第{0}页'.format(max_page))
page_count = 0  # 当前页数
max_threads = int(input('max threads: '))  # 最大下载线程数

thread_list = []

for i in range(max_threads):
    t = downloadThread(i, [])
    thread_list.append(t)

url_head = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='  # url头
page_url = url_head + request.quote(keyword, safe='/')  # 翻译关键字
path = './src_img/'  # 保存路径

while page_url != '':
    try:
        html = requests.get(page_url).text
    except Exception as e:
        print(e)
        print('Break')
        break
    url_to_download = re.findall(r'"objURL":"(.*?)"', html, re.S)
    nxt = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'),
                     html,
                     flags=0)
    if nxt and page_count <= max_page:
        page_url = 'http://image.baidu.com' + nxt[0]
    else:
        page_url = ''
    thread_list[page_count % max_threads].list.extend(url_to_download)
    print('完成{0}页链接检索'.format(page_count))
    page_count += 1

for t in thread_list:
    print('线程{0}开始下载'.format(t.id))
    t.start()

for t in thread_list:
    t.join()
