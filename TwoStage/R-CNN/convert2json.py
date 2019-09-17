import os

s1="/home/cai/Desktop/TableCapturer/json/"
s2=".json"

for i in range(99):
    s3 = str(i).zfill(6)
    os.system("labelme_json_to_dataset"+" "+ s1 + s3 + s2)
    i+=1
