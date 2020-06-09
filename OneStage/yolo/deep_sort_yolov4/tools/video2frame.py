import cv2

image_folder = './mask_face'
video_name = './cut_test.m4v'

vc = cv2.VideoCapture(video_name)
c = 1
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    cv2.imwrite('./mask_face/IMG_'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()
