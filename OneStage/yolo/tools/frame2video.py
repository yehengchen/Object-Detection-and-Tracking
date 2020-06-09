import cv2
import os
import moviepy.editor as mp

video = "00046"
#input_imgs
im_dir = '/home/cai/Desktop/yolo_dataset/t1_video/t1_video_'+video+'/'
#im_dir =
#output_video
video_dir = '/home/cai/Desktop/yolo_dataset/t1_video/test_video/det_t1_video_'+video+'_test_q.avi'
#fps
fps = 50
#num_of_imgs
num = 310
#img_size
img_size = (1920,1080)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#opencv3

videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
for i in range(0,num):
    #im_name = os.path.join(im_dir,'frame-' + str(i) + '.png')
    im_name = os.path.join(im_dir,'t1_video_'+video+'_' + "%05d" % i + '.jpg')
    frame = cv2.imread(im_name)
    #frame = cv2.resize(frame, (480, 320))
    #frame = cv2.resize(frame,(520,320), interpolation=cv2.INTER_CUBIC)
    videoWriter.write(frame)
    print (im_name)
videoWriter.release()
print('finish')
