import os
import sys
import argparse
import cv2
import time
import numpy as np
from config_reader import config_reader

from processing import extract_parts, draw

from cmu_model import get_testing_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test1.mp4', help='input video file name')
    parser.add_argument('--model', type=str, default='model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=15, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=1,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()

    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    print('start processing...')

    # Video input
    video = args.video
    video_path = ''
    video_file = video_path + video

    # Output location
    output_path = 'output/'
    output_format = '.avi'
    video_output = output_path + 'op' + str(start_datetime) + output_format

    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    # Video reader
    cam = cv2.VideoCapture(video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    scale_search = [.5, 1, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]

    params['scale_search'] = scale_search

    i = 0  # default is 0
#    f=open('points.txt','a')
        
    while(cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:

            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()

            # generate image with body parts
            all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
##            for x in all_peaks:
##                f.write(str(x))
            canvas = draw(orig_image, all_peaks, subset, candidate)
            

            print('Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))

            out.write(canvas)

        ret_val, orig_image = cam.read()
    
        i += 1
    f.close()
    cam.release()
    out.release()
    cv2.destroyAllWindows()
