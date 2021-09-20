import cv2
import os
import json
import numpy as np
import argparse

def write_key_frame(root, frame_list, idx, json_dict, search_start, search_end, umpire_start, umpire_end):
    json_dict[idx] = {}
    json_dict[idx]['strike_zone_roi']   = [search_start[1], search_start[0], search_end[1], search_end[0]]# 2
    json_dict[idx]['umpire_roi']        = [umpire_start[1], umpire_start[0], umpire_end[1], umpire_end[0]]# 1
    json_dict[idx]['gt']                = -1 # 0: in grid, 1: out of grid, 2: on the edge  # 3 
    json_dict[idx]['file_names'] = []
    for i, frame in enumerate(frame_list):
        frame_path = os.path.join(root, 'key-frame' + str(idx) + '-' + str(i) + '.png')
        cv2.imwrite(frame_path, frame)

        json_dict[idx]['file_names'].append(frame_path)
    return json_dict

def is_key_frame(frame, template_img):
    # if there is a strike zone -> return True
    # if not -> return False
    result = cv2.matchTemplate(frame, template_img, cv2.TM_SQDIFF_NORMED)
    threshold_val = 0.95
    loc = np.where(result >= threshold_val)
    
    if len(loc[0]) == 0:
        return False
    else:
        return True

def is_ball_in(frame, template_ball):
    result = cv2.matchTemplate(frame, template_ball, cv2.TM_CCOEFF_NORMED)
    threshold_val = 0.5
    loc = np.where(result >= threshold_val)
    
    if len(loc[0]) == 0:
        return False
    else:
        return True
def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--template_pitching_board', type=str, default=None)
    parser.add_argument('--template_ball', type=str, default=None)
    return parser.parse_args()

def main():
    args=opt()

    search_start = (1030, 490)
    search_end = (1240, 690)
    
    umpire_start = (600, 900)
    umpire_end = (150, 490)
    
    # [1] video read
    SRC_VIDEO_PATH = args.input_video
    JSON_PATH      = args.output_json
    DST_FRAME_PATH = args.output_dir

    template_grid = cv2.imread(args.template_pitching_board)
    template_ball = cv2.imread(args.template_ball)
    
    grid_h, grid_w, _ = template_grid.shape
    ball_h, ball_w, _ = template_ball.shape
    
    if not os.path.exists(DST_FRAME_PATH):
        os.makedirs(DST_FRAME_PATH)

    cap = cv2.VideoCapture(SRC_VIDEO_PATH)
    
    PREV_FRAME_PERIOD = 6
    frame_list = [] # [framet-5 t-4 t-3 t-2 t-1 t]
    idx = 0
    json_dict = {}
    while(cap.isOpened()):
        idx += 1
        ret, frame = cap.read()

        if len(frame_list) != PREV_FRAME_PERIOD:
            frame_list.append(frame)
        elif len(frame_list) == PREV_FRAME_PERIOD:
            frame_list.append(frame)
            frame_list = frame_list[1:]
        else:
            assert False

        # [2] Key-frame detection (using template matching)
        if is_key_frame(frame[search_start[1]:search_end[1], search_start[0]:search_end[0]], template_grid):
            
            # [3] Ball detection (using template matching)
            if is_ball_in(frame[search_start[1]:search_end[1], search_start[0]:search_end[0]], template_ball):
                json_dict = write_key_frame(DST_FRAME_PATH, frame_list, idx, json_dict,
                                            search_start, search_end, umpire_start, umpire_end)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    with open(JSON_PATH, 'w') as f:
        json.dump(json_dict, f, indent='\t')
    print('json annotation file has been written in {}!'.format(JSON_PATH))

if __name__ == '__main__':
    main()