import os
import os.path as osp
import ffmpeg
import cv2
import copy
import json

from glob import glob

import numpy as np


def draw_label_on_video(frame_dir, labels, save_path, fps=24):
    size = (320, 180)
    frame_name_template = 'img_*.jpg'
    frame_path_template = osp.join(frame_dir, frame_name_template)
    # print(frame_path_template)
    frame_paths = glob(frame_path_template)
    frame_paths = sorted(frame_paths)
    print(frame_paths)
    num_frames = len(frame_paths)

    new_labels = [[] for i in range(num_frames)]
    for i in range(len(labels)):
        new_labels[i] = labels[i]
    assert num_frames == len(new_labels)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    videowrite = cv2.VideoWriter(save_path, fourcc, fps, size)

    # 7.合成视频
    for i in range(num_frames):
        img = cv2.imread(frame_paths[i])
        img = cv2.resize(img, size)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(new_labels[i]):
            show_label = copy.deepcopy(new_labels[i])
            if "Background" in show_label:
                show_label.remove("Background")
            if len(show_label):
                text = show_label[0]
                color = (0, 0, 255)  # BGR 格式颜色
                thickness = 2
                font_scale = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                # text_x = 0
                # text_y = 100
                text_x = int((img.shape[1] - text_size[0]) / 2)
                text_y = int((img.shape[0] + text_size[1]) / 2)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        videowrite.write(img)
        print('第{}张图片合成成功'.format(i))


if __name__ == "__main__":
    THUMOS14_ROOT = osp.join("data", "thumos14")
    session = "video_validation_0000051"
    frame_dir = osp.join(THUMOS14_ROOT, "rawframes", "val", session)
    save_dir = osp.join(THUMOS14_ROOT, "annotated_video", "val")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, session + ".mp4")
    json_file = osp.join("data", "data_info.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        data_info = json.load(f)
        thumos14_info = data_info["THUMOS"]
        class_names = thumos14_info["class_names"]
        print(class_names)
    label_file = osp.join(THUMOS14_ROOT, "labels", "val", session + ".npz")
    labels = np.load(label_file, allow_pickle=True)
    labels = labels["pred_labels"]
    labels = labels.tolist()
    interval = 6
    new_labels = [[] for i in range(interval * len(labels))]
    for i in range(len(labels)):
        # for j in range(len(labels[i])):
        #     label = []
        #     label.append(class_names[int(labels[i][j])])
        for j in range(interval):
            names = []
            for id in labels[i]:
                names.append(class_names[id])
            new_labels[interval * i + j] = names
    draw_label_on_video(frame_dir, new_labels, save_path)
