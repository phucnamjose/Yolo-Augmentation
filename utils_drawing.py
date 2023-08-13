import cv2
import random


def draw_yolo_bboxes(img, bboxes):
    drawn_img = img.copy()
    
    for bbox in bboxes:
        left, top, right, bottom = bbox2points(convert2pixel(img, bbox))
        print(left, top, right, bottom)
        color = random_colors()
        label = bbox[0]
        cv2.rectangle(drawn_img, (left, top), (right, bottom), color, 1)
        cv2.putText(drawn_img, "{}".format(label),
                    (left + 1, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)
    return drawn_img

def convert2pixel(img, bbox):
    height, width, _ = img.shape
    _, x, y, w, h = bbox
    return x*width, y*height, w*width, h*height

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def random_colors():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255))