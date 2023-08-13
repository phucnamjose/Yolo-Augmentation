import os

def read_label_yolo(image_path):
    dir, image_name = os.path.split(image_path)
    label_name = image_name.split('.')[0] + '.txt'
    label_path = os.path.join(dir, label_name)
    if not os.path.isfile(label_path):
        return []
    bboxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('\n'):
                continue
            else:
                fields = line.split(' ')
                bboxes.append([
                    fields[0],
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                ])
    print('READ LABEL: {}'.format(bboxes))
    return bboxes

def write_label_yolo(image_path, bboxes):
    print(bboxes)
    dir, image_name = os.path.split(image_path)
    label_name = image_name.split('.')[0] + '.txt'
    label_path = os.path.join(dir, label_name)
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            line = '{} {} {} {} {}\n'\
            .format(bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    bbox[4])
            file.write(line)

def yolo_to_albumentations(bboxes):
    albumentation_bboxes = []
    for bbox in bboxes:
        albumentation_bboxes.append(bbox[1:] + bbox[0:1])
    return albumentation_bboxes

def albumentations_to_yolo(bboxes):
    yolo_bboxes = []
    for bbox in bboxes:
        yolo_bboxes.append(bbox[-1:] + bbox[:-1])
    print('CONVERTED LABEL: {}'.format(yolo_bboxes))
    return yolo_bboxes