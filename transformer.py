import albumentations as A
import cv2
from utils_format import *
from utils_drawing import draw_yolo_bboxes

# Read the Image
image = cv2.imread("./images/000000000036.jpg")
height, width, _ = image.shape
min_area = 0.1 * height * 0.1 * width
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Read the Label
yolo_bboxes = read_label_yolo('./images/000000000009.jpg')
album_bboxes = yolo_to_albumentations(yolo_bboxes)

# Transformer Pipeline
transformer = A.Compose([
           A.RandomCrop(width=450, height=450),
           A.HorizontalFlip(p=5),
           A.RandomBrightnessContrast(p=0.5),
        ],
        bbox_params = A.BboxParams(format='yolo',
                    min_area=min_area, 
                    min_visibility=0.1,
                    )
        )

# Operation
transformed = transformer(image=rgb_image, bboxes=album_bboxes)
transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
transformed_bboxes = transformed['bboxes']
print('TRANSFORMED BB: {}'.format(transformed_bboxes))
# Display Result
before = draw_yolo_bboxes(image, yolo_bboxes)
after = draw_yolo_bboxes(transformed_image, albumentations_to_yolo(transformed_bboxes))
cv2.imshow("original", before)
cv2.imshow("result", after)
cv2.waitKey(0)
# Save Result
cv2.imwrite('./results/000000000036_augment.jpg', after)
write_label_yolo('./results/000000000036_augment.jpg', albumentations_to_yolo(transformed_bboxes))