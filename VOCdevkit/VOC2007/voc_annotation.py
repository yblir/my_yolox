# ---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
# ---------------------------------------------#
import os
import xml.etree.ElementTree as ET
from os import getcwd

root_path = os.path.dirname(os.path.realpath(__file__))

xml_path = os.path.join(root_path, 'Annotations')
set_path = os.path.join(root_path, 'ImageSets')
img_path = os.path.join(root_path, 'JPEGImages')

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(xml_path, f'{image_id}.xml'), encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    for year, image_set in sets:
        image_ids = open(os.path.join(set_path, f'{image_set}.txt')).read().strip().split()
        # 产生三个文件夹 2007_train.txt ...
        with open(os.path.join(os.path.dirname(root_path), 'setting', f'{year}_{image_set}.txt'), 'w') as f:
            for image_id in image_ids:
                f.write(os.path.join(img_path, f'{image_id}.jpg'))
                convert_annotation(image_id, f)
                f.write('\n')

