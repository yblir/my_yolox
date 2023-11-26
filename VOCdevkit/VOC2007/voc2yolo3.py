# ----------------------------------------------------------------------#
#   验证集的划分在train.py代码里面进行
#   test.txt和val.txt里面没有内容是正常的。训练不会使用到。
# ----------------------------------------------------------------------#
import os
import random

random.seed(0)

root_path = os.path.dirname(os.path.realpath(__file__))
# xmlfilepath = r'./VOCdevkit/VOC2007/Annotations'
# saveBasePath = r"./VOCdevkit/VOC2007/ImageSets/Main/"

xmlfilepath = os.path.join(root_path, 'Annotations')
saveBasePath = os.path.join(root_path, 'ImageSets')
# ----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   train_percent不需要修改
# ----------------------------------------------------------------------#
trainval_percent = 1
train_percent = 1

# 获得所有xml文件的路径列表
total_xml = [xml for xml in os.listdir(xmlfilepath) if xml.endswith('.xml')]

num = len(total_xml)

tv = int(num * trainval_percent)
tr = int(tv * train_percent)

# 从range(0,num)中随机获取tv个元素,作为一个片段返回
trainval = random.sample(range(num), tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("traub suze", tr)

ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in range(num):
    # 取文件名的名字部分,去掉.xml, 并换行
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        ftrain.write(name) if i in train else fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
