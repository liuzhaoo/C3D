import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)                 # 定义根目录和输出目录
        folder = os.path.join(self.output_dir, split)                         # 处理好的数据集输出目录
        self.clip_len = clip_len                                              # 输入的帧长度
        self.split = split                                                    # 读取的数据集类型

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():                                        # 若不存在数据集路径，则报错
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:                       # 如果预处理的条件未达到或izhe预处理参数为True，则进行预处理数据
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):                # 取所有文件名作为类别
            for fname in os.listdir(os.path.join(folder, label)):     # 取训练集（或其他）下每个类别下的所有文件名（由视频转换过来的）
                self.fnames.append(os.path.join(folder, label, fname))      # 将视频名放到self.fnames
                labels.append(label)                                        # 每个视频对应的标签（类别）都放到list里

        assert len(labels) == len(self.fnames)                              # 保证视频数量和其对应的标签数一样
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))  # 打印某个数据集（训练集）的所有类别的视频数量

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}    # 先把标签转换为集合，去除重复项，再从小到大排序，然后用标签的索引来表示每个标签
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)   # 将标签名字的索引转换为数组

        if dataset == "ucf101":
            if not os.path.exists('./dataloaders/ucf_labels.txt'):
                with open('./dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')                          # 生成标签文件

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)                                                            # 查看训练集中所有视频的数量

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])                           # 加载帧
        buffer = self.crop(buffer, self.clip_len, self.crop_size)               # 裁剪
        labels = np.array(self.label_array[index])                              # 取标签

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)                                    # 在测试集上随机反转
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)                #  返回帧和

    def check_integrity(self):                                  # 判断是否存在数据集路径，存在返回True,否则返回False
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):                                 # 判断是否存在数据集的输出路径和子路径‘train’，同时，检查生成的图片的size
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):                 # 判断是否存在数据集的输出路径，若不存在，直接返回False，若存在，继续判断是否存在子目录
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):          # 判断子目录
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):   # 在train目录中循环文件名（动作类型）
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):        # 在每个动作下循环每个图片
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:                     #判断size
                    return False
                else:
                    break

            if ii == 10:                                                                     # 只看10个动作
                break

        return True                                                                         # 若以上条件都满足，则返回True

    def preprocess(self):
        if not os.path.exists(self.output_dir):                                     # 如果输出的数据集目录不存在，则创建
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):                                     # 在数据集中循环，取出每个类别的文件名放到file里
            file_path = os.path.join(self.root_dir, file)                          # 得到每个类别的名称
            video_files = [name for name in os.listdir(file_path)]                 # 在每个类别下迭代文件名，放到列表里

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)   # 将数据集的20%作为测试集
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)           # 将剩下的80%再分出20%做验证集

            train_dir = os.path.join(self.output_dir, 'train', file)            # 分别在三种数据集下建立各种类别的文件夹
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:                                                    # 分别在三种文件夹下的类别里添加处理后的数据，video是每个视频的文件名
                self.process_video(video, file, train_dir)                         # 将分离出来的帧分别放到对应路径

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]                                         # 将视频名从‘.’前后分开，取前面的内容作为文件夹名
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))                         # 在各类别下新建视频名的文件夹

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))  # 取数据集每个类别下的视频

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))        # 计算视频的帧数
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))        # 计算视频宽度
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 计算视频高度

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)   # 保存每一帧
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer                                            # 随机反转


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])            #
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))                  # 转换为tensor形式的size

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])       # 在给定路径中取文件名放到列表里
        frame_count = len(frames)                                                            # 计算长度
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))   # 创建一个新的数组
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)                      #
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break