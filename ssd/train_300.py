from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, History
from keras import backend as K
from keras.models import Model
from keras.utils.data_utils import get_file
from math import ceil
import argparse
import os
from typing import Tuple, Iterable, List
import numpy as np
from glob import glob
import re

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation


def get_args() -> argparse.Namespace:
    # data paremeters
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_height', type=int, default=300)
    parser.add_argument('--img_width', type=int, default=300)
    parser.add_argument('--img_channel', type=int, default=3)
    parser.add_argument('--mean_color', type=int, nargs='+', default=[123, 117, 104],
                        help="The per-channel mean of the images in the dataset. " +
                             "Do not change this value if you're using any of the pre-trained weights.")

    # model general paremeters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--n_classes', type=int, default=20)

    # model paremeters
    parser.add_argument('--swap_channels', type=int, nargs='+', default=[2, 1, 0],
                        help="The color channel order in the original SSD is BGR, " +
                             "so we'll have the model reverse the color channel order of the input images.")
    parser.add_argument('--two_boxes_for_ar1', action='store_true', default=True)
    parser.add_argument('--clip_boxes', action='store_true')
    parser.add_argument('--normalize_coords', action='store_true', default=True)
    parser.add_argument('--variances', type=float, nargs=4, default=[0.1, 0.1, 0.2, 0.2],
                        help="The variances by which the encoded target coordinates are " +
                             "divided as in the original implementation")

    N_PREDICTOR_LAYERS = 6  # same value as models/keras_ssd300.py
    parser.add_argument('--scales', type=float, nargs=N_PREDICTOR_LAYERS + 1,
                        default=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05])
    parser.add_argument('--steps', type=int, nargs=N_PREDICTOR_LAYERS, default=[8, 16, 32, 64, 100, 300],
                        help="The space between two adjacent anchor box center points for each predictor layer.")
    parser.add_argument('--offsets', type=float, nargs=N_PREDICTOR_LAYERS, default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        help="The offsets of the first anchor box center points from the top and left borders " +
                             "of the image as a fraction of the step size for each predictor layer.")
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]
    for i in range(N_PREDICTOR_LAYERS):
        parser.add_argument('--aspect_ratio_layer_{}'.format(i+1), nargs='+',
                            type=int, default=aspect_ratios[i])

    # SageMaker paremeters
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--data_dir', type=str, default='../../datasets/VOCdevkit')

    return parser.parse_args()


def restore_checkpoint(ckpt_path: str) -> Tuple[str, int, ModelCheckpoint]:
    ckpt_file_format = 'weights.{}.h5'
    ckpt_files = glob(os.path.join(ckpt_path, ckpt_file_format.format('*')))
    checkpoint = ModelCheckpoint(
        monitor='var_loss',
        save_best_only=True,
        filepath=os.path.join(ckpt_path, ckpt_file_format.format('{epoch:03d}')),
        save_weights_only=True)

    if len(ckpt_files) == 0:
        remote_weight_path = "https://drive.google.com/uc?export=download&id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox"
        weights_path = get_file('vgg_weight.h5', remote_weight_path, cache_subdir=ckpt_path)
        return weights_path, 0, checkpoint
    else:
        weights_path = ckpt_files[-1]
        search_result = re.compile(ckpt_file_format.format('([0-9]+)')).search(weights_path)
        if search_result is None:
            raise ValueError('saved weight files have invalid formats')
        epoch = int(search_result.group(1))
        return weights_path, epoch, checkpoint


def build_model(args: argparse.Namespace, weights_path: str) -> Model:
    K.clear_session()
    model = ssd_300(image_size=(args.img_height, args.img_width, args.img_channels),
                    n_classes=args.n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=args.scales,
                    aspect_ratios_per_layer=args.aspect_ratios,
                    two_boxes_for_ar1=args.two_boxes_for_ar1,
                    steps=args.steps,
                    offsets=args.offsets,
                    clip_boxes=args.clip_boxes,
                    variances=args.variances,
                    normalize_coords=args.normalize_coords,
                    subtract_mean=args.mean_color,
                    swap_channels=args.swap_channels)

    model.load_weights(weights_path, by_name=True)

    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

    return model


def get_dataset(args: argparse.Namespace,
                model: Model) -> Tuple[Iterable[List[np.array]], Iterable[List[np.array]], int]:
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    VOC_2007_images_dir = os.path.join(args.data_dir, '/VOC2007/JPEGImages/')
    VOC_2012_images_dir = os.path.join(args.data_dir, '/VOC2012/JPEGImages/')

    VOC_2007_annotations_dir = os.path.join(args.data_dir, '/VOC2007/Annotations/')
    VOC_2012_annotations_dir = os.path.join(args.data_dir, '/VOC2012/Annotations/')

    VOC_2007_trainval_image_set_filename = os.path.join(args.data_dir,
                                                        '/VOC2007/ImageSets/Main/trainval.txt')
    VOC_2012_trainval_image_set_filename = os.path.join(args.data_dir,
                                                        '/VOC2012/ImageSets/Main/trainval.txt')
    VOC_2007_test_image_set_filename = os.path.join(args.data_dir,
                                                    '/VOC2007/ImageSets/Main/test.txt')

    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                         VOC_2012_images_dir],
                            image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                                 VOC_2012_trainval_image_set_filename],
                            annotations_dirs=[VOC_2007_annotations_dir,
                                              VOC_2012_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                          image_set_filenames=[VOC_2007_test_image_set_filename],
                          annotations_dirs=[VOC_2007_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
                                      resize=False,
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

    ssd_data_augmentation = SSDDataAugmentation(img_height=args.img_height,
                                                img_width=args.img_width,
                                                background=args.mean_color)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=args.img_height, width=args.img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=args.img_height,
                                        img_width=args.img_width,
                                        n_classes=args.n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=args.scales,
                                        aspect_ratios_per_layer=args.aspect_ratios,
                                        two_boxes_for_ar1=args.two_boxes_for_ar1,
                                        steps=args.steps,
                                        offsets=args.offsets,
                                        clip_boxes=args.clip_boxes,
                                        variances=args.variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=args.normalize_coords)

    train_generator = train_dataset.generate(batch_size=args.batch_size,
                                             shuffle=True,
                                             transformations=[ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=args.batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)
    return train_generator, val_generator, val_dataset.get_dataset_size()


def train(args: argparse.Namespace, current_epoch: int, ckpt: ModelCheckpoint, model: Model, val_dataset_size: int,
          train_generator: Iterable[List[np.array]], val_generator: Iterable[List[np.array]]) -> History:
    def lr_schedule(epoch):
        if epoch < 80:
            return 0.001
        elif epoch < 100:
            return 0.0001
        else:
            return 0.00001

    lr_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [ckpt, lr_scheduler, terminate_on_nan]

    return model.fit_generator(generator=train_generator,
                               steps_per_epoch=args.steps_per_epoch,
                               epochs=args.epochs,
                               callbacks=callbacks,
                               validation_data=val_generator,
                               validation_steps=ceil(val_dataset_size/args.batch_size),
                               initial_epoch=current_epoch)


def main():
    args = get_args()
    weights_path, current_epoch, ckpt = restore_checkpoint(args.ckpt_path)
    model = build_model(args, weights_path)
    train_generator, val_generator, val_dataset_size = get_dataset(args, model)
    train(args, current_epoch, ckpt, model, val_dataset_size, train_generator, val_generator)
