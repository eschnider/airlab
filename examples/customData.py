import glob
import multiprocessing as mp
import os
import numpy as np

import SimpleITK as sitk

import airlab as al
from airlab import Image


class RegistrationData:
    image = None
    mask = None
    landmarks = None

    def __init__(self, image, mask=None, landmarks=None):
        self.image = image
        self.mask = mask
        self.landmarks = landmarks


class ScanGroup:
    scans = None
    common_origin = None
    common_extent = None
    common_spacing = None
    common_size = None

    def __init__(self, scans):
        self.scans = scans

    def compute_common_domain(self):
        moving_images = []
        for scan_number, scan in enumerate(self.scans):
            if scan_number == 0:
                reference_image = scan.volume
            else:
                moving_images.append(scan.volume)
        self.common_origin, \
        self.common_extent, \
        self.common_spacing, \
        self.common_size = al.utils.domain.find_common_domain(reference_image, moving_images)

    def resample_common_inplace(self, file_type='all', default_value=0.0, interpolator=2):

        if self.common_extent is None or self.common_spacing is None or self.common_size is None:
            self.compute_common_domain()

        resampler = create_resampler(self.common_origin, self.common_spacing, self.common_size,
                                     default_value=default_value, interpolator=interpolator)
        for scan in self.scans:
            if file_type == 'all' or file_type == 'volume':
                scan.volume = Image(resampler.Execute(scan.volume.itk()))
            elif file_type == 'mask':
                scan.mask = Image(resampler.Execute(scan.mask.itk()))
            else:
                raise ValueError('unknown file type')

    def save_to_disk(self, save_dir, exist_ok=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        elif not exist_ok:
            while os.path.exists(save_dir):
                save_dir = "{}_Copy".format(save_dir)
        for scan in self.scans:
            scan.save_scan_to(save_dir, exist_ok=True)


class Scan:
    name = None
    path = None
    volume_path = None
    label_path = None
    mask_path = None
    displacement_path = None
    landmarks_path = None

    def __init__(self, name):
        self.name = name
        self._volume = None
        self._label = None
        self._mask = None
        self._landmarks = None
        self._displacement = None

    @property
    def volume(self):
        if self._volume is None:
            self._volume = al.Image.read(self.volume_path)
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def label(self):
        if self._label is None:
            self._label = al.Image.read(self.label_path)
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def mask(self):
        if self._mask is None:
            self._mask = al.Image.read(self.mask_path)
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def landmarks(self):
        if self._landmarks is None:
            self._landmarks = al.utils.Points.read(self.landmarks_path)
        return self._landmarks

    @landmarks.setter
    def landmarks(self, value):
        self._landmarks = value

    @property
    def displacement(self):
        if self._displacement is None:
            self._displacement = al.Displacement.read(self.displacement_path)
        return self._displacement

    @displacement.setter
    def displacement(self, value):
        self._displacement = value

    def find_domain_for(self, volume, spacing=None, size=None):

        origin = volume.origin
        extent = np.array(volume.origin) + (np.array(volume.size) - 1) * np.array(
            volume.spacing)

        if spacing is not None:
            size = np.ceil(((extent - origin) / spacing) + 1).astype(int)
        elif size is not None:
            spacing = (extent - origin) / (size - 1)
        direction = volume.direction

        return origin, extent, spacing, size, direction

    def resample_to(self, reference_scan=None, spacing=None, size=None, file_type='all', default_value=0.0,
                    interpolator=2):
        if reference_scan is not None:
            moving_images = []
            if (self.volume is not None):
                moving_images.append(self.volume)
            elif self.label is not None:
                moving_images.append(self.label)
            if reference_scan.volume is not None:
                reference_image = reference_scan.volume
            elif reference_scan.label is not None:
                reference_image = reference_scan.label
            origin, extent, spacing, size = al.utils.domain.find_common_domain(reference_image, moving_images)
        else:
            origin, extent, spacing, size, direction = self.find_domain_for(self.volume, spacing=spacing, size=size)

        resampler = create_resampler(origin, spacing, size, direction=direction,
                                     default_value=default_value, interpolator=interpolator)
        self.apply_sitk_filter(resampler, file_type)

    def apply_sitk_filter(self, sitk_filter, file_type):
        if file_type == 'volume' or file_type == 'all':
            self.volume = Image(sitk_filter.Execute(self.volume.itk()))
        if file_type == 'mask' or file_type == 'all':
            self.mask = Image(sitk_filter.Execute(self.mask.itk()))
        if file_type == 'label' or file_type == 'all':
            self.label = Image(sitk_filter.Execute(self.label.itk()))

    def flip(self, flip_axes, file_type='all'):
        flip_filter = sitk.FlipImageFilter()
        flip_filter.SetFlipAxes(flip_axes)
        flip_filter.FlipAboutOriginOff()  # flip around the center of the axis.
        self.apply_sitk_filter(flip_filter, file_type)

    def permute_axes(self, new_order, file_type='all'):
        permute_axes_filter = sitk.PermuteAxesImageFilter()
        permute_axes_filter.SetOrder(new_order)
        self.apply_sitk_filter(permute_axes_filter, file_type)

    def shrink(self, shrink_factor, file_type='all'):
        shrink_filter = sitk.ShrinkImageFilter()
        shrink_filter.SetShrinkFactor(shrink_factor)
        self.apply_sitk_filter(shrink_filter, file_type)

    def add_padding(self, padding_per_dimension, filling_constant, file_type='all'):
        pad_filter = sitk.ConstantPadImageFilter()
        padding_upper = list(np.ceil(np.array(padding_per_dimension)/2))
        padding_lower = list(np.floor(np.array(padding_per_dimension)/2))
        pad_filter.SetPadLowerBound(sitk.VectorUInt32([int(i) for i in padding_lower]))
        pad_filter.SetPadUpperBound(sitk.VectorUInt32([int(i) for i in padding_upper]))
        pad_filter.SetConstant(filling_constant)
        self.apply_sitk_filter(pad_filter, file_type)

    def save_scan_to(self, save_dir, exist_ok=False, save_gzipped=True):
        def keep_gz_ending(file_path):
            return file_path
        def remove_gz_ending(file_path):
            base_name=os.path.basename(file_path)
            if base_name.endswith('.gz'):
                base_name_without_gz = base_name[0:-3]
            else:
                base_name_without_gz = base_name
            return os.path.join(os.path.dirname(file_path), base_name_without_gz)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        elif not exist_ok:
            while os.path.exists(save_dir):
                save_dir = "{}_Copy".format(save_dir)
        if save_gzipped is False:
            suffix_operation=remove_gz_ending
        else:
            suffix_operation=keep_gz_ending
        scan_save_path = os.path.join(save_dir, self.name)
        if not os.path.exists(scan_save_path):
            os.makedirs(scan_save_path, exist_ok=False)
        if os.path.isfile(self.volume_path):
            volume_file_path = os.path.join(scan_save_path, os.path.basename(suffix_operation(self.volume_path)))
            self.volume.write(volume_file_path)
        if os.path.isfile(self.mask_path):
            mask_file_path = os.path.join(scan_save_path, os.path.basename(suffix_operation(self.mask_path)))
            self.mask.write(mask_file_path)
        if os.path.isfile(self.label_path):
            label_file_path = os.path.join(scan_save_path, os.path.basename(suffix_operation(self.label_path)))
            self.label.write(label_file_path)


class SkeletonScan(Scan):
    _default_volume_name = "volume.nii.gz"
    _default_label_name = "bones.nii.gz"
    _default_mask_name = "ROI*.nii.gz"
    _default_landmarks_name = "landmark.fcsv"
    _default_displacement_name = "bspline_displacement_image_unit.vtk"

    def __init__(self, scan_dir, file_naming=None):
        super().__init__(os.path.basename(scan_dir))
        self.path = scan_dir

        volume_name = self._default_volume_name
        label_name = self._default_label_name
        mask_name = self._default_mask_name
        landmarks_name = self._default_landmarks_name
        displacement_name = self._default_displacement_name

        if file_naming is not None:
            if 'volume' in file_naming.keys():
                volume_name = file_naming['volume']
            if 'label' in file_naming.keys():
                label_name = file_naming['label']
            if 'mask' in file_naming.keys():
                mask_name = file_naming['mask']
            if 'landmarks' in file_naming.keys():
                landmarks_name = file_naming['landmarks']
            if 'displacement' in file_naming.keys():
                displacement_name = file_naming['displacement']

        for bone_label_file in glob.glob(os.path.join(scan_dir, volume_name)):
            self.volume_path = bone_label_file
        for mask_label_file in glob.glob(os.path.join(scan_dir, label_name)):
            self.label_path = mask_label_file
        for mask_file in glob.glob(os.path.join(scan_dir, mask_name)):
            self.mask_path = mask_file
        for land_mark_file in glob.glob(os.path.join(scan_dir, landmarks_name)):
            self.land_mark_path = land_mark_file
        for displacement_file in glob.glob(os.path.join(scan_dir, displacement_name)):
            self.displacement_path = displacement_file


class VerseScan(Scan):
    _default_volume_name = ".nii.gz"
    _default_label_name = "_seg.nii.gz"
    _default_mask_name = ""
    _default_landmarks_name = "_ctd.json"
    _default_displacement_name = ""

    def __init__(self, scan_dir, base_name, file_naming=None):
        super().__init__(base_name)
        self.path = scan_dir

        volume_name = '{}{}'.format(base_name, self._default_volume_name)
        label_name = '{}{}'.format(base_name, self._default_label_name)
        mask_name = '{}{}'.format(base_name, self._default_mask_name)
        landmarks_name = '{}{}'.format(base_name, self._default_landmarks_name)
        displacement_name = '{}{}'.format(base_name, self._default_displacement_name)

        if file_naming is not None:
            if 'volume' in file_naming.keys():
                volume_name = '{}{}'.format(base_name, file_naming['volume'])
            if 'label' in file_naming.keys():
                label_name = '{}{}'.format(base_name, file_naming['label'])
            if 'mask' in file_naming.keys():
                mask_name = '{}{}'.format(base_name, file_naming['mask'])
            if 'landmarks' in file_naming.keys():
                landmarks_name = '{}{}'.format(base_name, file_naming['landmarks'])
            if 'displacement' in file_naming.keys():
                displacement_name = '{}{}'.format(base_name, file_naming['displacement'])

        self.volume_path = os.path.join(scan_dir, volume_name)
        self.label_path = os.path.join(scan_dir, label_name)
        self.mask_path = os.path.join(scan_dir, mask_name)
        self.landmarks_path = os.path.join(scan_dir, landmarks_name)
        self.displacement_path = os.path.join(scan_dir, displacement_name)

class AbdominalMultiAtlasScan(Scan):
    _default_volume_name = "img"
    _default_label_name = "label"
    _default_mask_name = ""
    _default_landmarks_name = ""
    _default_displacement_name = ""

    def __init__(self, scan_dir, base_name, file_naming=None):
        super().__init__(base_name)
        self.path = scan_dir

        volume_name = '{}{}.nii.gz'.format(self._default_volume_name, base_name)
        label_name = '{}{}.nii.gz'.format(self._default_label_name, base_name)
        mask_name = '{}{}'.format(self._default_mask_name, base_name)
        landmarks_name = '{}{}'.format(self._default_landmarks_name, base_name)
        displacement_name = '{}{}'.format(self._default_displacement_name, base_name)

        if file_naming is not None:
            if 'volume' in file_naming.keys():
                volume_name = '{}{}.nii.gz'.format(file_naming['volume'], base_name)
            if 'label' in file_naming.keys():
                label_name = '{}{}.nii.gz'.format(file_naming['label'], base_name)
            if 'mask' in file_naming.keys():
                raise NotImplementedError
            if 'landmarks' in file_naming.keys():
                raise NotImplementedError
            if 'displacement' in file_naming.keys():
                raise NotImplementedError
        self.volume_path = os.path.join(scan_dir, volume_name)
        self.label_path = os.path.join(scan_dir, label_name)
        self.mask_path = os.path.join(scan_dir, mask_name)
        self.landmarks_path = os.path.join(scan_dir, landmarks_name)
        self.displacement_path = os.path.join(scan_dir, displacement_name)


def create_resampler(origin, spacing, size, direction=None, default_value=0, interpolator=2):
    # Resample images
    # images are resampled in new domain
    # the default value for resampling is set to a predefined value
    # (minimum possible value of the fixed image type) to use it
    # to create masks. At the end, default values are replaced with
    # the provided default value

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    resampler.SetNumberOfThreads(mp.cpu_count())

    if direction is not None:
        resampler.SetOutputDirection(direction)

    return resampler


def collect_skeleton_scans(data_path, reference_scan_name=None, body_part_choice=None, file_naming=None):
    # collect all dirs in data path
    scan_dirs = []
    for scan_dir_name in os.listdir(data_path):
        scan_dir_name = os.fsdecode(scan_dir_name)
        scan_dir = os.path.join(data_path, scan_dir_name)
        scan_dirs.append(scan_dir)

    # only return scans with a given pattern eg. lower, upper in their name
    reference_scan = None
    moving_scans = []

    for scan_dir in scan_dirs:
        if body_part_choice is None or body_part_choice in scan_dir:
            current_scan = SkeletonScan(scan_dir, file_naming)
            if reference_scan_name is not None and current_scan.name == reference_scan_name:
                reference_scan = current_scan
            else:
                moving_scans.append(current_scan)

    # return one list with all N chosen scans, or split in 1 reference_scan and N-1 moving_scans
    if reference_scan is not None:
        return moving_scans, reference_scan
    else:
        return moving_scans


def collect_verse_scans(data_path, reference_scan_name=None, body_part_choice=None, file_naming=None):
    # collect all dirs in data path
    scan_files = []
    for scan_dir_name in os.listdir(data_path):
        scan_dir_name = os.fsdecode(scan_dir_name)
        data_path_child = os.path.join(data_path, scan_dir_name)
        if os.path.isfile(data_path_child):
            scan_files.append(data_path_child)
        elif os.path.isdir(data_path_child):
            for sub_sub_dir in os.listdir(data_path_child):
                sub_sub_dir_path=os.path.join(data_path_child,sub_sub_dir)
                if os.path.isfile(sub_sub_dir_path):
                    scan_files.append(sub_sub_dir_path)

    reference_scan = None
    moving_scans = []

    all_base_names = []
    for scan_file in scan_files:
        file_base_name = os.path.basename(scan_file)
        file_name_parts = file_base_name.split('.')
        first_part = file_name_parts[0]
        base_name = first_part.split('_')[0]
        if base_name not in all_base_names:
            all_base_names.append(base_name)
            current_scan = VerseScan(os.path.dirname(scan_file), base_name, file_naming)
            if reference_scan_name is not None and current_scan.name == reference_scan_name:
                reference_scan = current_scan
            else:
                moving_scans.append(current_scan)

    # return one list with all N chosen scans, or split in 1 reference_scan and N-1 moving_scans
    if reference_scan is not None:
        return moving_scans, reference_scan
    else:
        return moving_scans

def collect_abdominal_scans(data_path, reference_scan_name=None, body_part_choice=None, file_naming=None):
    # collect all dirs in data path
    scan_files = []
    for scan_dir_name in os.listdir(data_path):
        scan_dir_name = os.fsdecode(scan_dir_name)
        data_path_child = os.path.join(data_path, scan_dir_name)
        if os.path.isfile(data_path_child):
            scan_files.append(data_path_child)
        elif os.path.isdir(data_path_child):
            for sub_sub_dir in os.listdir(data_path_child):
                sub_sub_dir_path=os.path.join(data_path_child,sub_sub_dir)
                if os.path.isfile(sub_sub_dir_path):
                    scan_files.append(sub_sub_dir_path)

    reference_scan = None
    moving_scans = []

    all_base_names = []
    for scan_file in scan_files:
        file_base_name = os.path.basename(scan_file)
        file_name_parts = file_base_name.split('.')
        first_part = file_name_parts[0]
        base_name = ''.join(filter(str.isdigit, first_part))  # extracts the numbers part e.g. 'imag00123' to '00123'
        if base_name not in all_base_names:
            all_base_names.append(base_name)
            current_scan = AbdominalMultiAtlasScan(os.path.dirname(scan_file), base_name, file_naming)
            if reference_scan_name is not None and current_scan.name == reference_scan_name:
                reference_scan = current_scan
            else:
                moving_scans.append(current_scan)

    # return one list with all N chosen scans, or split in 1 reference_scan and N-1 moving_scans
    if reference_scan is not None:
        return moving_scans, reference_scan
    else:
        return moving_scans