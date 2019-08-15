import glob
import multiprocessing as mp
import os

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

    def resample_inplace(self, file_type='all', default_value=0.0, interpolator=2):

        if self.common_extent is None or self.common_spacing is None or self.common_size is None:
            self.compute_common_domain()

        resampler = create_resampler(self.common_origin, self.common_spacing, self.common_size, default_value,
                                     interpolator)
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
            scan_save_path = os.path.join(save_dir, scan.name)
            if not os.path.exists(scan_save_path):
                os.makedirs(scan_save_path, exist_ok=False)
            volume_file_path = os.path.join(scan_save_path, os.path.basename(scan.volume_path))
            scan.volume.write(volume_file_path)
            scan.mask.write(os.path.join(scan_save_path, os.path.basename(scan.mask_path)))


class Scan:
    name = None
    path = None
    volume_path = None
    mask_path = None
    displacement_path = None

    def __init__(self, name):
        self.name = name
        self._volume = None
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


class SkeletonScan(Scan):
    _default_volume_name = "bones.nii.gz"
    _default_mask_name = "ROI*.nii.gz"
    _default_landmarks_name = "landmark.fcsv"
    _default_displacement_name = "bspline_displacement_image_unit.vtk"

    def __init__(self, scan_dir, file_naming=None):
        super().__init__(os.path.basename(scan_dir))
        self.path = scan_dir

        volume_name = self._default_volume_name
        mask_name = self._default_mask_name
        landmarks_name = self._default_landmarks_name
        displacement_name = self._default_displacement_name

        if file_naming is not None:
            if 'volume' in file_naming.keys():
                volume_name = file_naming['volume']
            if 'mask' in file_naming.keys():
                mask_name = file_naming['mask']
            if 'landmarks' in file_naming.keys():
                landmarks_name = file_naming['landmarks']
            if 'displacement' in file_naming.keys():
                displacement_name = file_naming['displacement']

        for bone_label_file in glob.glob(os.path.join(scan_dir, volume_name)):
            self.volume_path = bone_label_file
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

    def __init__(self, scan_dir, base_name):
        super().__init__(base_name)
        self.path = scan_dir

        volume_name = '{}{}'.format(base_name, self._default_volume_name)
        label_name = '{}{}'.format(base_name, self._default_label_name)
        mask_name = '{}{}'.format(base_name, self._default_mask_name)
        landmarks_name = '{}{}'.format(base_name, self._default_landmarks_name)
        displacement_name = '{}{}'.format(base_name, self._default_displacement_name)

        self.volume_path = os.path.join(scan_dir, volume_name)
        self.label_path = os.path.join(scan_dir, label_name)
        self.mask_path = os.path.join(scan_dir, mask_name)
        self.landmarks_path = os.path.join(scan_dir, landmarks_name)
        self.displacement_path = os.path.join(scan_dir, displacement_name)


def create_resampler(origin, spacing, size, default_value=0, interpolator=2):
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

    return resampler