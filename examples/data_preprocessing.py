import collections
import json
import os

import nibabel as nib
import numpy as np
import torch as th

from examples.customData import collect_skeleton_scans, collect_verse_scans, ScanGroup, collect_abdominal_scans, \
    collect_lits_scans, Scan


def resample_to_common_domain(data_path, save_path, body_part_choice, scan_type):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    # bring all files to a joint image domain and save to disk
    scan_group = ScanGroup(all_chosen_scans)
    scan_group.compute_common_domain()
    # scan_group.resample_common_inplace(file_type='volume', default_value=-1024, interpolator=2)
    # scan_group.resample_common_inplace(file_type='label', default_value=0, interpolator=1)
    # scan_group.resample_common_inplace(file_type='mask', default_value=0, interpolator=1)
    # scan_group.save_to_disk(save_path, exist_ok=False)
    for scan in scan_group.scans:
        scan.resample_to(reference_scan=scan_group.scans[0], spacing=scan_group.common_spacing, size=scan_group.common_size, file_type='volume', interpolator=2)
        scan.resample_to(spacing=scan_group.common_spacing, size=scan_group.common_size, file_type='label', interpolator=1)
        scan.save_scan_to(save_path, exist_ok=True)

def resample_all_to_reference(data_path, save_path, body_part_choice, scan_type, reference_scan_name):
    moving_scans, reference_scan = get_reference_and_all_moving_scans(data_path, scan_type, body_part_choice, reference_scan_name)
    # bring all files to a joint image domain and save to disk
    scan_group = ScanGroup(moving_scans)
    scan_group.compute_common_domain()
    for scan in scan_group.scans:
        if scan.volume is not None:
            scan.resample_to(reference_scan=reference_scan, spacing=scan_group.common_spacing,
                             size=scan_group.common_size, file_type='volume', interpolator=2)
        if scan.label is not None:
            scan.resample_to(reference_scan=reference_scan, spacing=scan_group.common_spacing, size=scan_group.common_size, file_type='label',
                             interpolator=1)
        if scan.mask is not None:
            scan.resample_to(reference_scan=reference_scan, spacing=scan_group.common_spacing, size=scan_group.common_size, file_type='mask',
                             interpolator=1)
        scan.save_scan_to(save_path, exist_ok=True)
    reference_scan.save_scan_to(save_path, exist_ok=True)


def resample_to_common_spacing(data_path, save_path, scan_type, body_part_choice=None, spacing=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    # bring all files to a joint spacing and save to disk
    scan_group = ScanGroup(all_chosen_scans)
    if spacing is None:
        scan_group.compute_common_domain()
        spacing = scan_group.common_spacing
    for scan in scan_group.scans:
        if scan.volume is not None:
            scan.resample_to(spacing=spacing, file_type='volume', interpolator=2, default_value=-1024)
        if scan.label is not None:
            scan.resample_to(spacing=spacing, file_type='label', interpolator=1, default_value=0)
        scan.save_scan_to(save_path, exist_ok=True)

def resample_to_given_ratio_spacing(data_path, save_path, scan_type, body_part_choice=None, spacing_ratio=[1,1,1]):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    # bring all files to a joint spacing and save to disk
    scan_group = ScanGroup(all_chosen_scans)

    for scan in scan_group.scans:
        ratio=np.array(spacing_ratio)
        spacing = tuple(ratio * np.array(scan.volume.spacing))
        if scan.volume is not None:
            scan.resample_to(spacing=spacing, file_type='volume', interpolator=2, default_value=-1024)
        if scan.label is not None:
            scan.resample_to(spacing=spacing, file_type='label', interpolator=1, default_value=0)
        scan.save_scan_to(save_path, exist_ok=True)


def get_all_scans(data_path, scan_type, body_part_choice):
    if scan_type == 'NIFTY':
        all_chosen_scans = collect_skeleton_scans(data_path, reference_scan_name=None,
                                                  body_part_choice=body_part_choice)
    elif scan_type == 'VERSE':
        all_chosen_scans = collect_verse_scans(data_path, reference_scan_name=None)
    elif scan_type == 'ABD':
        all_chosen_scans = collect_abdominal_scans(data_path, reference_scan_name=None)
    elif scan_type == 'LITS':
        all_chosen_scans = collect_lits_scans(data_path, reference_scan_name=None)
    return all_chosen_scans


def get_reference_and_all_moving_scans(data_path, scan_type, body_part_choice, reference_scan_name):
    if scan_type == 'NIFTY':
        moving_scans, reference_scan = collect_skeleton_scans(data_path, reference_scan_name=reference_scan_name,
                                                  body_part_choice=body_part_choice)
    elif scan_type == 'VERSE':
        moving_scans, reference_scan = collect_verse_scans(data_path, reference_scan_name=reference_scan_name)
    elif scan_type == 'ABD':
        moving_scans, reference_scan = collect_abdominal_scans(data_path, reference_scan_name=reference_scan_name)
    elif scan_type == 'LITS':
        moving_scans, reference_scan = collect_lits_scans(data_path, reference_scan_name=reference_scan_name)
    return moving_scans, reference_scan


def shrink_all_scans(data_path, save_path, shrink_factor, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    # bring all files to a joint image domain and save to disk
    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        scan.shrink(shrink_factor)
        scan.save_scan_to(save_path, exist_ok=True)


def get_name_to_label_dict(colormap_file_path):
    labels_and_names = np.genfromtxt(colormap_file_path, delimiter=' ', usecols=(0, 1), skip_header=0, dtype=str)
    labels = labels_and_names[:, 0]
    names = labels_and_names[:, 1]
    name_to_label_dict = {}
    for label, name in zip(labels, names):
        name_to_label_dict[name.lower()] = label

    return name_to_label_dict


def relabel_all_scans(data_path, save_path, current_colormap, target_colormap, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    current_label_dict = get_name_to_label_dict(current_colormap)
    target_label_dict = get_name_to_label_dict(target_colormap)

    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        relabel_scan_inplace(scan, current_label_dict, target_label_dict)
        scan.save_scan_to(save_path, exist_ok=True)


def relabel_scan_inplace(scan, current_label_dict, target_label_dict):
    label_file = scan.label.numpy()
    label_original = label_file.copy()
    for name, label in current_label_dict.items():
        if name.lower() in target_label_dict:
            label_file[label_original == int(label)] = target_label_dict[name.lower()]
        else:
            if label_file[label_original == int(label)].size > 0:
                raise Exception('could not find {} in target dict'.format(name.lower()))
    tensor_image = th.from_numpy(label_file).unsqueeze(0).unsqueeze(0)
    scan.label.image = tensor_image


def flip_all(data_path, save_path, scan_type, flip_axes, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    for scan in all_chosen_scans:
        scan.flip(flip_axes)
        scan.save_scan_to(save_path, exist_ok=True)


def flip_left_right(data_path, save_path, scan_type, colormap=None, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    flip_axis_left_right = [True, False, False]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    if scan_type == 'VERSE':
        for scan in all_chosen_scans:
            scan.flip(flip_axis_left_right)
            scan.save_scan_to(save_path, exist_ok=True)
    elif scan_type == 'NIFTY':
        if colormap is None:
            raise Exception('We need the colormap for nifty scans, to properly flip the labels.')
        original_label_dict = get_name_to_label_dict(colormap)
        flipped_label_dict = create_flipped_label_dict(original_label_dict)
        for scan in all_chosen_scans:
            scan.flip(flip_axis_left_right)
            relabel_scan_inplace(scan, original_label_dict, flipped_label_dict)
            scan.save_scan_to(save_path, exist_ok=True)


def flip_front_back(data_path, save_path, scan_type, body_part_choice=None):
    flip_all(data_path, save_path, scan_type, flip_axes=[False, True, False], body_part_choice=body_part_choice)


def flip_all_dimensions(data_path, save_path, scan_type, body_part_choice=None):
    flip_all(data_path, save_path, scan_type, flip_axes=[True, True, True], body_part_choice=body_part_choice)


def permute_axes(data_path, save_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    new_order = [2, 0, 1]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    if scan_type == 'VERSE':
        for scan in all_chosen_scans:
            scan.permute_axes(new_order)
            scan.save_scan_to(save_path, exist_ok=True)


def save_all_scans_to(data_path, save_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        scan.save_scan_to(save_path, exist_ok=True)


def create_flipped_label_dict(original_label_dict):
    flipped_label_dict = {}
    for name, label in original_label_dict.items():
        name_parts = name.split('_')
        if len(name_parts) >= 2:
            if name_parts[-1] == 're':
                new_name = name[0:-2] + 'li'
            elif name_parts[-1] == 'li':
                new_name = name[0:-2] + 're'
            else:
                new_name = name
        else:  # nothing changes for labels without left/right
            new_name = name

        flipped_label_dict[new_name] = label
    return flipped_label_dict


def pad_all_to_fixed_size(data_path, save_path, scan_type, target_size, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    for scan in all_chosen_scans:
        scan_size = scan.volume.size
        padding_to_add = [max(0, target_size[i] - scan_size[i]) for i in range(len(scan_size))]
        scan.add_padding(padding_to_add, 0.0, file_type='label')
        scan.add_padding(padding_to_add, -1024.0, file_type='volume')
        scan.save_scan_to(save_path, exist_ok=True)


def print_all_orientations(data_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    for scan in all_chosen_scans:
        nib_img = nib.load(scan.volume_path)
        print(nib.aff2axcodes(nib_img.affine))


def change_all_orientations_to_canonical(data_path, save_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    for scan in all_chosen_scans:
        nib_img = nib.load(scan.volume_path)
        nib_seg = nib.load(scan.label_path)
        canonical_img = nib.as_closest_canonical(nib_img)
        canonical_seg = nib.as_closest_canonical(nib_seg)
        print(nib.aff2axcodes(canonical_img.affine))
        base_name_volume = os.path.basename(scan.volume_path)
        dir_name = base_name_volume.split('.')[0]
        nib.save(canonical_img, os.path.join(save_path, dir_name, os.path.basename(scan.volume_path)))
        nib.save(canonical_seg, os.path.join(save_path, dir_name, os.path.basename(scan.label_path)))


def get_all_size_priors(data_path, save_path, scan_type, body_part_choice=None):
    sizes = collections.defaultdict(list)
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)

    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        label = scan.label.numpy()
        unique_elements, counts_elements = np.unique(label, return_counts=True)
        for element, count in zip(unique_elements, counts_elements):
            sizes[int(element)].append(count)
        scan = None

    average_sizes = {}
    for key, item in sizes.items():
        average_sizes[key] = np.mean(item)

    if save_path.endswith('.json'):
        out_file = save_path
    else:
        out_file = os.path.join(save_path, 'average_sizes.json')

    with open(out_file, 'w+') as json_file:
        json.dump(average_sizes, json_file)


# somehow the skeleton scan have a mixed up A-S direction. Change it accordingly for the verse scans
def change_directions_for_all(data_path, save_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        old_direction = scan.volume.direction
        # new_direction = np.array(old_direction) * np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1])
        new_direction = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1], dtype=np.double)
        scan.label.direction = new_direction
        scan.volume.direction = new_direction
        scan.save_scan_to(save_path, exist_ok=True)


def print_size_priors(data_path):
    with open(data_path) as f:
        label_size_dict = json.load(f)
        sorted_labels_ascending = sorted([int(k) for k in label_size_dict.keys()])
        for label in sorted_labels_ascending:
            size_prior = label_size_dict[str(label)]
            print(str(int(size_prior)) + ',', end=' ')


def un_gzip_all(data_path, save_path, scan_type, body_part_choice=None):
    all_chosen_scans = get_all_scans(data_path, scan_type, body_part_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=False)
    scan_group = ScanGroup(all_chosen_scans)
    for scan in scan_group.scans:
        scan.save_scan_to(save_path, exist_ok=True, save_gzipped=False)

if __name__ == '__main__':
    data_path_verse = "/home/eva/PhD/Data/VerSe2019/Raw"

    save_path_dir_change = "/home/eva/PhD/Data/VerSe2019/Processed/directionChanged"
    save_path_resample = "/home/eva/PhD/Data/VerSe2019/Processed/resampled"
    dummy_path = '/home/eva/PhD/Data/VerSe2019/Processed/padded_debug'
    dummy_path_out='/home/eva/PhD/Data/VerSe2019/Processed/padded_ro'
    scan_type = "VERSE"
    # change_directions_for_all('/home/eva/PhD/Data/VerSe2019/Processed/padded_debug', '/home/eva/PhD/Data/VerSe2019/Processed/padded_ro', scan_type=scan_type,  body_part_choice=None)
    # resample_to_common_spacing(save_path_dir_change, save_path_resample, scan_type=scan_type,  body_part_choice=None, spacing=[1,1,1])
    # shrink_in_path = save_path_verse
    shrink_out_path = "/home/eva/PhD/Data/VerSe2019/Processed/halfres"
    sizes_out_path = "/home/eva/PhD/Data/VerSe2019/Processed/size_priors"
    # shrink_all_scans(save_path_resample, shrink_out_path, shrink_factor=2, scan_type=scan_type)

    # get_all_size_priors(shrink_out_path, sizes_out_path, scan_type='VERSE')
    # get_all_size_priors('/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres', sizes_out_path, scan_type='NIFTY')

    colormap_current = '/home/eva/PhD/Data/VerSe2019/colormap.ctbl'
    target_colormap = '/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base/colormap/colormap.ctbl'
    relabel_path = '/home/eva/PhD/Data/VerSe2019/Processed/halfres_relabeled'
    # relabel_all_scans(shrink_out_path, relabel_path, colormap_current, target_colormap,
    #                   scan_type='VERSE')

    flipped_path = '/home/eva/PhD/Data/VerSe2019/Processed/flipped'
    # flip_left_right(relabel_path, flipped_path, scan_type='VERSE')

    nifty_halfres = '/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres'
    nifty_flipped = '/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres_flipped'

    # flip_left_right(nifty_halfres, nifty_flipped, scan_type='NIFTY', colormap=target_colormap, body_part_choice=None)

    padded_path_flipped = '/home/eva/PhD/Data/VerSe2019/Processed/padded_flipped'
    padded_path = '/home/eva/PhD/Data/VerSe2019/Processed/padded'
    # pad_all_to_fixed_size(flipped_path, padded_path_flipped, scan_type='VERSE', target_size=[256,256,256])
    # pad_all_to_fixed_size(relabel_path, padded_path, scan_type='VERSE', target_size=[256,256,256])

    # get_all_orientations(data_path_verse, 'VERSE')
    print('halfres')
    # print_all_orientations(nifty_halfres, 'NIFTY')
    print('flipped')
    # print_all_orientations(nifty_flipped, 'NIFTY')

    permute_path_out = '/home/eva/PhD/Data/VerSe2019/Processed/padded_permuted'
    # permute_axes(padded_path, permute_path_out, 'VERSE')

    flipped_all_path='/home/eva/PhD/Data/VerSe2019/Processed/padded_permuted_flipped'
    # flip_all_dimensions(permute_path_out, flipped_all_path, 'VERSE')
    flipped_le_ri_path = '/home/eva/PhD/Data/VerSe2019/Processed/padded_permuted_flipped_le_ri'
    # flip_left_right(flipped_all_path, flipped_le_ri_path, scan_type='VERSE')

    # print_all_orientations(reorient_res_path, scan_type='VERSE')
    reorient_path2 = '/home/eva/PhD/Data/VerSe2019_reorient_res2'
    # change_all_orientations_to_canonical(reorient_res_path, reorient_path2, 'VERSE')

    # print_size_priors('/home/eva/PhD/Data/VerSe2019/Processed/size_priors/average_sizes.json')

    reorient_path = '/home/eva/PhD/Data/VerSe2019_reorient'
    reorient_res_path = '/home/eva/PhD/Data/VerSe2019_reorient_res'
    # change_all_orientations_to_canonical(dummy_path, dummy_path_out, 'VERSE')

    # un_gzip_all('/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres_flipped','/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres_flipped_unzipped', 'NIFTY', None)
    # un_gzip_all('/home/eva/PhD/Data/VerSe2019/Processed/final_flipped','/home/eva/PhD/Data/VerSe2019/Processed/final_flipped_unzipped', 'VERSE', None)
    # relabel_path='/home/eva/PhD/Data/VerSe2019/Processed/upper_body_relabelled_flipped'
    # colormap_full_body='/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base/colormap/colormap.ctbl'
    # colormap_upper_body='/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base/colormap_upper_lower_separate/upper_body_labels.ctbl'
    # relabel_all_scans('/home/eva/PhD/Data/VerSe2019/Processed/final_flipped', relabel_path, colormap_full_body, colormap_upper_body,
    #                   scan_type='VERSE', body_part_choice=None)

    # resample_to_common_spacing('/home/eva/PhD/Data/MultiAtlasAbdomen/original', '/home/eva/PhD/Data/MultiAtlasAbdomen/res', scan_type='ABD',  body_part_choice=None, spacing=[2,2,2])
    pad_all_to_fixed_size('/home/eva/PhD/Data/MultiAtlasAbdomen/res', '/home/eva/PhD/Data/MultiAtlasAbdomen/pad', scan_type='ABD', target_size=[256,256,128])

