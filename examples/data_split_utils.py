import collections
import csv
import os

import numpy as np

from examples import customData
from examples.customData import collect_skeleton_scans, collect_verse_scans, collect_abdominal_scans, collect_lits_scans

ScanGroupToInlude = collections.namedtuple('ScanGroupToInlude', ['root_dir', 'scan_type', 'name'])





def scans_to_csv(scan_groups_to_include: ScanGroupToInlude, destination_path: str, body_part_choice=None,
                 label_type='bones', volume_type='volume'):
    scan_combination_name = _create_destination_dir_name(scan_groups_to_include)

    all_scans = _collect_all_scans(scan_groups_to_include, body_part_choice, label_type, volume_type)

    all_file_names = []

    # in case of duplicates, rename scans
    for scan in all_scans:
        i = 0
        scan_name = scan.name
        while scan_name in all_file_names:
            scan_name = scan.name
            scan_name = '{}_{}'.format(scan_name, i)
            i = i + 1
        if i > 0:
            scan.name = scan_name
        all_file_names.append(scan.name)

    destination_path = os.path.join(destination_path, scan_combination_name)
    write_nifty_csvs(all_scans, destination_path, volume_type, label_type)


def _collect_all_scans(scan_groups_to_include, body_part_choice=None, label_type='bones', volume_type='volume'):
    all_scans = []  # type: (list[SkeletonScan])
    scan_group: ScanGroupToInlude
    for scan_group in scan_groups_to_include:

        if scan_group.scan_type == customData.SkeletonScan:
            file_naming = {'volume': '{}.nii.gz'.format(volume_type),
                           'label': '{}.nii.gz'.format(label_type)}
            scans_to_add = collect_skeleton_scans(scan_group.root_dir,
                                                  reference_scan_name=None,
                                                  body_part_choice=body_part_choice,
                                                  file_naming=file_naming)  # type: (list[SkeletonScan])

        elif scan_group.scan_type == customData.VerseScan:
            file_naming_verse = None
            if label_type == 'binary_bones':
                file_naming_verse = {'label': '_seg_binary.nii.gz'}
            scans_to_add = collect_verse_scans(scan_group.root_dir, file_naming=file_naming_verse)

        elif scan_group.scan_type == customData.AbdominalMultiAtlasScan:
            scans_to_add = collect_abdominal_scans(scan_group.root_dir)
        elif scan_group.scan_type == customData.Lits17Scan:
            scans_to_add = collect_lits_scans(scan_group.root_dir)

        all_scans = all_scans + scans_to_add
    return all_scans


def _create_destination_dir_name(scan_groups_to_include):
    run_type = ''
    for i, scan_group in enumerate(scan_groups_to_include):
        if i == 0:
            run_type = scan_group.name
        else:
            run_type = '{}_{}'.format(run_type, scan_group.name)
    return run_type


def write_nifty_csvs(all_scans, csv_path, volume_type, label_type):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path, exist_ok=False)

    # VOLUME
    file_mapping_csv_path = os.path.join(csv_path, 'volume_file_{}.csv'.format(volume_type))
    with open(file_mapping_csv_path, mode='w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for scan in all_scans:
            filewriter.writerow([scan.name, scan.volume_path])
    # LABEL
    file_mapping_csv_path = os.path.join(csv_path, 'label_file_{}.csv'.format(label_type))
    with open(file_mapping_csv_path, mode='w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for scan in all_scans:
            filewriter.writerow([scan.name, scan.label_path])
    # DATA SPLIT
    data_split_csv_path = os.path.join(csv_path, 'data_split.csv')
    with open(data_split_csv_path, mode='w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for scan in all_scans:
            if '005' in scan.name:
                filewriter.writerow([scan.name, 'inference'])
            elif '004' in scan.name:
                filewriter.writerow([scan.name, 'validation'])
            else:
                filewriter.writerow([scan.name, 'training'])


if __name__ == '__main__':
    # SKELETONS AND VERSE
    scan_group_base = ScanGroupToInlude('/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres',
                                        customData.SkeletonScan, 'base')
    scan_group_pca = ScanGroupToInlude('/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/pca_halfres',
                                       customData.SkeletonScan, 'pca')
    scan_group_flipped = ScanGroupToInlude(
        '/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_halfres_flipped', customData.SkeletonScan,
        'flip')
    scan_group_verse = ScanGroupToInlude('/home/eva/PhD/Data/VerSe2019/Processed/final', customData.VerseScan, 'verse')
    scan_group_verse_flipped = ScanGroupToInlude('/home/eva/PhD/Data/VerSe2019/Processed/final_flipped',
                                                 customData.VerseScan, 'v-flip')
    scan_groups_to_include_skels = [scan_group_base, scan_group_pca, scan_group_flipped, scan_group_verse,
                                    scan_group_verse_flipped]

    # ABDOMINALS
    scan_group_abd = ScanGroupToInlude('/home/eva/PhD/Data/MultiAtlasAbdomen/Processed/final',
                                       customData.AbdominalMultiAtlasScan, 'abd')
    scan_group_lits = ScanGroupToInlude('/home/eva/PhD/Data/LITS17/Processed/Final', customData.Lits17Scan, 'lits')

    scan_groups_to_include_abdominals = [scan_group_abd, scan_group_lits]

    # destination
    csv_path = "/home/eva/PhD/Data/MultiAtlasAbdomen/csv_files"

    # run
    # scans_to_csv(scan_groups_to_include_abdominals, csv_path)

    data_split = '/home/eva/PhD/Data/blade_data/MultiAtlasAbdomen/csv_files/abd_lits/data_split_1.csv'
    volume_file = '/home/eva/PhD/Data/blade_data/MultiAtlasAbdomen/csv_files/abd_lits/volume_file_volume.csv'
    label_file = '/home/eva/PhD/Data/blade_data/MultiAtlasAbdomen/csv_files/abd_lits/label_file_bones.csv'

