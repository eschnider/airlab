import os
import sys
import logging
import argparse
from examples.customData import ScanGroup, SkeletonScan, collect_skeleton_scans, \
    collect_verse_scans

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al


def main(data_path, save_path, spacing=None, size=None, body_part_choice='lower', reference_path=None):
    to_be_resampled_scans = collect_verse_scans(data_path, reference_scan_name=None, body_part_choice=body_part_choice)

    reference_scan = None
    if reference_path is not None:
        reference_scan = SkeletonScan(reference_path)

    for scan in to_be_resampled_scans:
        scan.resample_to(reference_scan=reference_scan, spacing=spacing, file_type='volume', default_value=-1024, interpolator=2)
        scan.resample_to(reference_scan=reference_scan, spacing=spacing, file_type='label', default_value=0, interpolator=1)
        scan.save_scan_to(save_path)


def decide_on_reference(args):
    reference_path = None
    spacing = None
    size = None

    if args.reference_path is not None:
        reference_path = args.reference_path
        if args.spacing is not None or args.size is not None:
            logging.warning('Will ignore spacing and size and go with the reference')
    elif args.spacing is not None:
        spacing = args.spacing
        if args.size is not None:
            logging.warning('Will ignore size and go with spacing')
    elif args.size is not None:
        size = args.size
    else:
        raise Exception('Must provide either reference_dir, or spacing, or size')

    return reference_path, spacing, size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='save data with new size or spacing')
    parser.add_argument('--data_path', type=str,
                        help='the path to the data', required=True)
    parser.add_argument('--save_path', type=str,
                        help='where the data will be saved')
    parser.add_argument('--reference_path', type=str,
                        help='resample to the same spacing as this reference')
    parser.add_argument('--spacing', type=float, nargs='*',
                        help='which new spacing')
    parser.add_argument('--size', type=int,
                        help='which new size')

    args = parser.parse_args()

    for arg_path in [args.data_path, args.reference_path]:
        if arg_path is not None and not os.path.isdir(arg_path):
            exception_string = 'provided path {0} must be a directory'.format(arg_path)
            raise Exception(exception_string)

    data_path = args.data_path

    if args.save_path is None or args.save_path == '':
        base_dir = os.path.dirname(data_path)
        new_base_name = '{}_resampled'.format(os.path.basename(data_path))
        save_path = os.path.join(base_dir, new_base_name)
    else:
        save_path = args.save_path

    reference_path, spacing, size = decide_on_reference(args)

    main(data_path, save_path, spacing=spacing, size=size, body_part_choice='all', reference_path=reference_path)
