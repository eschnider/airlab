import os
from typing import List, Any

import sys
from examples.customRegistration import RigidRegistrator, BsplineRegistrator
from examples.customData import RegistrationData, ScanGroup, SkeletonScan
import numpy as np
from sklearn.decomposition import PCA
import torch as th
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al


def main(body_part_choice='lower', reference_scan_name='001_lower'):
    IS_DEBUG = False

    DO_RESAMPLE = False
    DO_RIGID_REGISTRATION = True
    DO_BSPLINE_REGISTRATION = True
    DO_PCA = True
    DO_SCANS_TO_CSV = False
    DO_TEST_REAPPLY_DISPLACEMENT = False

    test_scan_name = '005_{}'.format(body_part_choice)
    n_samples_to_be_generated = 10

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")
    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")
    # directory to store results

    # load all data wee need
    if IS_DEBUG:
        data_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base_debug"
        resampled_data_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/resampled_debug"
        rigid_registered_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/rigid_registered_debug"
        bspline_registered_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/bspline_registered_debug"
        pca_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/pca_debug"
        test_displacement_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/test_displacement_debug"
    else:
        data_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/base"
        resampled_data_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/resampled"
        rigid_registered_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/rigid_registered"
        bspline_registered_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/bspline_registered"
        pca_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/pca"
        test_displacement_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/test_displacement"

    if DO_RESAMPLE:
        resample_to_common_domain(data_path, resampled_data_path, body_part_choice)

    # rigid registration towards the mean scan (this one is the moving image)

    if DO_RIGID_REGISTRATION:
        moving_scans, reference_scan = collect_skeleton_scans(resampled_data_path,
                                                              reference_scan_name=reference_scan_name,
                                                              body_part_choice=body_part_choice)  # type: (list[Scan], Scan)

        down_sample_factor = [8, 8, 8]

        perform_and_save_rigid_registration_on_all_scans(reference_scan, moving_scans, rigid_registered_path,
                                                         down_sample_factor, device, dtype)
    if DO_BSPLINE_REGISTRATION:
        reference_scan = SkeletonScan(os.path.join(resampled_data_path, reference_scan_name))
        file_naming = {'volume': 'rigid_warped_image.nii.gz', 'mask': 'rigid_warped_mask.nii.gz'}
        moving_scans = collect_skeleton_scans(rigid_registered_path,
                                              reference_scan_name=None,
                                              body_part_choice=body_part_choice,
                                              file_naming=file_naming)  # type: (list[Scan])

        perform_and_save_bspline_registration_on_all_scans(reference_scan, moving_scans, bspline_registered_path,
                                                           device, dtype)

    if DO_PCA:
        file_naming = {'volume': 'bspline_warped_image.nii.gz', 'displacement': 'bspline_displacement_image_unit.vtk'}
        moving_scans = collect_skeleton_scans(bspline_registered_path,
                                              reference_scan_name=None,
                                              body_part_choice=body_part_choice,
                                              file_naming=file_naming)  # type: (list[Scan])

        displacement_fields = [scan.displacement for scan in moving_scans if
                               scan.name != test_scan_name]  # type: List[al.Displacement]
        X, displacement_field_original_shape = create_component_matrix_for_PCA(displacement_fields)

        # Compute PCA from X
        sample_number = X.shape[0]
        pca = PCA(n_components=sample_number)
        pca.fit(X)

        print("=================================================================")
        print("PCA done")

        for n_th_sample in range(n_samples_to_be_generated):
            # sample new deformation fields from shape model
            # sample up to six standard deviations away from the mean)
            alphas_for_sampling = np.random.uniform(-6, 6, pca.n_components)
            new_feature_vector_sample = sample_from_pca(pca, alphas=alphas_for_sampling)
            # reshape the newly sampled deformation field
            new_displacement = create_displacement_from_feature_vector(new_feature_vector_sample,
                                                                       displacement_field_original_shape,
                                                                       displacement_fields)

            scan_save_dir = os.path.join(pca_path, '{}_pca_{}'.format(reference_scan_name, n_th_sample))
            # create new bones scan
            file_type = 'bones.nii.gz'
            file_naming = {'volume': file_type}
            reference_scan = SkeletonScan(os.path.join(data_path, reference_scan_name), file_naming=file_naming)
            warped_image = resize_and_apply_displacement_to_image(reference_scan.volume, new_displacement)
            file_name = file_type
            save_file(warped_image, scan_save_dir, file_name)
            # create new CT scan:
            file_type = 'volume.nii.gz'
            file_naming = {'volume': file_type}
            reference_scan_ct = SkeletonScan(os.path.join(data_path, reference_scan_name), file_naming=file_naming)
            warped_image_ct = resize_and_apply_displacement_to_image(reference_scan_ct.volume, new_displacement)
            file_name_ct = file_type
            save_file(warped_image_ct, scan_save_dir, file_name_ct)

            print("New sample saved")

        print("=================================================================")
        print("All new samples generated and saved")

    if DO_TEST_REAPPLY_DISPLACEMENT:
        reference_scan = SkeletonScan(os.path.join(resampled_data_path, reference_scan_name))
        file_naming = {'volume': 'bspline_warped_image.nii.gz',
                       'displacement': 'bspline_displacement_image_unit.vtk'}
        moving_scans = collect_skeleton_scans(bspline_registered_path,
                                              reference_scan_name=None,
                                              body_part_choice=body_part_choice,
                                              file_naming=file_naming)  # type: (list[SkeletonScan])

        test_displacement = moving_scans[0].displacement
        test_image_warped = resize_and_apply_displacement_to_image(reference_scan.volume, test_displacement)
        scan_save_dir = test_displacement_path
        file_name = 'test_{}.nii.gz'.format(moving_scans[0].name)
        save_file(test_image_warped, scan_save_dir, file_name)


def save_file(warped_image, scan_save_dir, file_name):
    if not os.path.exists(scan_save_dir):
        os.makedirs(scan_save_dir, exist_ok=False)
    warped_image.write(os.path.join(scan_save_dir, file_name))


def create_displacement_from_feature_vector(feature_vector, displacement_field_original_shape, displacement_fields):
    new_deformation_tensor = np.reshape(feature_vector, displacement_field_original_shape)
    new_deformation = displacement_fields[0].image.clone()  # type: al.Displacement
    new_deformation_tensor = th.from_numpy(new_deformation_tensor)
    new_deformation.image = new_deformation_tensor.squeeze().squeeze()
    new_deformation.squeeze().squeeze()

    return new_deformation


def apply_displacement_to_image(reference_scan, displacement):
    upsampled_deformation_field = al.transformation.utils.upsample_displacement(displacement.image.squeeze(),
                                                                                reference_scan.volume.size)
    def_field = th.unsqueeze(upsampled_deformation_field, 0).to(dtype=th.float32)
    warped_image = al.transformation.utils.warp_image(reference_scan.volume, def_field, interpolation_mode='nearest',
                                                      padding_mode='zeros')
    return warped_image


def sample_from_pca(pca, alphas=None):
    if alphas is None:
        alphas = np.random.standard_normal(pca.n_components)

    new_sample = np.zeros(pca.n_features_)
    alphas = np.random.standard_normal(pca.n_components)
    for component, eigenvalue, alpha in zip(pca.components_, pca.singular_values_, alphas):
        new_sample = new_sample + np.sqrt(eigenvalue) * alpha * component
    return new_sample


def create_component_matrix_for_PCA(displacement_fields):
    displacement_field_spatial_shape = displacement_fields[0].size
    ndims_displacement_vectors = len(displacement_field_spatial_shape)
    feature_number = ndims_displacement_vectors * np.prod(displacement_field_spatial_shape)
    sample_number = len(displacement_fields)
    # vectorize all deformation fields and save in X
    X = np.zeros((sample_number, feature_number))
    for index, displacement_field in enumerate(displacement_fields):
        displacement = displacement_field.image.squeeze()  # type: al.Displacement
        displacement_field_array = displacement.numpy()
        displacement_field_reshaped = np.reshape(displacement_field_array, [feature_number, 1, 1, 1])
        displacement_field_reshaped = np.squeeze(displacement_field_reshaped)
        X[index, :] = displacement_field_reshaped
    displacement_field_original_shape = displacement_field_array.shape
    return X, displacement_field_original_shape


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


def resample_to_common_domain(data_path, save_path, body_part_choice):
    lower_limb_scans = collect_skeleton_scans(data_path, reference_scan_name=None, body_part_choice=body_part_choice)
    moving_images = []
    for scan in lower_limb_scans:
        moving_images.append(scan.volume)
    # bring all files to a joint image domain and save to disk
    scan_group = ScanGroup(lower_limb_scans)
    scan_group.compute_common_domain()
    scan_group.resample_inplace(file_type='volume', default_value=0, interpolator=1)
    scan_group.resample_inplace(file_type='mask', default_value=0, interpolator=1)
    scan_group.save_to_disk(save_path, exist_ok=False)


def perform_and_save_bspline_registration_on_all_scans(reference_scan, moving_scans, save_dir, device=th.device("cpu"),
                                                       dtype=th.float32, using_landmarks=False):
    # kernel registration towards the mean scan
    start = time.time()

    # Do you want to use landmarks?

    fixed_points = None
    moving_points = None

    f_image = reference_scan.volume
    f_mask = reference_scan.mask

    f_image.to(dtype, device)
    f_mask.to(dtype, device)

    if using_landmarks:
        fixed_points = reference_scan.landmarks

    number_of_iterations = [120, 50, 0, 0]
    pyramid_sampling_factors = [[16, 16, 16], [8, 8, 8]]  # a full sized image always tops off the pyramid

    fixed_image_pyramid = al.create_image_pyramid(f_image, pyramid_sampling_factors)
    fixed_mask_pyramid = al.create_image_pyramid(f_mask, pyramid_sampling_factors)

    fixed_registration_data = RegistrationData(fixed_image_pyramid, fixed_mask_pyramid, fixed_points)

    for moving_scan in moving_scans:

        m_image = moving_scan.volume
        m_mask = moving_scan.mask

        m_image.to(dtype, device)
        m_mask.to(dtype, device)

        # create image pyramid size/8 size/4, size/2, size/1
        moving_image_pyramid = al.create_image_pyramid(m_image, pyramid_sampling_factors)
        moving_mask_pyramid = al.create_image_pyramid(m_mask, pyramid_sampling_factors)

        if using_landmarks:
            moving_points = moving_scan.landmarks
            initial_tre = al.Points.TRE(fixed_points, moving_points)
            print("initial TRE: " + str(initial_tre))

        moving_registration_data = RegistrationData(moving_image_pyramid, moving_mask_pyramid, moving_points)

        bspline_registrator = BsplineRegistrator(number_of_iterations, save_intermediate_displacements=False, device=device, dtype=dtype)
        bspline_registrator.using_landmarks = using_landmarks
        fixed_to_moving_displacement, upsampled_displacement = bspline_registrator.perform_fixed_to_moving_registration(fixed_registration_data, moving_registration_data)

        warped_image = al.transformation.utils.warp_image(f_image, upsampled_displacement, interpolation_mode='nearest',
                                                          padding_mode='zeros')
        # domain measures
        displacement_image = al.create_displacement_image_from_image(fixed_to_moving_displacement, fixed_image_pyramid[-2])
        end = time.time()
        print("Registration done in: ", end - start, " seconds")
        # in order to not invert the displacement field, the fixed points are transformed to match the moving points
        if using_landmarks:
            print("Initial TRE: " + str(initial_tre))
            fixed_points_transformed = al.Points.transform(fixed_points, upsampled_displacement)
            print("Final TRE: " + str(al.Points.TRE(moving_points, fixed_points_transformed)))
        # write result images
        print("writing results")

        scan_save_dir = os.path.join(save_dir, moving_scan.name)
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir, exist_ok=False)
        warped_image.write('{}/bspline_warped_image.nii.gz'.format(scan_save_dir))
        # we only save the downsampled displacement, otherwise it gets HUGE
        displacement_image.write('{}/bspline_displacement_image_unit.vtk'.format(scan_save_dir))
        if using_landmarks:
            al.Points.write('{}/bspline_fixed_points_transformed.vtk'.format(save_dir), fixed_points_transformed)
            al.Points.write('{}/bspline_moving_points_aligned.vtk'.format(save_dir), moving_points)

    print("=================================================================")
    print("All registrations done")


def perform_and_save_rigid_registration_on_all_scans(reference_scan, moving_scans, save_dir,
                                                     down_sample_factor=None, device=th.device("cpu"),
                                                     dtype=th.float32):
    f_image_original_size = reference_scan.volume
    f_mask = reference_scan.mask
    if down_sample_factor is not None or down_sample_factor == [1, 1, 1]:
        f_image = al.utils.image.resize_image(f_image_original_size, down_sample_factor)
        f_mask = al.utils.image.resize_image(f_mask, down_sample_factor)
    else:
        f_image = f_image_original_size

    f_image.to(dtype, device)
    f_mask.to(dtype, device)

    fixed_data = RegistrationData(f_image, f_mask)

    for moving_scan in moving_scans:

        m_image_original_size = moving_scan.volume
        m_mask_original_size = moving_scan.mask

        if down_sample_factor is not None or down_sample_factor == [1, 1, 1]:
            m_image = al.utils.image.resize_image(m_image_original_size, down_sample_factor)
            m_mask = al.utils.image.resize_image(m_mask_original_size, down_sample_factor)
        else:
            m_image = m_image_original_size
            m_mask = m_mask_original_size

        m_image.to(dtype, device)
        m_mask.to(dtype, device)

        moving_data = RegistrationData(m_image, m_mask)

        number_of_iterations = 120

        rigid_registrator = RigidRegistrator(number_of_iterations, device, dtype)

        print("perform registration")
        start = time.time()

        displacement = rigid_registrator.perform_registration(fixed_data, moving_data)

        print("upsample displacement field")
        upsampled_displacement = al.transformation.utils.upsample_displacement(displacement.clone().to(device='cpu'),
                                                                               m_image_original_size.size,
                                                                               interpolation="linear")

        warped_image = al.transformation.utils.warp_image(m_image_original_size, upsampled_displacement,
                                                          interpolation_mode='bilinear',
                                                          padding_mode='border')
        warped_mask = al.transformation.utils.warp_image(m_mask_original_size, upsampled_displacement,
                                                         interpolation_mode='bilinear',
                                                         padding_mode='border')

        end = time.time()
        print("Registration done in: ", end - start, " seconds")

        # write result images
        print("writing results")

        upsampled_displacement_image = al.create_displacement_image_from_image(upsampled_displacement,
                                                                               f_image_original_size)

        scan_save_dir = os.path.join(save_dir, moving_scan.name)
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir, exist_ok=False)
        warped_image.write('{}/rigid_warped_image.nii.gz'.format(scan_save_dir))
        warped_mask.write('{}/rigid_warped_mask.nii.gz'.format(scan_save_dir))
        upsampled_displacement_image.write('{}/rigid_displacement_image_unit.nii.gz'.format(scan_save_dir))
    print("=================================================================")
    print("Rigid registration done")


if __name__ == "__main__":
    # main(body_part_choice='lower', reference_scan_name='001_lower')
    # main(body_part_choice='lower', reference_scan_name='002_lower')
    # main(body_part_choice='lower', reference_scan_name='003_lower')
    main(body_part_choice='lower', reference_scan_name='001_lower')
    main(body_part_choice='lower', reference_scan_name='002_lower')
    main(body_part_choice='lower', reference_scan_name='003_lower')

    # main(body_part_choice='all')
