import os
from typing import List

import sys

from airlab import transformation as T
from examples.customRegistration import RigidRegistrator, BsplineRegistrator, SimilarityRegistrator, AffineRegistrator
from examples.customData import RegistrationData, SkeletonScan, collect_skeleton_scans, Scan
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch as th
import torch.nn.functional as F
import time

from examples.data_preprocessing import resample_to_common_domain, resample_all_to_reference

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al


def main(body_part_choice='lower', reference_scan_name='001_lower', data_path_root='/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/new_PCA'):
    IS_DEBUG = False

    DO_RESAMPLE = False
    DO_RIGID_REGISTRATION = False
    DO_BSPLINE_REGISTRATION = True
    DO_PCA = True
    DO_TEST_REAPPLY_DISPLACEMENT = False
    DO_PRINT_PRIMARY_PCA_DIRECTIONS = False

    test_scan_name = '005_{}'.format(body_part_choice)
    n_samples_to_be_generated = 10

    # set the used data type
    dtype = th.float32
    # set the device for the computation to CPU
    # device = th.device("cpu")
    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")
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
        data_path = os.path.join(data_path_root, 'base')
        resampled_data_path = os.path.join(data_path_root, "resampled_{}".format(body_part_choice))
        rigid_registered_path = os.path.join(data_path_root, "rigid_registered_towards_{}".format(reference_scan_name))
        bspline_registered_path = os.path.join(data_path_root, "bspline_registered_from_{}".format(reference_scan_name))
        pca_path = os.path.join(data_path_root, "pca")
        test_displacement_path = "/home/eva/PhD/Data/WholeSkeletonsCleaned/Processed/with_labels/test_displacement"

    if DO_RESAMPLE:
        print('start resampling')
        # resample_to_common_domain(data_path, resampled_data_path, body_part_choice, 'NIFTY')
        resample_all_to_reference(data_path, resampled_data_path, body_part_choice, 'NIFTY', reference_scan_name)

    # rigid registration towards the mean scan (this one is the moving image)

    if DO_RIGID_REGISTRATION:
        moving_scans, reference_scan = collect_skeleton_scans(resampled_data_path,
                                                              reference_scan_name=reference_scan_name,
                                                              body_part_choice=body_part_choice)  # type: (list[Scan], Scan)

        down_sample_factor = [8,8,8]

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
        moving_scans: List[Scan] = collect_skeleton_scans(bspline_registered_path,
                                              reference_scan_name=None,
                                              body_part_choice=body_part_choice,
                                              file_naming=file_naming)

        displacement_fields: List[al.Displacement] = [scan.displacement for scan in moving_scans if
                               scan.name != test_scan_name]
        X, displacement_field_original_shape = create_component_matrix_for_PCA(displacement_fields)

        # Compute PCA from X
        sample_number = X.shape[0]
        pca = PCA(n_components=sample_number)
        pca.fit(X)

        print("=================================================================")
        print("PCA done")


        debug_plot(moving_scans[0].volume, moving_scans[1].volume,
                   '/home/eva/PhD/Data/Test/prior_to_rigid_registration.png')

        for n_th_sample in range(n_samples_to_be_generated):
            # sample new deformation fields from shape model
            # sample up to six standard deviations away from the mean)
            # alphas_for_sampling = np.random.uniform(-6, 6, pca.n_components)
            alphas_for_sampling=None
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
            warped_image = resize_and_apply_displacement_to_image(reference_scan.volume, new_displacement,
                                                                  padding_mode='zeros', interpolation_mode='nearest')
            file_name = file_type
            save_file(warped_image, scan_save_dir, file_name)
            # create new CT scan:
            file_type = 'volume.nii.gz'
            file_naming = {'volume': file_type}
            reference_scan_ct = SkeletonScan(os.path.join(data_path, reference_scan_name), file_naming=file_naming)
            warped_image_ct = resize_and_apply_displacement_to_image(reference_scan_ct.volume, new_displacement,
                                                                     padding_mode='border', interpolation_mode='bilinear')
            file_name_ct = file_type
            save_file(warped_image_ct, scan_save_dir, file_name_ct)

            print("New sample saved")

        print("=================================================================")
        print("All new samples generated and saved")

    if DO_PRINT_PRIMARY_PCA_DIRECTIONS:

            reference_scan_ct = SkeletonScan(os.path.join(data_path, reference_scan_name), file_naming=file_naming)
            warped_image_cts=[]
            for direction in range(pca.n_components):
                alphas_for_sampling=[0 for i in range(pca.n_components)]
                alphas_for_sampling[direction]=np.sqrt(2*pca.singular_values_[direction])
                new_feature_vector_sample = sample_from_pca(pca, alphas=alphas_for_sampling)
                new_displacement = create_displacement_from_feature_vector(new_feature_vector_sample,
                                                                           displacement_field_original_shape,
                                                                           displacement_fields)

                warped_image_cts.append(resize_and_apply_displacement_to_image(reference_scan_ct.volume, new_displacement,
                                                                         padding_mode='border', interpolation_mode='bilinear'))

            debug_plot(warped_image_cts[0], warped_image_cts[1],
                           '/home/eva/PhD/Data/Test/pca_0_1.png')
            debug_plot(warped_image_cts[2], warped_image_cts[3],
                           '/home/eva/PhD/Data/Test/pca_2_3.png')
            debug_plot(warped_image_cts[4], warped_image_cts[5],
                           '/home/eva/PhD/Data/Test/pca_4_5.png')


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


def resize_and_apply_displacement_to_image(volume, displacement, interpolation_mode='nearest', padding_mode='zeros'):
    upsampled_deformation_field = al.transformation.utils.upsample_displacement(displacement.image.squeeze(),
                                                                                volume.size)
    def_field = th.unsqueeze(upsampled_deformation_field, 0).to(dtype=th.float32)
    warped_image = al.transformation.utils.warp_image(volume, def_field, interpolation_mode=interpolation_mode,
                                                      padding_mode=padding_mode)
    return warped_image


def sample_from_pca(pca, alphas=None):
    if alphas is None:
        # alphas = np.random.standard_normal(pca.n_components)
        alphas=[None for i in range(pca.n_components)]

    new_sample = np.zeros(pca.n_features_)
    for component, eigenvalue, alpha in zip(pca.components_, pca.singular_values_, alphas):
        if alpha is None:
            w = np.random.normal(loc=0, scale=np.sqrt(eigenvalue), size=None)  # loc is mean, scale is standard deviation, so we sample here from N(0, eigenvalue)
        # new_sample = new_sample + np.sqrt(eigenvalue) * alpha * component
        else:
            w=alpha
        new_sample = new_sample + w * component
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
    if f_mask is not None:
        f_mask.to(dtype, device)

    if using_landmarks:
        fixed_points = reference_scan.landmarks

    number_of_iterations = [150, 100, 80, 50, 20]
    pyramid_sampling_factors = [[32,32,32],[16, 16, 16],[12,12,12],[8, 8, 8],[4,4,4]]  # a full sized image always tops off the pyramid

    fixed_image_pyramid = al.create_image_pyramid(f_image, pyramid_sampling_factors)
    if f_mask is not None:
        fixed_mask_pyramid = al.create_image_pyramid(f_mask, pyramid_sampling_factors)
    else:
        fixed_mask_pyramid = [None for i in range(len(fixed_image_pyramid))]

    fixed_registration_data = RegistrationData(fixed_image_pyramid, fixed_mask_pyramid, fixed_points)

    first_scan = True
    debug_plot(reference_scan.volume, moving_scans[0].volume, '/home/eva/PhD/Data/Test/prior_to_bspline_registration.png')
    for moving_scan in moving_scans:

        m_image = moving_scan.volume
        m_mask = moving_scan.mask

        m_image.to(dtype, device)
        if m_mask is not None:
            m_mask.to(dtype, device)

        # create image pyramid size/8 size/4, size/2, size/1
        moving_image_pyramid = al.create_image_pyramid(m_image, pyramid_sampling_factors)
        if m_mask is not None:
            moving_mask_pyramid = al.create_image_pyramid(m_mask, pyramid_sampling_factors)
        else:
            moving_mask_pyramid = [None for i in range(len(moving_image_pyramid))]

        if using_landmarks:
            moving_points = moving_scan.landmarks
            initial_tre = al.Points.TRE(fixed_points, moving_points)
            print("initial TRE: " + str(initial_tre))

        moving_registration_data = RegistrationData(moving_image_pyramid, moving_mask_pyramid, moving_points)

        bspline_registrator = BsplineRegistrator(number_of_iterations, save_intermediate_displacements=False,
                                                 device=device, dtype=dtype)
        bspline_registrator.using_landmarks = using_landmarks
        fixed_to_moving_displacement, upsampled_displacement = bspline_registrator.perform_fixed_to_moving_registration(
            fixed_registration_data, moving_registration_data)

        warped_image = al.transformation.utils.warp_image(f_image, upsampled_displacement.to(device=device), interpolation_mode='bilinear',
                                                          padding_mode='border')
        # domain measures
        displacement_image = al.create_displacement_image_from_image(fixed_to_moving_displacement,
                                                                     fixed_image_pyramid[-2])
        end = time.time()
        print("Registration done in: ", end - start, " seconds")
        # in order to not invert the displacement field, the fixed points are transformed to match the moving points
        if using_landmarks:
            print("Initial TRE: " + str(initial_tre))
            fixed_points_transformed = al.Points.transform(fixed_points, upsampled_displacement)
            print("Final TRE: " + str(al.Points.TRE(moving_points, fixed_points_transformed)))
        # write result images
        print("writing results")

        if first_scan:
            debug_plot(f_image, warped_image, '/home/eva/PhD/Data/Test/after_bspline_registration.png')
            first_scan=False

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

def perform_experimental_bspline_registration_on_all_scans(reference_scan, moving_scans, save_dir, device=th.device("cpu"),
                                                       dtype=th.float32, using_landmarks=False):
    # kernel registration towards the mean scan
    start = time.time()

    # Do you want to use landmarks?

    fixed_points = None
    moving_points = None

    f_image = reference_scan.volume
    f_mask = reference_scan.mask

    f_image.to(dtype, device)
    if f_mask is not None:
        f_mask.to(dtype, device)

    if using_landmarks:
        fixed_points = reference_scan.landmarks

    number_of_iterations = [150, 100, 80, 50, 20]
    pyramid_sampling_factors = [[32,32,32],[16, 16, 16],[12,12,12],[8, 8, 8],[4,4,4]]  # a full sized image always tops off the pyramid

    fixed_image_pyramid = al.create_image_pyramid(f_image, pyramid_sampling_factors)
    if f_mask is not None:
        fixed_mask_pyramid = al.create_image_pyramid(f_mask, pyramid_sampling_factors)
    else:
        fixed_mask_pyramid = [None for i in range(len(fixed_image_pyramid))]

    fixed_registration_data = RegistrationData(fixed_image_pyramid, fixed_mask_pyramid, fixed_points)

    first_scan = True
    debug_plot(reference_scan.volume, moving_scans[0].volume, '/home/eva/PhD/Data/Test/prior_to_bspline_registration.png')
    for moving_scan in moving_scans:

        m_image = moving_scan.volume
        m_mask = moving_scan.mask

        m_image.to(dtype, device)
        if m_mask is not None:
            m_mask.to(dtype, device)

        # create image pyramid size/8 size/4, size/2, size/1
        moving_image_pyramid = al.create_image_pyramid(m_image, pyramid_sampling_factors)
        if m_mask is not None:
            moving_mask_pyramid = al.create_image_pyramid(m_mask, pyramid_sampling_factors)
        else:
            moving_mask_pyramid = [None for i in range(len(moving_image_pyramid))]

        if using_landmarks:
            moving_points = moving_scan.landmarks
            initial_tre = al.Points.TRE(fixed_points, moving_points)
            print("initial TRE: " + str(initial_tre))

        moving_registration_data = RegistrationData(moving_image_pyramid, moving_mask_pyramid, moving_points)

        bspline_registrator = BsplineRegistrator(number_of_iterations, save_intermediate_displacements=False,
                                                 device=device, dtype=dtype)
        bspline_registrator.using_landmarks = using_landmarks
        fixed_to_moving_displacement, upsampled_displacement = bspline_registrator.perform_fixed_to_moving_registration(
            fixed_registration_data, moving_registration_data)

        warped_image = al.transformation.utils.warp_image(f_image, upsampled_displacement.to(device=device), interpolation_mode='bilinear',
                                                          padding_mode='border')
        # domain measures
        displacement_image = al.create_displacement_image_from_image(fixed_to_moving_displacement,
                                                                     fixed_image_pyramid[-2])
        end = time.time()
        print("Registration done in: ", end - start, " seconds")
        # in order to not invert the displacement field, the fixed points are transformed to match the moving points
        if using_landmarks:
            print("Initial TRE: " + str(initial_tre))
            fixed_points_transformed = al.Points.transform(fixed_points, upsampled_displacement)
            print("Final TRE: " + str(al.Points.TRE(moving_points, fixed_points_transformed)))
        # write result images
        print("writing results")

        if first_scan:
            debug_plot(f_image, warped_image, '/home/eva/PhD/Data/Test/after_bspline_registration.png')
            first_scan=False

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
    f_mask_original_size = reference_scan.mask
    if down_sample_factor is not None or down_sample_factor == [1, 1, 1]:
        f_image = al.utils.image.resize_image(f_image_original_size, down_sample_factor)
        if f_mask_original_size is not None:
            f_mask = al.utils.image.resize_image(f_mask_original_size, down_sample_factor)
            f_mask.to(dtype, device)
        else:
            f_mask = None
    else:
        f_image = f_image_original_size

    f_image.image = 1024 + f_image.image  # to make sure background has value 0. To calculate the center of mass for initial displacement.
    f_image.to(dtype, device)


    fixed_data = RegistrationData(f_image, f_mask)

    debug_plot(f_image_original_size, moving_scans[0].volume, '/home/eva/PhD/Data/Test/prior_to_rigid_registration.png')
    first_scan=True
    for moving_scan in moving_scans:

        m_image_original_size = moving_scan.volume
        m_mask_original_size = moving_scan.mask

        if down_sample_factor is not None or down_sample_factor == [1, 1, 1]:
            m_image = al.utils.image.resize_image(m_image_original_size, down_sample_factor)
            if m_mask_original_size is not None:
                m_mask = al.utils.image.resize_image(m_mask_original_size, down_sample_factor)
            else:
                m_mask = None
        else:
            m_image = m_image_original_size
            m_mask = m_mask_original_size

        m_image.image = 1024 + m_image.image
        m_image.to(dtype, device)
        if m_mask is not None:
            m_mask.to(dtype, device)

        moving_data = RegistrationData(m_image, m_mask)

        # fixed_data = RegistrationData(m_image, m_mask)

        number_of_iterations = 50

        rigid_registrator = SimilarityRegistrator(number_of_iterations, device, dtype, step_size=0.01)

        # transformation = rigid_registrator.get_projective_transformation(m_image, opt_cm=True)
        # # initialize the translation with the center of mass of the fixed image
        # transformation.init_translation(f_image)
        #
        # rigid_registrator.set_projective_transformation(transformation)

        print("perform registration")
        start = time.time()

        displacement = rigid_registrator.perform_registration(fixed_data, moving_data)
        # displacement = rigid_registrator.perform_registration(moving_data, fixed_data)

        print("upsample displacement field")
        upsampled_displacement = al.transformation.utils.upsample_displacement(displacement.clone().to(device='cpu'),
                                                                               m_image_original_size.size,
                                                                               interpolation="linear")

        warped_image = al.transformation.utils.warp_image(m_image_original_size, upsampled_displacement,
                                                          interpolation_mode='bilinear',
                                                          padding_mode='border')
        if m_mask_original_size is not None:
            warped_mask = al.transformation.utils.warp_image(m_mask_original_size, upsampled_displacement,
                                                             interpolation_mode='bilinear',
                                                             padding_mode='border')

        end = time.time()
        print("Registration done in: ", end - start, " seconds")

        # write result images
        print("writing results")

        upsampled_displacement_image = al.create_displacement_image_from_image(upsampled_displacement,
                                                                               f_image_original_size)
        if first_scan:
            debug_plot(f_image_original_size, warped_image, '/home/eva/PhD/Data/Test/after_rigid_registration.png')
            first_scan=False

        scan_save_dir = os.path.join(save_dir, moving_scan.name)
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir, exist_ok=False)
        warped_image.write('{}/rigid_warped_image.nii.gz'.format(scan_save_dir))
        if m_mask_original_size is not None:
            warped_mask.write('{}/rigid_warped_mask.nii.gz'.format(scan_save_dir))
        upsampled_displacement_image.write('{}/rigid_displacement_image_unit.nii.gz'.format(scan_save_dir))
    print("=================================================================")
    print("Rigid registration done")


def debug_plot(fixed_image, moving_image, filename):
    grid = T.utils.compute_grid(moving_image.size, dtype=moving_image.dtype,
                                device=moving_image.device)
    # compute displacement field
    displacement = grid
    # warp moving image with dispalcement field
    warped_moving_image = F.grid_sample(moving_image.image, displacement)
    # compute squared differences
    plot_images = [fixed_image.image, warped_moving_image]
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
    image_keys = [0, 0, 0, 1, 1, 1]
    counter = 0
    for ax, image_key in zip(axs.flat, image_keys):
        if counter % 3 == 0:
            ax.imshow(plot_images[image_key].cpu()[0, 0, 140, :, :])
            ax.set_title(str(image_key))
        elif counter % 3 == 1:
            ax.imshow(plot_images[image_key].cpu()[0, 0, :, 150, :])
            ax.set_title(str(image_key))
        elif counter % 3 == 2:
            ax.imshow(plot_images[image_key].cpu()[0, 0, :, :, 70])
            ax.set_title(str(image_key))
        counter = counter + 1
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # main(body_part_choice='upper', reference_scan_name='001_upper')
    # main(body_part_choice='upper', reference_scan_name='002_upper')
    # main(body_part_choice='upper', reference_scan_name='003_upper')
    # main(body_part_choice='lower', reference_scan_name='001_lower')
    # main(body_part_choice='lower', reference_scan_name='002_lower')
    # main(body_part_choice='lower', reference_scan_name='003_lower')


    include_pca_scans = False
    include_verse_scans = True
    include_flipped_scans = True
    body_part_choice = 'upper'
    scans_to_csv(body_part_choice, include_pca_scans, include_verse_scans, include_flipped_scans)
