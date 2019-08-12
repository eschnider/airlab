import os
from abc import abstractmethod
import torch as th
import airlab as al


class Registrator:
    using_landmarks = False
    device = None
    dtype = None

    def __init__(self, device=th.device("cpu"), dtype=th.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def perform_registration(self, fixed_reg_data, moving_reg_data):
        """

        :type fixed_reg_data: examples.CustomData.RegistrationData
        :type moving_reg_data: examples.CustomData.RegistrationData
        """
        ...


class ProjectiveRegistrator(Registrator):

    def __init__(self, device=th.device("cpu"), dtype=th.float32):
        super().__init__(device, dtype)

    @property
    @abstractmethod
    def projective_transformation(self):
        ...

    def perform_registration(self, fixed_reg_data, moving_reg_data):
        # create pairwise registration object
        registration = al.PairwiseRegistration()
        # choose the affine transformation model
        transformation = self.projective_transformation(fixed_reg_data.image, opt_cm=True)
        transformation.init_translation(fixed_reg_data.image)
        registration.set_transformation(transformation)
        # choose the crazy multilabel Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE_multilabel(fixed_reg_data.image, moving_reg_data.image,
                                                     fixed_mask=fixed_reg_data.mask,
                                                     moving_mask=moving_reg_data.mask)
        registration.set_image_loss([image_loss])
        # choose the Adam optimizer to minimize the objective
        optimizer = th.optim.Adam(transformation.parameters(), lr=self.step_size)
        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(self.number_of_iterations)
        # start the registration
        registration.start()

        # return the created displacement field
        displacement = transformation.get_displacement()  # this is a unit displacement
        return displacement


class AffineRegistrator(ProjectiveRegistrator):
    number_of_iterations = None
    step_size = None

    def __init__(self, number_of_iterations, device=th.device("cpu"), dtype=th.float32, step_size=0.01):
        super().__init__(device, dtype)
        self._projective_transformation = None
        self.using_landmarks = False  # landmarks in similarity reg not supported yet
        self.number_of_iterations = number_of_iterations
        self.step_size = step_size

    @property
    def projective_transformation(self):
        self._projective_transformation = al.transformation.pairwise.AffineTransformation
        return self._projective_transformation


class SimilarityRegistrator(ProjectiveRegistrator):
    number_of_iterations = None
    step_size = None

    def __init__(self, number_of_iterations, device=th.device("cpu"), dtype=th.float32, step_size=0.01):
        super().__init__(device, dtype)
        self._projective_transformation = None
        self.using_landmarks = False  # landmarks in similarity reg not supported yet
        self.number_of_iterations = number_of_iterations
        self.step_size = step_size

    @property
    def projective_transformation(self):
        self._projective_transformation = al.transformation.pairwise.SimilarityTransformation
        return self._projective_transformation


class RigidRegistrator(ProjectiveRegistrator):
    number_of_iterations = None
    step_size = None

    def __init__(self, number_of_iterations, device=th.device("cpu"), dtype=th.float32, step_size=0.01):
        super().__init__(device, dtype)
        self._projective_transformation = None
        self.using_landmarks = False  # landmarks in similarity reg not supported yet
        self.number_of_iterations = number_of_iterations
        self.step_size = step_size

    @property
    def projective_transformation(self):
        self._projective_transformation = al.transformation.pairwise.RigidTransformation
        return self._projective_transformation


class BsplineRegistrator(Registrator):
    regularisation_weight = [1e-2, 1e-1, 1e-0, 1e+2]
    number_of_iterations = [20, 10, 0, 0]
    sigma = [[9, 9, 9], [9, 9, 9], [9, 9, 9], [9, 9, 9]]
    step_size = [3e-3, 4e-3, 2e-3, 2e-3]
    bspline_order = 3
    temp_displacement_save_path = None
    save_intermediate_displacements = False

    def __init__(self, number_of_iterations, save_intermediate_displacements=False, device=th.device("cpu"), dtype=th.float32):
        super().__init__(device, dtype)
        self.number_of_iterations = number_of_iterations
        self.save_intermediate_displacements = save_intermediate_displacements

    def perform_fixed_to_moving_registration(self, fixed_reg_data, moving_reg_data):
        return self.perform_registration(moving_reg_data, fixed_reg_data)

    def perform_registration(self, fixed_reg_data, moving_reg_data):
        moving_image_pyramid = moving_reg_data.image
        moving_mask_pyramid = moving_reg_data.mask
        fixed_image_pyramid = fixed_reg_data.image
        fixed_mask_pyramid = fixed_reg_data.mask

        moving_points = moving_reg_data.landmarks
        fixed_points = fixed_reg_data.landmarks

        constant_flow = None

        print("perform registration")
        for level, (mov_im_level, mov_msk_level, fix_im_level, fix_msk_level) in enumerate(zip(moving_image_pyramid,
                                                                                               moving_mask_pyramid,
                                                                                               fixed_image_pyramid,
                                                                                               fixed_mask_pyramid)):
            if level < len(moving_image_pyramid) - 1:  # no registration at the full level! Doesn't fit into GPU :(
                print("---- Level " + str(level) + " ----")
                registration = al.PairwiseRegistration()

                # define the transformation
                transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                                  sigma=self.sigma[level],
                                                                                  order=self.bspline_order,
                                                                                  dtype=self.dtype,
                                                                                  device=self.device)

                if level > 0:
                    constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                                  mov_im_level.size,
                                                                                  interpolation="linear")

                    transformation.set_constant_flow(constant_flow)

                registration.set_transformation(transformation)

                # choose the Mean Squared Error as image loss
                image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level, mov_msk_level, fix_msk_level)

                registration.set_image_loss([image_loss])

                # define the regulariser for the displacement
                regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
                regulariser.SetWeight(self.regularisation_weight[level])

                registration.set_regulariser_displacement([regulariser])

                # define the optimizer
                optimizer = th.optim.Adam(transformation.parameters(), lr=self.step_size[level], amsgrad=True)

                registration.set_optimizer(optimizer)
                registration.set_number_of_iterations(self.number_of_iterations[level])

                registration.start()

                # store current flow field
                constant_flow = transformation.get_flow()

                if self.using_landmarks or self.save_intermediate_displacements:
                    current_displacement = transformation.get_displacement()
                    # generate SimpleITK displacement field and calculate TRE
                    tmp_displacement = al.transformation.utils.upsample_displacement(
                        current_displacement.clone().to(device='cpu'),
                        moving_image_pyramid[-1].size, interpolation="linear")
                    tmp_displacement = al.transformation.utils.unit_displacement_to_dispalcement(
                        tmp_displacement)  # unit measures to image domain measures
                    tmp_displacement = al.create_displacement_image_from_image(tmp_displacement,
                                                                               moving_image_pyramid[-1])

                    if self.save_intermediate_displacements:
                        tmp_displacement_path = os.path.join(self.temp_displacement_save_path,
                                                             'tmp/bspline_displacement_image_level_' + str(level) + '.vtk')
                        tmp_displacement.write(tmp_displacement_path)

                    # in order to not invert the displacement field, the fixed points are transformed to match the moving points
                    if self.using_landmarks:
                        print("TRE on that level: " + str(
                            al.Points.TRE(moving_points, al.Points.transform(fixed_points, tmp_displacement))))

        # create final result
        displacement = transformation.get_displacement()  # this is a unit displacement
        upsampled_displacement = al.transformation.utils.upsample_displacement(
            displacement.clone().to(device='cpu'),
            moving_image_pyramid[-1].size, interpolation="linear")
        return displacement, upsampled_displacement