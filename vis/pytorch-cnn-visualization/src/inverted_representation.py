"""
Created on Wed Jan 17 08:05:11 2018

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import cv2
import torch
from torch.autograd import Variable
from torch.optim import SGD
import os
import os.path as osp

from misc_functions import get_params, recreate_image


class InvertedRepresentation():
    def __init__(self, model, path):
        self.model = model
        self.model.eval()
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta / 2)).sum()
        return total_variation

    def euclidian_loss(self, org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    def get_output_from_specific_layer(self, x, layer_id, index_):
        """
            Saves the output after a forward pass until nth layer
            This operation could be done with a forward hook too
            but this one is simpler (I think)
        """
        layer_output = None
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x[0, index_, :, :]

        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if str(index) == str(layer_id):
                layer_output = x[0]
                break
        return layer_output[index_, :, :]

    def generate_inverted_image_specific_layer(self, input_image, index, img_size, target_layer=3, steps=50):
        print('index =', index)
        os.makedirs(osp.join(self.path, str(index)), exist_ok=True)
        # Generate a random image which we will optimize
        opt_img = Variable(1e-1 * torch.randn(1, 3, *img_size), requires_grad=True)
        # Define optimizer for previously created image
        optimizer = SGD([opt_img], lr=1e3, momentum=0.9)
        # Get the output from the model after a forward pass until target_layer
        # with the input image (real image, NOT the randomly generated one)
        input_image_layer_output = \
            self.get_output_from_specific_layer(input_image, target_layer, index)

        # Alpha regularization parametrs
        # Parameter alpha, which is actually sixth norm
        alpha_reg_alpha = 6
        # The multiplier, lambda alpha
        alpha_reg_lambda = 1e-7

        # Total variation regularization parameters
        # Parameter beta, which is actually second norm
        tv_reg_beta = 2
        # The multiplier, lambda beta
        tv_reg_lambda = 1e-8

        for i in range(steps + 1):
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)
            output = self.get_output_from_specific_layer(opt_img, target_layer, index)
            # Calculate euclidian loss
            euc_loss = 1e-1 * self.euclidian_loss(input_image_layer_output.detach(), output)
            # Calculate alpha regularization
            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
            # Calculate total variation regularization
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)
            # Sum all to optimize
            loss = euc_loss + reg_alpha + reg_total_variation
            # Step
            loss.backward()
            optimizer.step()
            # Generate image every 5 iterations
            if i % 10 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.numpy())
                x = recreate_image(opt_img)
                cv2.imwrite(
                    osp.join(self.path, str(index), 'Inv_Image_Layer_' + str(target_layer) +
                             '_Iteration_' + str(i) + '.jpg'),
                    x
                )
                # cv2.imwrite('../generated/Inv_Image_Layer_' + str(target_layer) +
                #             '_Iteration_' + str(i) + '.jpg', x)
            # Reduce learning rate every 40 iterations
            if i % 40 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1 / 10


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)

    inverted_representation = InvertedRepresentation(pretrained_model)
    image_size = 224  # width & height
    target_layer = 12
    inverted_representation.generate_inverted_image_specific_layer(prep_img,
                                                                   image_size,
                                                                   target_layer)
