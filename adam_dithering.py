# works but does not improve on the floyd & steinberg algorithm so far
#   * explore different similarity measures and the values in optimized image: ssim, corr, etc.
#   * explore different models of human early visual processing: alexnet, etc.
#   * explore different optimizers or use different parameters


import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


############################
## some functions
############################

def sigmoid(x, k):
    return 1 / (1 + torch.exp(-k * (x - 0.5)))


def spatial_contrast_im(im_values, deg, Lmin, Lmax):
    # Implementation of the "Standard B" model from Watson & Ahumada (2005).
    # Assumes that the computer monitor has been calibrated to allow linear
    # manipulation of luminance. Outputs the image filtered with the CSF and 
    # OEF.
    # 
    # im    = a square uint8 image
    # deg   = the width of im on the computer monitor in deg of visual angle
    # Lmin  = minimum luminance displayable on the computer monitor
    # Lmax  = minimum luminance displayable on the computer monitor
    # 
    # Frederic Gosselin, 17/12/2023
    # 
    # Reference:
    # Watson, A. B., & Ahumada, A. J., Jr. (2005). A standard model for 
    # foveal detection of spatial contrast. Journal of Vision, 5(9):6, 
    # 717-740, http://journalofvision.org/5/9/6/, doi:10.1167/5.9.6.
    
    im_values = im_values.float()
    Lrange = (Lmax - Lmin)
    Lmean = (Lmax + Lmin) / 2
    im_L = im_values / 255 * Lrange + Lmin
    im = (im_L - Lmean) / Lmean

    patchSize = im.shape[0]
    halfPatchSize = patchSize / 2
    cpd_max = halfPatchSize / (deg / 2)

    x, y = torch.meshgrid(torch.arange(-halfPatchSize, halfPatchSize), torch.arange(-halfPatchSize, halfPatchSize))
    x = x / (patchSize - 1) * 2 * cpd_max
    y = y / (patchSize - 1) * 2 * cpd_max
    f = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x) + np.pi

    x, y = torch.meshgrid(torch.arange(-halfPatchSize, halfPatchSize), torch.arange(-halfPatchSize, halfPatchSize))
    x = x / (patchSize - 1) * deg
    y = y / (patchSize - 1) * deg
    #r = torch.sqrt(x**2 + y**2)

    f0 = 4.3469
    f1 = 1.4476
    a = 0.8514
    p = 0.7929
    #delta = 0.3652
    gamma = 3.48
    lambda_ = 13.57

    S_HPmH = 1/torch.cosh((f / f0)**p) - a * 1/torch.cosh(f / f1)
    OEF = 1 - (1 - torch.exp(-(f - gamma) / lambda_)) * torch.sin(2 * theta)**2
    OEF[f <= gamma] = 1
    CSOEF = S_HPmH * OEF

    fim = torch.fft.fftshift(torch.fft.fft2(im))
    filt_im = torch.real(torch.fft.ifft2(torch.fft.ifftshift(CSOEF * fim)))

    # A = torch.exp(-r**2 / (2 * delta**2))
    # ap_filt_im = A * filt_im

    return filt_im


def binary_to_real(im_binary):
    
    bits = im_binary.shape[2]
    im_real = torch.zeros(im_binary.shape[0], im_binary.shape[1])
    
    for the_bit in torch.arange(bits):
        im_real += im_binary[:,:,the_bit]*2**the_bit

    return im_real


def real_to_binary(im_real):

    # im_real must contain an even number of values between 0 and 255
    im_integer = torch.round(im_real*255)
    im_binary = torch.zeros(im_integer.shape[0], im_integer.shape[1], 1)
    remainder = torch.zeros(im_integer.shape[0], im_integer.shape[1], 1)

    while torch.sum(im_integer) > 0:
        remainder[:,:,0] = im_integer % 2
        im_binary = torch.cat((im_binary, remainder), dim=2)
        im_integer = im_integer // 2

    # removes the first value of dimension 2 that contains only zeros
    return im_binary[:,:,1:]



def sigmoid_spatial_contrast_im(im_values, deg, Lmin, Lmax, k_value):
    temp = spatial_contrast_im(binary_to_real(sigmoid(im_values, k_value)), deg, Lmin, Lmax)
    temp = (temp-torch.min(temp))/(torch.max(temp)-torch.min(temp)) # stretch between 0 and 1
    return temp


def floyd_steinberg(im, depth):
    # Implements the dithering algorithm presented in:
    # R.W. Floyd, L. Steinberg, An adaptive algorithm for spatial grey scale. 
    # Proceedings of the Society of Information Display 17, 75077 (1976).

    # im is an image matrix in double that varies between 0 and 1, and depth is 
    # the number of evenly separated luminance values at your disposal. tim is an 
    # matrix containg integer values between 0 and depth-1, indicating which 
    # luminance value should be used for every pixel. 

    # E.g.:
    # im = (rand(256)-.5)*.1+.5;
    # tim = floyd_steinberg(im, 256);
    # figure, imshow(uint8(tim))

    # This example assumes that all rgb values are linearly related to luminance 
    # values (e.g. on a Mac, put your LCD monitor gamma parameter to 1 in the Displays 
    # section of the System Preferences). If this is not the case, use a lookup table 
    # to transform the tim integer values into rgb values corresponding to evenly 
    # spaced luminance values.

    # Frederic Gosselin, 16/12/2023
    # frederic.gosselin@umontreal.ca
    
    tim = im * (depth - 1)
    
    for yy in range(1, im.shape[0] - 1):
        for xx in range(1, im.shape[1] - 1):
            oldpixel = tim[yy, xx]
            newpixel = round(tim[yy, xx])
            im[yy, xx] = newpixel
            quant_error = oldpixel - newpixel
            tim[yy, xx + 1] = tim[yy, xx + 1] + 7/16 * quant_error
            tim[yy + 1, xx - 1] = tim[yy + 1, xx - 1] + 3/16 * quant_error
            tim[yy + 1, xx] = tim[yy + 1, xx] + 5/16 * quant_error
            tim[yy + 1, xx + 1] = tim[yy + 1, xx + 1] + 1/16 * quant_error
            
    tim = np.round(tim)
    
    return tim


def noisy_bit_dithering(im, depth = 256):
    # Implements the dithering algorithm presented in:
    # Allard, R., Faubert, J. (2008) The noisy-bit method for digital displays:
    # converting a 256 luminance resolution into a continuous resolution. Behavior 
    # Research Method, 40(3), 735-743.
    # It takes 2 arguments:
    #   im: is an image matrix in float64 that varies between 0 and 1, 
    #   depth: is the number of evenly separated luminance values at your disposal. 
    #     Default is 256 (1 byte).
    # It returns:
    #   tim: a matrix containg integer values between 0 and depth-1, indicating which 
    #     luminance value should be used for every pixel. 
    #
    # E.g.:
    #   tim = noisy_bit_dithering(im, depth = 256)
    #
    # This example assumes that all rgb values are linearly related to luminance 
    # values (e.g. on a Mac, put your LCD monitor gamma parameter to 1 in the Displays 
    # section of the System Preferences). If this is not the case, use a lookup table 
    # to transform the tim integer values into rgb values corresponding to evenly 
    # spaced luminance values.
    #
    # Frederic Gosselin, 27/09/2022
    # frederic.gosselin@umontreal.ca
    tim = im * (depth - 1.0)
    tim = np.uint8(np.fmax(np.fmin(np.around(tim + np.random.random(np.shape(im)) - 0.5), depth - 1.0), 0.0))
    return tim









def adam_dithering(original_image_tensor, n_bits):

    # initial_optimized_image

    # could use floyd steinberg or allard faubert with real_to_binary treatment
    # if random initialization:
    #optimized_image_tensor = torch.rand(original_image_tensor.shape[0], original_image_tensor.shape[1], n_bits).requires_grad_(True)
    # elseif allard-faubert initialization:
    optimized_image_np = noisy_bit_dithering(original_image_tensor.detach().numpy(), depth=2**n_bits)/255
    optimized_image_tensor_real = torch.from_numpy(optimized_image_np).float()
    optimized_image_tensor = real_to_binary(optimized_image_tensor_real).requires_grad_(True)
    # elseif floyd-steinberg initilization:

    # Apply spatial contrast standard model (Ahumada & Watson, 2005) to the original image
    deg = 3
    Lmin = 0
    Lmax = 100
    k_slope = 3
    filtered_original_image_tensor = spatial_contrast_im(original_image_tensor, deg, Lmin, Lmax)
    filtered_original_image_tensor = (filtered_original_image_tensor - torch.min(filtered_original_image_tensor)) / (torch.max(filtered_original_image_tensor) - torch.min(filtered_original_image_tensor)) # stretch between 0 and 1

    # plt.imshow(filtered_original_image_tensor.detach().numpy(), cmap='gray')
    # plt.show()

    # Set up the Adam optimizer
    optimizer = optim.Adam([optimized_image_tensor], lr=0.001)

    # Loss function to measure dissimilarity
    criterion = nn.MSELoss()

    # Training loop for optimization
    max_epochs = 1000
    tol = 0
    loss_history = []

    for epoch in range(max_epochs):
        # Forward pass through the model
        a_filtered_optimized_image_tensor = sigmoid_spatial_contrast_im(optimized_image_tensor, deg, Lmin, Lmax, k_slope)

        # Calculate the loss
        loss = criterion(a_filtered_optimized_image_tensor, filtered_original_image_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_history.append(loss.item())

        print(f'Epoch [{epoch + 1}/{max_epochs}], Loss: {loss.item():.4f}')
        # stop when loss is sufficiently small
        if loss.item()<tol:
            break

    adam_dithered_image_tensor = binary_to_real(sigmoid(optimized_image_tensor, 1000))

    return adam_dithered_image_tensor, optimized_image_tensor, loss_history




# adam dithering
# Initialize
deg = 3
Lmin = 0
Lmax = 100
k_slope = 3
   
xy_size = 256
n_bits = 2 # number of bits of grayscale values


# Load the first image
original_image_pil = Image.open('/Users/fredericgosselin/Desktop/adam dithering/elephant.jpg').convert('L')
original_image_pil = original_image_pil.resize((xy_size, xy_size))
original_image_np = np.asarray(original_image_pil) / 255 # lit une image couleur et la transforme en niveaux de gris; 
                                                                                    # les images sont des int entre 0 et 255; avec Image.open de PIL
original_image_tensor = torch.from_numpy(original_image_np).float()


plt.imshow(original_image_np, cmap='gray')
plt.show()


adam_dithered_image_tensor, optimized_image_tensor, loss_history = adam_dithering(original_image_tensor, n_bits)

# plt.imshow(a_filtered_optimized_image_tensor.detach().numpy(), cmap='gray')
# plt.show()


filtered_adam_dithered_image_tensor = spatial_contrast_im(adam_dithered_image_tensor, deg, Lmin, Lmax)

# plt.imshow(optimized_image_tensor.detach().numpy(), cmap='gray')
# plt.show()

plt.imshow(adam_dithered_image_tensor.detach().numpy(), cmap='gray')
plt.show()

plt.imshow(filtered_adam_dithered_image_tensor.detach().numpy(), cmap='gray')
plt.show()

# plt.imshow(filtered_original_image_tensor.detach().numpy(), cmap='gray')
# plt.show()

plt.plot(loss_history)
plt.show()

