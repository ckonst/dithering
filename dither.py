from PIL import Image
import numpy as np

#%%
# miku will help us out here
image = Image.open('hatsune_miku.png')
input_data = np.array(image)
cols, rows = image.size

#%%
color_reduced_image = image.convert('P', palette=Image.WEB, dither=Image.NONE).convert('RGB')
reduced_color_data = np.array(color_reduced_image)
color_reduced_image.save('color_reduced.png')

#%%
error_data = np.subtract(input_data, reduced_color_data, dtype=np.float32)
#%%
# error diffusion indices
u = np.array([0,  1, 1, 1])
v = np.array([1, -1, 0, 1])
w = np.array([0,  1, 2, 3])
#%%
# error diffusion ratios
d = np.array([[[[7, 3, 5, 1]]]]) / 16
distribution = d * np.reshape([error_data, error_data,
                                     error_data, error_data],
                                   (*error_data.shape, 4))

#%%
new_image = reduced_color_data.astype(np.float32)
for x in range(0, cols - 1):
    for y in range(0, rows - 1):
        new_image[y + u, x + v] += distribution[y, x, :, w]

#%%
ouput_data = np.clip(new_image, 0, 255).astype(np.uint8)
dithered_image = Image.fromarray(ouput_data)
dithered_image.save('dithered.png')

#%%
dithering = image.convert('P', palette=Image.WEB)

#%%
def floyd_steinberg(image):
    image = image.copy()
    distribution = np.array([7, 3, 5, 1], dtype=float) / 16
    u = np.array([0, 1, 1, 1])
    v = np.array([1, -1, 0, 1])

    for y in range(image.shape[0] - 1):
        for x in range(image.shape[1] - 1):
            value = np.round(image[y, x] / 255.)
            error = image[y, x] - value
            image[y, x] = value
            image[y + u, x + v] += np.round(error * distribution).astype(np.uint8)

    image[:, -1] = 1
    image[-1, :] = 1
    return image

im = Image.open('smile.bmp')
indata = np.array(im)[:, :, 0]
x = floyd_steinberg(indata) - 1
#%%
ximg = Image.fromarray(x)
ximg.save('test.png')