import os
import csv
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mySequential(nn.Sequential):
    def forward(self, *input, **kwargs):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input

def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    print(3, replacement)
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

def visualize_text_heatmap(hm, start_idx, end_idx, text_words):
    # hm should be a 2d array
    hm = np.array(hm)
    trim_hm = hm[:, start_idx:end_idx]
    trim_len = end_idx - start_idx
    _, ax = plt.subplots()
    x = ax.imshow(trim_hm, cmap="Blues")
    ax.set_xticks(range(trim_len))
    ax.set_xticklabels(text_words[start_idx:end_idx])
    ax.set_yticks(range(len(hm)))
    ax.set_yticklabels(['layer{}'.format(i) for i in range(len(hm))])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
    plt.colorbar(x, label="Relative Importance", orientation="vertical")
    plt.tight_layout()
    plt.savefig("text.png")

def visualize_image_heatmap(capacity, img=None, ax=None,
                      colorbar_label='Bits / Pixel',
                      colorbar_fontsize=14,
                      min_alpha=0.2, max_alpha=0.7, vmax=None,
                      colorbar_size=0.3, colorbar_pad=0.08, color_bar=True):

    """
    Plots the heatmap with an bits/pixel colorbar and optionally overlays the image.
    Args:
        saliency_map (np.ndarray): the saliency_map.
        img (np.ndarray):  show this image under the saliency_map.
        ax: matplotlib axis. If ``None``, a new plot is created.
        colorbar_label (str): label for the colorbar.
        colorbar_fontsize (int): fontsize of the colorbar label.
        min_alpha (float): minimum alpha value for the overlay. only used if ``img`` is given.
        max_alpha (float): maximum alpha value for the overlay. only used if ``img`` is given.
        vmax: maximum value for colorbar.
        colorbar_size: width of the colorbar. default: Fixed(0.3).
        colorbar_pad: width of the colorbar. default: Fixed(0.08).
    Returns:
        The matplotlib axis ``ax``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.axes_size import Fixed
    from skimage.color import rgb2gray, gray2rgb

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    
    if img is not None:
        # Underlay the image as grayscale
        gray = gray2rgb(rgb2gray(img))
        ax.imshow(gray)
        saliency_map = to_saliency_map(capacity, img.shape)
    else:
        saliency_map = to_saliency_map(capacity)
    
    

    if vmax is None:
        vmax = saliency_map.max()
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    n = 256
    half_jet_rgba = plt.cm.seismic(np.linspace(0.5, 1, n))
    half_jet_rgba[:, -1] = np.linspace(0.2, 1, n)
    cmap = mpl.colors.ListedColormap(half_jet_rgba)
    hmap_jet = cmap(norm(saliency_map))
    if img is not None:
        hmap_jet[:, :, -1] = (max_alpha - min_alpha)*norm(saliency_map) + min_alpha
    ax.imshow(hmap_jet, alpha=max_alpha)
    if color_bar:
        ax1_divider = make_axes_locatable(ax)
        if type(colorbar_size) == float:
            colorbar_size = Fixed(colorbar_size)
        if type(colorbar_pad) == float:
            colorbar_pad = Fixed(colorbar_pad)
        cax1 = ax1_divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        cbar = mpl.colorbar.ColorbarBase(cax1, cmap=cmap, norm=norm)
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')
    ax.set_frame_on(False)
    return ax

def to_saliency_map(capacity, shape=None, data_format='channels_last'):
    from skimage.transform import resize
    """
    Converts the layer capacity (in nats) to a saliency map (in bits) of the given shape.
    PyTorch:    Use data_format == 'channels_first'
    Tensorflow: Use data_format == 'channels_last'
    """
    # dim: (50, 768) where 768 is the channel size
    if capacity.shape[0] == capacity.shape[1]:
        return capacity
    capacity = capacity[1:].reshape(7,7,-1) # first token is cls
    if data_format == 'channels_first':
        saliency_map = np.nansum(capacity, 0)
    elif data_format == 'channels_last':
        saliency_map = np.nansum(capacity, -1)
    else:
        raise ValueError

    # to bits
    #saliency_map /= float(np.log(2))

    if shape is not None:
        ho, wo = saliency_map.shape
        h, w, c = shape
        # Scale bits to the pixels
        saliency_map *= (ho*wo) / (h*w)
        return resize(saliency_map, (h,w), order=1, preserve_range=True)
    else:
        return saliency_map

class CosSimilarity:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)
    
class ImageFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ImageFeatureExtractor, self).__init__()
        self.model = model
                
    def __call__(self, x):
        return self.model.get_image_features(x)

class TextFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(TextFeatureExtractor, self).__init__()   
        self.model = model
                
    def __call__(self, x):
        return self.model.get_text_features(x)
    
# Transformation for CAM
def image_transform(t, height=7, width=7):
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, 1 :  , :].reshape(t.size(0), height, width, t.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Transformation for CAM
def text_transform(t):
    if t.size(1) == 1: t = t.permute(1,0,2)
    result = t[:, :  , :].reshape(t.size(0), 1, -1, t.size(2))
    return result
