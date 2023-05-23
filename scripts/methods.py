from scripts.information_bottleneck.IBA import *
from scripts.information_bottleneck.fitting_estimators import *
from scripts.utils import to_saliency_map
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
import clip
import copy
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from functools import partial
from matplotlib.text import Text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     
# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True) 
        # +1 because the first output is embedding 
        feature = states['hidden_states'][layer_idx+1]
        return feature

# Extract BERT Layer
def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'layers' or n == 'resblocks':
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        desired_layer = s2
                        return desired_layer

def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var*np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator

def get_fitting_estimator(name):
    if name == 'cos':
        return cos_estim
    elif name == 'pearson':
        return pearson_estim
    elif name == 'mgauss':
        return gauss_estim
    else:
        return gauss_estim #default
    
def get_heatmap(image_relevance):
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).to(device).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance

# Text Layer Heatmap
def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, fitting_estimator, lr=1, train_steps=10):
    features = extract_feature_map(model.text_model, layer_idx, text_t)
    layer = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    fitting_estimator = get_fitting_estimator(fitting_estimator)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, fitting_estimator=fitting_estimator, progbar=False)
    heatmap, loss_c, loss_f, loss_t = reader.text_heatmap(text_t, image_t)
    return heatmap

# Vision Layer Heatmap
def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, fitting_estimator, lr=1, train_steps=10):
    features = extract_feature_map(model.vision_model, layer_idx, image_t)
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    fitting_estimator = get_fitting_estimator(fitting_estimator)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, fitting_estimator=fitting_estimator, progbar=False)
    heatmap, loss_c, loss_f, loss_t = reader.vision_heatmap(text_t, image_t)
    heatmap = get_heatmap(torch.nansum(heatmap, -1)[1:])
    #heatmap = to_saliency_map(heatmap, image_t[0].permute(1,2,0).shape)
    #norm = mpl.colors.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    #heatmap = norm(heatmap)
    return heatmap


class TextWithBGColor(Text):
    def __init__(self, x, y, text, bgcolor, *args, **kwargs):
        super().__init__(x, y, text, *args, **kwargs)
        self.bgcolor = bgcolor

    def draw(self, renderer, *args, **kwargs):
        bbox = dict(facecolor=self.bgcolor, edgecolor=self.bgcolor, boxstyle='round,pad=0.01', alpha=self.bgcolor[3])
        self.set_bbox(bbox)
        super().draw(renderer, *args, **kwargs)


def plot_text_with_colors(ax, tokens, rgba_colors, max_width, max_height, fontsize=12):
    x, y = 0, max_height
    space_width = fontsize * 0.2  # Width of a single space
    for i, (token, color) in enumerate(zip(tokens, rgba_colors)):
        text = TextWithBGColor(x, y*0.3, token, color, fontsize=fontsize)
        ax.add_artist(text)
        
        token_width = fontsize * 0.4 * len(token)

        if i + 1 < len(tokens):
            next_token_width = fontsize * 0.4 * len(tokens[i + 1]) + space_width
            if x + token_width + next_token_width >= max_width:
                x = 0
                y -= 1  # Decrease the space between rows
            else:
                x += token_width + space_width  # Add the width of a single space after each token
        else:
            x += token_width

def generate_red_shades_with_alpha(scores, min_alpha=0, max_alpha=0.8):
    normalized_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
    red_colors = []
    for score in normalized_scores:
        alpha = min_alpha + score * (max_alpha - min_alpha)
        red_color = (0, 0, 0.8, alpha)
        red_colors.append(red_color)
    return red_colors

def visualize_vandt_heatmap(tmap, vmap, text_words, image, title=None, bb=None, vtitle=None, ttitle=None, max_width=200, max_height=2):
    text_words = [x.split('<')[0] for x in text_words[1:-1]]
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(5,9) 
    #axs[0].imshow(np.float32(image)*np.expand_dims(vmap, axis=2))
    axs[0].imshow(show_cam_on_image(np.float32(image), vmap, use_rgb=True))
    if bb:
        for x, y, w, h in bb:
            rect = mpl.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axs[0].add_patch(rect)
    axs[0].axis('off')
    rgba_colors = generate_red_shades_with_alpha(tmap[1:-1])
    plot_text_with_colors(axs[1], text_words, rgba_colors, max_width=max_width, max_height=max_height, fontsize=18)
    axs[1].set_xlim(-2, max_width)
    axs[1].set_ylim(-max_height, max_height)
    axs[1].axis('off')
    plt.subplots_adjust(wspace=0, hspace=-1, bottom = 0)
    plt.tight_layout()
    if title:
        plt.savefig(title,bbox_inches='tight')