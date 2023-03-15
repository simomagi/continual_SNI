import numpy as np 
import jpegio as jio 
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt 
import pywt
from scipy.ndimage import filters
from multiprocessing import cpu_count

def flatten_dct(all_dcts, i):
    dct_matrix = all_dcts[i,:, :]
    rows=dct_matrix.shape[0]
    columns=dct_matrix.shape[1]
    
    dct_coefficent =[[] for i in range(rows+columns-1)]
    
    for i in range(rows):
        for j in range(columns):
            sum=i+j
            if(sum%2 ==0):
                #add at beginning
                dct_coefficent[sum].insert(0,dct_matrix[i][j])
            else:
                #add at end of the list
                dct_coefficent[sum].append(dct_matrix[i][j])
            
    flatten_dct = []     
    # print the solution as it as
    for i in dct_coefficent:
        for j in i:
            flatten_dct.append(j)

    return np.array(flatten_dct)
 
 
def filter_and_compute_hist(zig_zag_coeff):
    filtered_coeff =  zig_zag_coeff[(zig_zag_coeff <= +50) & (zig_zag_coeff >= -50)]
    hist = np.histogram(filtered_coeff, bins=[i for i in range(-50, 52, 1)])[0]
    return hist 

def get_histogram_dct(img_path, patch_size):
    # cut image 
    jpeg_image =  jio.read(img_path)
    coef = jpeg_image.coef_arrays[0]
    
    h, w    = coef.shape[0], coef.shape[1]
    # remove not divisible part of 256x256 
    coef = coef[:int(patch_size * np.floor(h/patch_size)), :int(patch_size * np.floor(w/patch_size))]
    new_h = coef.shape[0]
    new_w = coef.shape[1]
    tiled_coef  = coef.reshape(int(new_h // patch_size), patch_size,  int(new_w // patch_size),  patch_size)
    tiled_coef= tiled_coef.swapaxes(1, 2)
    tiled_coef = tiled_coef.reshape(-1, patch_size, patch_size)
    
    patch_ids = []
    hist_list = []
    for patch_id, patch_coef in enumerate(tiled_coef):
    
        nr_blk = patch_coef.shape[0] // 8   
        nc_blk = patch_coef.shape[1] // 8
        patch_coef_blk = patch_coef.reshape(nr_blk, 8, nc_blk, 8)
        patch_coef_blk = patch_coef_blk.transpose(0, 2, 1, 3)
        
        patch_coef_blk = patch_coef_blk.reshape(-1, 8, 8)
        n_dcts = patch_coef_blk.shape[0]
        zig_zag_coeff = Parallel(n_jobs=cpu_count())(delayed(flatten_dct)(patch_coef_blk, i) for i in range(n_dcts))
        zig_zag_coeff  = np.array(zig_zag_coeff)
        # remove DC
        zig_zag_coeff = zig_zag_coeff[:,1:10]
   
        hist = np.apply_along_axis(lambda a: filter_and_compute_hist(a), 0, zig_zag_coeff)
        
 
        
        patch_ids.append(patch_id)
        hist_list.append(hist)
        
    return patch_ids, hist_list





def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x

def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    assert (im.dtype == np.uint8)
    assert (im.ndim in [2, 3])

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W

def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)