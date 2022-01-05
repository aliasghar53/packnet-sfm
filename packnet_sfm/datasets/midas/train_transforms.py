import numpy as np
import cv2
import math
from functools import reduce
from packnet_sfm.utils.misc import filter_dict
from packnet_sfm.datasets.augmentations import to_tensor_sample

def to_numpy(sample):

    if "rgb" in sample:
        sample["rgb"] = np.asarray(sample["rgb"])

    if "rgb_context" in sample:
        sample["rgb_context"] = [np.asarray(image) for image in sample["rgb_context"]]
    
    return sample

def composite_function(*func):
      
    def compose(f, g):
        return lambda x : f(g(x))
              
    return reduce(compose, func, lambda x : x)


def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.
    Parameters
    ----------
    sample : dict
        Input sample
    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample



class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
       
        width, height = self.get_size(
            sample["rgb"].shape[1], sample["rgb"].shape[0]
        )
        
        # resize sample
        sample["rgb"] = cv2.resize(
            sample["rgb"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if "rgb_context" in sample:
            sample["rgb_context"] = [cv2.resize(
                                        image,
                                        (width, height),
                                        interpolation=self.__image_interpolation_method,
                                        )
                                        for image in sample["rgb_context"]]


        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample): 
        

        sample["rgb"] = sample["rgb"] / 255.0
        sample["rgb"] = (sample["rgb"] - self.__mean) / self.__std

        if "rgb_context" in sample:
            sample["rgb_context"] = [((image / 255.0) - self.__mean) / self.__std 
                                    for image in sample["rgb_context"]]

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        
        # image = np.transpose(sample["rgb"], (2, 0, 1))
        sample["rgb"] = np.ascontiguousarray(sample["rgb"]).astype(np.float32)
        
        if "rgb_context" in sample:
            sample["rgb_context"] = [np.ascontiguousarray(image).astype(np.float32)
                                    for image in sample["rgb_context"]]
        return sample

def get_midas_transform(**kwargs):
    
    net_w, net_h = kwargs["image_shape"]
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    resize = Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    prep = PrepareForNet()

    # sample will be passed from last function to the first
    transform = composite_function(to_tensor_sample, 
                                    prep, 
                                    normalization, 
                                    duplicate_sample, 
                                    resize,
                                    to_numpy)

    return transform

    

    