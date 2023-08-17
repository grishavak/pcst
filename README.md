# Patch-Craft Self-Supervised Training (PCST)

[Arxiv](https://arxiv.org/abs/2211.09919) | [CVF (pdf)](https://openaccess.thecvf.com/content/CVPR2023/papers/Vaksman_Patch-Craft_Self-Supervised_Training_for_Correlated_Image_Denoising_CVPR_2023_paper.pdf) | [CVF (suppl)](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Vaksman_Patch-Craft_Self-Supervised_Training_CVPR_2023_supplemental.pdf) | [Project Website](https://grishavak.github.io/pcst/) | [Presentation](https://www.youtube.com/watch?v=UDg-VG5VTc4) 

### Project website: [https://grishavak.github.io/pcst/](https://grishavak.github.io/pcst/)

### Official PyTorch implementation of the paper: "Patch-Craft Self-Supervised Training for Correlated Image Denoising."

####  CVPR 2023

## Overview
PCST (Patch-Craft Self-Supervised Training) is a self-supervised training technique designed for the removal of unknown correlated noise. This techniques does not require knowledge of the noise model nor access to ground truth targets. It takes short noisy video sequences (or bursts of noisy images) as input and does not requier frame registration within the video sequences. 

The method assumes that the contaminating noise is additive, zero mean, but not necessarily Gaussian, and it could be cross-channel and spatially correlated. An additional assumption is that the noise is (mostly) independent of the image and nearly homogeneous, i.e., having low to moderate spatially variant statistics. Examples of such noise could be Gaussian correlated noise or real image noise in digital cameras.

# Code
This code was tested with python 3.8, cuda 11.8 and pytorch 1.9.

## Requirements
- numpy
- opencv
- matplotlib
- torch
- scikit-image
- h5py

## Download datasets
The datasets utilized in the paper are available for download at the following link:
[https://drive.google.com/drive/folders/1gPyZA5LxMezhAHv3sG7g01YBqV5mEJct?usp=sharing](https://drive.google.com/drive/folders/1gPyZA5LxMezhAHv3sG7g01YBqV5mEJct?usp=sharing)

The experiments involving correlated Gaussian noise were conducted using dataset *DAVIS*, while dataset *CRVD* was used for experiments with real-world image noise. Once downloaded, place the (unzipped) datasets under the *data_set* folder.

The original *DAVIS* and *CRVD* datasets are available at the following links, repectively:

[https://davischallenge.org](https://davischallenge.org)

[https://github.com/cao-cong/RViDeNet](https://github.com/cao-cong/RViDeNet)


## Install dependencies:
``` 
python -m pip install -r requirements.txt
```

## Demos

### Denoising demonstration with correllted Gaussian noise
```
python demo_denoise_correlated_gaussian.py

```

#### The command to denoise with specific *sigma* and *k* values using a specific *model* is as follows:
```
python demo_denoise_correlated_gaussian.py --sigma <sigma> --k <k> --model <model>
```

### Denoising demonstration with real-world image noise
```
python demo_denoise_unknown_noise.py
```

#### The command to denoise with a specific *ISO* value using a specific *model* is as follows:
```
python demo_denoise_unknown_noise.py --iso <ISO> --model <model>
```

### Demo of training with correlated Gaussian noise
```
python demo_train_for_correlated_gaussian.py
```

#### The command to train with specific *sigma* and *k* values using a specific *model* is as follows:
```
python demo_train_for_correlated_gaussian.py --sigma <sigma> --k <k> --model <model>
```

### Demo of training with real-world image noise
```
python demo_train_for_unknown_noise.py
```

#### The command to train with a specific *ISO* value using a specific *model* is as follows:
```
python demo_train_for_unknown_noise.py --iso <ISO> --model <model>
```

## Detailed documentation
### preprocess_synthetic_noise.py
- This script prepares a training set for self-supervised patch-craft training with synthetic noise.

- The script takes as input a path to a folder containing clean video sequences (or bursts of images). 

- The script can be used with any noise model. 

- For using correlated Gaussian noise, set the **noise_type** to **correlated_gaussian** and select
the values of **sigma** and **noise_kernel_size**. For applying the script with any other noise model, set the **noise_type** to **custom** and implement the noise model in the **AddCustomNoise** class.

    <ins>Note:</ins> to use **AddCustomNoise**, the user should implement **\_\_init\_\_** and **\_\_call\_\_** functions, defining the necessary arguments and functionality for their custom noise generation.

The folder contaning the clean video sequences should be organized as follows

    dataset_folder
    └── train_sequences
        ├── sequence_0
        |     ├── frame_0
        |     ├── frame_1
        |     ├── ...
        |     └── frame_K
        ├── sequence_1
        |     ├── frame_0
        |     ├── frame_1
        |     ├── ...
        |     └── frame_K
        |
        ├── ...
        |
        └── sequence_M
            ├── frame_0
            ├── frame_1
            ├── ...
            └── frame_K

<ins>Note:</ins> each video sequence must contain the same number of frames which is specified by **frame_num** parameter.

#### Arguments:

**--dataset_name** (string): Name of the dataset.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images).

**--tile_size** (int): Size of samples in the training set.

**--noise_type** (string): Noise type. For correlated Gaussian noise set to **correlated_gaussian** and configure **sigma** and **noise_kernel_size** parameters. For any other noise model set to **custom** and implement **\_\_init\_\_** and **\_\_call\_\_** functions of the **AddCustomNoise** class.

**--sigma** (float): The standard deviation of the input noise (applicable only when **noise_type** is **correlated_gaussian**).

**--noise_kernel_size** (int): the size of the rectangular kernel used to convolve the noise. (applicable only when **noise_type** is **correlated_gaussian**).

**--frame_num** (int): Number of frames in each video sequence.

**--search_offs** (int): Defines spatial bounding box for nearest neaighbor search. The bounding box size is *(2* * __search_offs__ *+ 1) x (2* * **search_offs** *+ 1)*

**--patch_size** (int): Specifies patch size used in patch matching. It should be increased with **noise_sigma** and **noise_kernel_size**.

#### Outputs:
The script outputs are **h5_files/train_synthetic_noise_\<some string\>.h5** and **cov_mean_files/cov_mean_\<some string\>.pt** files. 

#### Example usage:
```
python data/preprocess_synthetic_noise.py --dataset_name DAVIS --dataset_dir "./data_set/DAVIS" --noise_type correlated_gaussian --sigma 20.0 --noise_kernel_size 3 
```

### class AddCorrelatedGaussianNoise(object)
This class provides an object for adding correlated Gaussian noise to an input video sequence or a burst of images. The object generates independent and identically distributed (i.i.d.) Gaussian noise, convolves the noise with a rectangular kernel of a given size and adds it to the input video sequence.

#### Arguments:
- **sigma** (float): The standard deviation of the Gaussian noise
- **noise_kernel_size** (int): The size of the rectangular kernel used to convolve the noise.
- **sequence** (tensor of size *3 x K x M x N*): An input video sequence composed of *K* frames of size *M x N*

#### Example usage:

Instantiate the noise object with a noise sigma of *20* and a kernel size of *3x3*
```
add_noise = AddCorrelatedGaussianNoise(20, 3)
```

Add noise to an input video sequence
```
noisy_sequence = add_noise(sequence)
```

### class AddCustomNoise(object)
This class is a template for creating a custom noise object for adding noise to an input video sequence or a burst of images. To use this class, the user should implement the **\_\_init\_\_** and **\_\_call\_\_** functions, defining the necessary arguments 
and functionality for their custom noise generation.


#### Arguments: 
The user should define the necessary arguments for their custom noise generation.

#### Example usage:

Instantiate the custom noise object
```
add_noise = AddCustomNoise(arg1, arg2, ...)
```

Add custom noise to an input video sequence
```
noisy_sequence = add_noise(sequence)
```

### preprocess_unknown_noise.py
- This script prepares a training set for self-supervised patch-craft training with an unknown noise model. 

- The script takes as input a path to a folder containing noisy video sequences (or bursts of images).

- The script does not require knowlege of the noise model. 

The folder contaning the noisy video sequences should be organized as follows:
    
    dataset_folder
    └── train_sequences
        └── noisy
            ├── sequence_0
            |     ├── frame_0
            |     ├── frame_1
            |     ├── ...
            |     └── frame_K
            ├── sequence_1
            |     ├── frame_0
            |     ├── frame_1
            |     ├── ...
            |     └── frame_K
            |
            ├── ...
            |
            └── sequence_M
                ├── frame_0
                ├── frame_1
                ├── ...
                └── frame_K

<ins>Note:</ins> each video sequence must contain the same number of frames which is specified by **frame_num** parameter.

#### Arguments:
**--dataset_name** (string): Name of the dataset.

**--dataset_dir** (string): Path to a folder containing noisy video sequences or bursts of images.

**--tile_size** (int): Size of samples in the training set.

**--frame_num** (int): Number of frames in each video sequence.

**--search_offs** (int): Defines spatial bounding box for nearest neaighbor search. The bounding box size is *(2* * **search_offs** *+ 1) x (2* * **search_offs** *+ 1)*

**--search_skip** (int): Defines skip size for nearest neaighbor search.

**--search_offs2** (int): Applicable only when __search_skip__ *> 1*. Defines spatial bounding box for nearest neaighbor search with __skip__ *= 1*. The bounding box size is *(2* * **search_offs2** + *1) x (2* * **search_offs2** *+ 1)*

**--patch_size** (int): Specifies patch size used in patch matching.

**--replicate_dataset** (int): If __replicate_dataset__ *= 1*, then the dataset is augmented by replicating each sequence __frame_num__ *- 1* times (to the total amount of **frame_num** frames), where in each replica a different image is used as a middle frame.

**--image_format** (string): Defines images format. **8_bit** - any *8* bit format supported by OpenCV imread function (e.g., PNG, JPG), **png_16_bit** - *PNG-16* bit. To reduce quantization errors we recommend to store noisy images in PNG-16 format. 

#### Outputs:
The script outputs are **h5_files/train_\<dataset name\>.h5** and **cov_mean_files/cov_mean_\<dataset name\>.pt** files. 

#### Example usage:
```
python data/preprocess_synthetic_noise.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD/CRVD_ISO25600" --replicate_dataset 1
```

### train_for_synthetic_noise.py
**IMPORTANT:**

**PRIOR APPLYING THIS SCRIPT WE RECOMMEND PRETRAINING THE NETWORK FOR BLIND GAUSSIAN DENOISISNG.**

- This script trains a given denoising architecture with synthetic noise.

- The script reads **train_\<some string\>.h5** and **cov_mean_\<some string\>.pt** files prepared by the **./data/preprocess_synthetic_noise.py** script. These files are located in **./h5_files** and **./cov_mean_files** folders respectively.

- The script can be used for training a custom architecture. To training a custom architecture, the user must implement the **\_\_init\_\_** and **forward(self, x)** functions in the **./models/custom_model.py** file, and **load_model_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

- This script can train models using user defined (custom) training function. To do this, the user must implement **custom_training_func(opt, load_model)** function.

#### Arguments:

**--dataset_name** (string): Name of the dataset. This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images). This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--batch_size** (int): Batch size for training.

**--reduce_dependency** (int): When equal to *1* applies dependency reduction method described in the paper. We recommend setting this oparameter to *1*.

**--noise_type** (string): Noise type. For correlated Gaussian noise set to **correlated_gaussian** and configure **sigma** and **noise_kernel_size** parameters. For any other noise model set to **custom** and implement **\_\_init\_\_** and **\_\_call\_\_** functions of the **AddCustomNoise** class.

This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--sigma** (float): The standard deviation of the input noise. (applicable only when **noise_type** is **correlated_gaussian**).

This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--noise_kernel_size** (int): the size of the rectangular kernel used to convolve the noise (applicable only when **noise_type** is **correlated_gaussian**).

 This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--lr** (float): Initial learning rate.

**--num_epochs** (int): Number of epochs for training.

**--model** (string): Training architecture. Currently supported models are: **dncnn**, **unet**, and **custom**. For training custom architecture user must set this parameter to **custom** and implement functions **\_\_init\_\_** and **forward(self, x)** in **./models/custom_model.py** file, and **load_model_custom(opt, device, model_path)** function in **./models/model_util**.py' file.

**--training_function** (string): Training function. When set to **ready-made** uses **training_func(opt, load_model)** function (implemented in the current file) for training. When set to **custom** uses user defined (custom) training function. For training using user defined function user must implement **custom_training_func(opt, load_model)** function.


#### Outputs:
The script outcome is **/experiments/\<model_name\>/learned/\<dataset_name\>/state_params_\<some string\>.pt** file.

#### Example usage:
```
python train_for_synthetic_noise.py --dataset_name DAVIS --dataset_dir "./data_set/DAVIS" --noise_type correlated_gaussian --sigma 20.0 --noise_kernel_size 3 --model dncnn
```

### train_for_unknown_noise.py
**IMPORTANT:**

**PRIOR APPLYING THIS SCRIPT WE RECOMMEND PRETRAINING THE NETWORK FOR BLIND GAUSSIAN DENOISISNG.**

- This script trains a given denoising architecture with noise with unknown model.

- The script reads **train_\<some string\>.h5'** and **cov_mean_\<some string\>.pt** files prepared by the **./data/preprocess_synthetic_noise.py** script. These files are located in **./h5_files** and **./cov_mean_files** folders respectively.

- The script can be used for training a custom architecture. To training a custom architecture, the user must implement the **\_\_init\_\_** and **forward(self, x)** functions in the **./models/custom_model.py** file, and **load_model_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

- This script can train models using user defined (custom) training function. To do this, the user must implement **custom_training_func(opt, load_model)** function.


The folder contaning the noisy validation images should be organized as follows:

    dataset_folder
    └── validation_images
        ├── clean
        |   ├── image_0
        |   ├── image_1
        |   ├── ...
        |   ├── image_k
        └── noisy
            ├── image_0
            ├── image_1
            ├── ...
            └── image_k

<Ins>Note:</ins> Names of the noisy images must match the names of the corresponding clean images.


#### Arguments:

**--dataset_name** (string): Name of the dataset. 

This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images).

This argument must match the corresponding argument of **./data/preprocess_synthetic_noise.py** script.

**--batch_size** (int): Batch size for training.

**--reduce_dependency** (int): When equal to *1* applies dependency reduction method described in the paper. We recommend setting this oparameter to *1*.

**--lr** (float): Initial learning rate.

**--num_epochs** (int): Number of epochs for training.

**--model** (string): Training architecture. Currently supported models are: **dncnn**, **unet**, and **custom**. For training custom architecture user must set this parameter to **custom** and implement functions **\_\_init\_\_** and **forward(self, x)** in **./models/custom_model.py** file, and **load_model_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

**--training_function** (string): Training function. When set to **ready-made** uses **training_func(opt, load_model)** function (implemented in the current file) for training. When set to **custom** uses user defined (custom) training function. For training using user defined function user must implement **custom_training_func(opt, load_model)** function.


#### Outputs:
The script outcome is **/experiments/\<model_name\>/learned/\<dataset_name\>/state_params_\<some string\>.pt** file.

#### Example usage:
```
python train_for_unknown_noise.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD_ISO25600" --model dncnn 
```

### test_with_synthetic_noise.py
- This script evaluates a given denoising architecture with synthetic noise.

- This script can evaluate models using user defined (custom) test function. To do this, the user must implement **custom_test_func(opt, load_model)** function.


#### Arguments:
**--dataset_name** (string): Name of the dataset. 

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images).

This argument must match the corresponding argument of **.train_for_synthetic_noise.py** script.

**--noise_type** (string): Noise type. For correlated Gaussian noise set to **correlated_gaussian** and configure **sigma** and **noise_kernel_size** parameters. For any other noise model set to ***custom** and implement **\_\_init\_\_** and **\_\_call\_\_** functions of the **AddCustomNoise** class.

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--sigma** (float): The standard deviation of the input noise (applicable only when **noise_type** is **correlated_gaussian**).

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--noise_kernel_size** (int): the size of the rectangular kernel used to convolve the noise (applicable only when the **noise_type** is **correlated_gaussian**).

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--save_images** (int): When set to *1*, the script saves clean, noisy and reconstructed images.

**--save_psnr_arr** (int): When set to *1*, the script saves arrays with *PSNR* and *SSIM* values.

**--model** (string): Tested architecture. Currently supported models are: **dncnn**, **unet**, and **custom**. For evaluating custom architecture user must set this parameter to **custom** and implement **load_state_params_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

**--test_function** (string): Test function. When set to **ready-made** uses **training_func(opt, load_model)** function (implemented in the current file) for training. When set to **custom** uses user defined (custom) training function. For training using user defined function user must implement **custom_training_func(opt, load_model)** function.


#### Outputs:
The script outcome is: 
- clean images
- noisy images
- denoised images
- psnr_ssim_arr.pt file that contain PSNR values.


#### Example usage:
```
python test_with_synthetic_noise.py --dataset_name DAVIS --dataset_dir "./data_set/DAVIS" --noise_type correlated_gaussian --sigma 20.0 --noise_kernel_size 3 --model dncnn
```

### test_with_unknown_noise.py
- This script evaluates a given denoising architecture with noise with unknown model.

- This script can evaluate models using user defined (custom) test function. To do this, the user must implement **custom_test_func(opt, load_model)** function.


#### Arguments:
**--dataset_name** (string): Name of the dataset. 

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images).

This argument must match the corresponding argument of **.train_for_synthetic_noise.py** script.

**--save_images** (int): When set to *1*, the script saves clean, noisy and reconstructed images.

**--save_psnr_arr** (int): When set to *1*, the script saves arrays with *PSNR* and *SSIM* values.

**--model** (string): Tested architecture. Currently supported models are: **dncnn**, **unet**, and **custom**. For evaluating custom architecture user must set this parameter to **custom** and implement **load_state_params_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

**--test_function** (string): Test function. When set to **ready-made** uses **training_func(opt, load_model)** function (implemented in the current file) for training. When set to **custom** uses user defined (custom) training function. For training using user defined function user must implement **custom_training_func(opt, load_model)** function.


#### Outputs:
The script outcome is: 
- clean images
- noisy images
- denoised images
- **psnr_ssim_arr.pt** file that contain *PSNR* and *SSIM* values.


#### Example usage:
```
python test_with_unknown_noise.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD_ISO25600" --model dncnn 
```

### test_with_synthetic_noise_training_sequences.py

- This script evaluates a given denoising architecture with noise with unknown model. The script uses for evaluation **TRAINING SEQUENCES**, as it was done in the paper.

- This script can evaluate models using user defined (custom) test function. To do this, the user must implement **custom_test_func(opt, load_model)** function.


#### Arguments:
**--dataset_name** (string): Name of the dataset. 

This argument must match the corresponding argument of **train_for_synthetic_noise.py** script.

**--dataset_dir** (string): Path to a folder containing clean video sequences (or bursts of images).

This argument must match the corresponding argument of **.train_for_synthetic_noise.py** script.

**--save_images** (int): When set to *1*, the script saves clean, noisy and reconstructed images.

**--save_psnr_arr** (int): When set to *1, the script saves arrays with *PSNR* and *SSIM* values.

**--model** (string): Tested architecture. Currently supported models are: **dncnn**, **unet**, and **custom**. For evaluating custom architecture user must set this parameter to **custom** and implement **load_state_params_custom(opt, device, model_path)** function in **./models/model_utils.py** file.

**--test_function** (string): Test function. When set to **ready-made** uses **training_func(opt, load_model)** function (implemented in the current file) for training. When set to **custom** uses user defined (custom) training function. For training using user defined function user must implement **custom_training_func(opt, load_model)** function.


#### Outputs:
The script outcome is: 
- clean images
- noisy images
- denoised images
- **psnr_ssim_arr.pt** file that contain PSNR values.


#### Example usage:
```
python test_with_unknown_noise_training_sequences.py --dataset_name CRVD_ISO25600 --dataset_dir "./data_set/CRVD_ISO25600" --model dncnn 
```





