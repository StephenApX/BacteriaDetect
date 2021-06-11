# BacteriaDetect
This project is the machine vision algorithm implementation of bacterial detection, which aims to detect Bacteria circles' location from biochip and output its CIE values in a rectangle area.

## Article
Point-of-Care Pathogen Testing Using Photonic Crystals and Machine Vision for Diagnosis of Urinary Tract Infections 
[https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.0c04942](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.0c04942 "https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.0c04942")

## Requirement
    python = 3.7.8
    numpy = 1.19.1
    opencv = 4.4.0
Creating a new conda environment is recommended. Run this script like:
`conda create -n test python=3.7 opencv numpy -c conda-forge
`

## List of Arguments
| Argument | Required/Not  | Details  |
| ------------ | ------------ | ------------ |
| -h, --help | optional  | show help message and exit.  |
|  -i, --in_dir  |  required | input directory (relative dir.) of images.  |
| -o, --out_dir  | optional  | output dir of detected images.  |
| -d, --min_circle_distance  | optional  |  Minimum distance of adjacent circles(pixels). **Default**: 60 |
|  -e, --edge_detect_thres |  optional | Contrast threshold between circle edge and background. **Default**: 26 |
|  -r, --roundness_thres | optional  | Roundness threshold of circles. **Default**: 31  |
| --min_circleRadius |  optional |  Minimum of circle radius. **Default**: 20 |
| --max_circleRadius  | optional  |  Maximum of circle radius. **Default**: 90 |

## Usage
1. Put test images in the **testdata/images.tif** in the form of a folder.
2. Open anaconda/cmd prompt, **activate** the installed environment, and change the **working directory** to the current project.
3. When not specifying the output path, a folder named by **runtime** will be generated in the current directory by default.
4. The above five default **detection** **parameters** can be modified appropriately when the detection is incomplete.
5. Run python script like: `python identify.py -i testdata/`

## Testdata and Results
#### Input image
[![](https://github.com/StephenApX/BacteriaDetect/blob/main/testdata/1.tif)](https://github.com/StephenApX/BacteriaDetect/blob/main/testdata/1.tif)

#### Output results
[![](https://github.com/StephenApX/BacteriaDetect/blob/main/testresults/1_Rect.tif)](https://github.com/StephenApX/BacteriaDetect/blob/main/testresults/1_Rect.tif)
[![](https://github.com/StephenApX/BacteriaDetect/blob/main/testresults/1_Circ.tif)](https://github.com/StephenApX/BacteriaDetect/blob/main/testresults/1_Circ.tif)

## Citation
Please consider citing the following article if you used this project in your research.

    @article{liu2021point,
    	title={Point-of-Care Pathogen Testing Using Photonic Crystals and Machine Vision for Diagnosis of Urinary Tract Infections},
    	author={Liu, Haoran and Li, Zhihao and Shen, Ruichen and Li, Zhiheng and Yang, Yanbing and Yuan, Quan},
    	journal={Nano Letters},
    	volume={21},
    	number={7},
    	pages={2854--2860},
    	year={2021},
    	publisher={ACS Publications}
    }

