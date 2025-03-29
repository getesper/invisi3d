# Gld8tr-3dcn: Generating 3D Scenes from single image

This repository contains the code to train the depth completion network, generate 3D scenes, and run the scene geometry evaluation benchmark 

Overview: here are two key advancements in Glad8tr for 3D scene generation, an area rapidly evolving due to progress in 2D diffusion models. 
First, we identify and rectify the limitations of monocular depth estimation for 3D lifting, by introducing a depth completion model that integrates existing scene geometry through teacher distillation and self-training, resulting in superior geometric consistency. 
Second, we establish a geometry-centric benchmarking scheme, replacing text-based evaluations, to directly quantify the structural fidelity of generated 3D scenes.

## Release Roadmap
- [x] Inference
- [x] High-Quality Gaussian Splatting Results
- [x] Training
- [x] Benchmark

## Getting Started
Use the `environment.yml` file to create a Conda environment with all requirements.

## Inference

To generate a 3D scene, invoke the `run.py` script:

```shell
python3 run.py \
  --image "examples/photo-1667788000333-4e36f948de9a.jpeg" \
  --prompt "a street with traditional buildings in Kyoto, Japan" \
  --output_path "./output.ply" \
  --mode "stage"
```

For the parameter `mode`, you may provide one of the following arguments:

* `single`: Simple depth projection of the input image (no hallucation)
* `stage`: Single-step hallucination of the scene to the left and right of the input image
* `360`: Full 360-degree hallucination around the given input image

To run a 360-degree "hallucination", it is recommened to use a GPU with at least 16 GB VRAM.

## Training

### Dataset Setup

To train the depth completion network from a fine-tuned model, we need to generate some data first. 
First, predict depth for [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) with [Marigold](https://github.com/prs-eth/Marigold). 
Second, use Marigold again to predict the depth for [Places365](http://places2.csail.mit.edu/) (original). Third, use the depth maps for Places365 to generate inpainting masks.
Places365 can be used as-is. 
For NYU Depth v2, please follow the instructions [here](https://github.com/cleinc/bts/tree/master/pytorch#nyu-depvh-v2) to obtain the `sync` folder.
We also need the official splits for NYU Depth v2, which can be extracted with the script `extract_official_train_test_set_from_mat.py` provided [here](https://github.com/wl-zhao/VPD/blob/main/depth/extract_official_train_test_set_from_mat.py):

```shell
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Next, update the paths in `predict_nyu_marigold.py`, `predict_places_marigold.py`, and `project_places_depth_preds.py`. Then run these files in this sequence. These scripts are equipped with `submitit` to be distributed across a SLURM cluster. If possible, we strongly suggest parallelizing the workload.

Make sure to update the paths in `zoedepth/utils/config.py:96-175`. 

Now, ready to roll!

### Training the Model

```shell
python3 train.py -m zoedepth -d marigold_nyu \
 --batch_size=12 --debug_mode=0 \
 --save_dir="checkpoints/"
```

Consider using the `_latest.pt` as opposed to the `_best.pt` checkpoint.

## Benchmark

This scene geometry evaluation benchmark is an approach to quantitatively compare the consistency of generated scenes. In this section, we describe how to obtain and process the datasets used, and how to run the evaluation itself.

The datasets are placed in a `datasets` folder at the root of the repository. However, this path can be adapted.

### ScanNet

Obtain [ScanNet](http://www.scan-net.org) and place it into `datasets/ScanNet`. Clone the ScanNet repository to obtain the library to read the sensor data, and then run our small script to extract the individual image, depth, pose, and intrinsics files.

```
git clone https://github.com/ScanNet/ScanNet
python3 benchmark/scannet_process.py ./datasets/ScanNet --input_dir ./datasets/ScanNet/scans
```

### Hypersim

Download [Hypersim](https://github.com/apple/ml-hypersim) into `datasets/hypersim`. We additionally need the camera parameters for each scene.

```bash
wget https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv -P datasets/hypersim/
```

Then, we find image pairs with a high overlap that we will evaluate on:

```
python3 hypersim_pair_builder.py --hypersim_dir ./datasets/hypersim
```

If you have access to a cluster running SLURM, you can submit a job similar to the following (you probably need to adapt `constraint` and `partition`).

```bash
#!/bin/bash

# Parameters
#SBATCH --array=0-100%20
#SBATCH --constraint=p40
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hypersim
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=low-prio-gpu
#SBATCH --requeue

python3 hypersim_pair_builder.py --hypersim_dir ./datasets/hypersim --scene_idx ${SLURM_ARRAY_TASK_ID}
```

### Running the Evaluation

Once you have finished setting up both datasets, the evaluation may be run for each dataset.


The ScanNet evaluation script will test on the first 50 scenes and print the mean absolute error across these scenes.
```
python3 scannet_eval.py 
```

The Hypersim evaluation script will consider all image pairs that were previously computed and generate the errors for each scene as csv files. We then concatenate them and also calculate the mean absolute error.

```
python3 hypersim_eval.py --out_dir ./datasets/hypersim/results_invisible_stitch
python3 hypersim_concat_eval.py ./datasets/hypersim/results_invisible_stitch
```

Adapting these scripts to your own model is straight forward: In `scannet_eval.py`, add a new mode for your model (see lines `372-384`). In `hypersim_eval.py`, duplicate the error computation for an existing model and adapt it to your own (see lines `411-441`).

## Acknowledgments

Without the great works from various researchers at Oxford, University of Liverpool, and this would not have been completed. Thank you! The code for the depth completion network heavily borrows from [ZoeDepth](https://github.com/isl-org/ZoeDepth). We utilize [PyTorch3D](https://pytorch3d.org) in our 3D scene generation pipeline.
