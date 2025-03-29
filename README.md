Alright, listen up, folks. We've got this thing, Glad8tr, right? It's cranking out 3D scenes, and it's getting wild because of these 2D diffusion models that are just... exploding. We've hit a couple of major breakthroughs.

First off, we tackled this huge problem with guessing depth from a single viewpoint. Like, trying to build a house with one blueprint, it's just... flawed. So, we built this depth completion model, a serious piece of tech. It learns from existing geometry, like a sponge, and then teaches itself, making these 3D scenes way more consistent, way more solid.

Then, we realized judging these scenes based on text descriptions? It's like, judging a racehorse by its poetry. You gotta look at the actual structure. So, we built a geometry-centric benchmark, measuring the actual shapes and angles. It's about real, objective data, not just opinions.

Now, here's the roadmap:

Inference: We're pushing for high-quality Gaussian Splatting results, so these scenes look real.
Training: We're fine-tuning the model for even better performance.
Benchmark: We're refining our geometry-based evaluation to get even more accurate measurements.
Alright, let's get you started:

You'll need a Conda environment, just use the environment.yml file.
Inference:

To generate a 3D scene, run this command:
Bash

python3 run.py \
  --image "examples/photo-1667788000333-4e36f948de9a.jpeg" \
  --prompt "a street with traditional buildings in Kyoto, Japan" \
  --output_path "./output.ply" \
  --mode "stage"
You've got different modes: single, stage, and 360. 360 is intense, you'll need a serious GPU.
Training:

Dataset setup is crucial. You'll be working with NYU Depth v2 and Places365. You'll need to predict depth with Marigold, generate inpainting masks, and handle the official splits for NYU Depth v2. This is a big job, and parallelizing it on a SLURM cluster is highly recommended.
Then you train the model with the training script. Use the latest checkpoint.
Benchmark:

We're using ScanNet and Hypersim for evaluation.
You'll need to download and process these datasets.
Then, you can run the evaluation scripts.
Adapting these scripts for your own models is straightforward. Just add a new mode in scannet_eval.py or duplicate and adapt the error computation in hypersim_eval.py.
Basically, we're pushing the boundaries of 3D scene generation, making it more accurate, more realistic. It's a wild ride, and we're just getting started.
