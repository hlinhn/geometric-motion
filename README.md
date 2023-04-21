# Interpretable axes from geometry analysis of human motion generators

### How to run

1. Set up the environment as instructed in [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [ACTOR](https://github.com/GuyTevet/MotionCLIP) and [GAN-Geometry](https://github.com/Animadversio/GAN-Geometry). To run ACTOR and MotionCLIP simultaneously, rename MotionCLIP `src` folder to `clip_src`
2. Change the path in the scripts to point to the correct path for these repos
3. The main file is `VAE_utils.py`. You can specify the generator and the distance function to test. There are a few tasks, the two most important ones are calculating the Hessian at some randomly sampled points (`--task calculate`) and visualizing the changes in generated motion sequences while moving along the dominant eigenvectors (`--task visualize`)

Example

```
python3 VAE_utils.py --task visualize --wrapper clip --scorer low --cutoff 50 --sample_class 4 --num_samples 2 --eiglist 0,4,10,30,49 --maxdist 0.5
```
