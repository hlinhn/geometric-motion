# Interpretable axes from geometry analysis of human motion generators

### How to run

1. `conda env create -f environment.yml`
2. Get the code from [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [ACTOR](https://github.com/GuyTevet/MotionCLIP) and [GAN-Geometry](https://github.com/Animadversio/GAN-Geometry). To run ACTOR and MotionCLIP simultaneously, rename MotionCLIP `src` folder to `clip_src`
3. Change the path in `tools/paths.py` to point to the correct path for these repos
4. The main file is `main.py`. You can specify the generator and the distance function to test. There are a few tasks, the two most important ones are calculating the Hessian at some randomly sampled points (`python3 main.py calculate`) and visualizing the changes in generated motion sequences while moving along the dominant eigenvectors (`python3 main.py visualize`)

Example

```
python3 main.py visualize --wrapper clip --scorer low --cutoff 50 --sample_class 4 --num_samples 2 --eiglist 0,4,10,30,49 --maxdist 0.5
```

### How to add another distance function

1. Add the distance function to `tools/sim_score.py`
2. Update the list of `available_dist_functions` in `tools/paths.py`
3. Update `make_scorer` in `main.py`

### How to add another generator

1. Add the wrapper file of the generator to `tools/`. This wrapper should include a `sample_vector` function and a `generate` function
2. Update the list of `available_wrappers` in `tools/paths.py`
3. Update `make_wrapper` in `main.py`
