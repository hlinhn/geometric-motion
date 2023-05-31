# Interpretable axes from geometry analysis of human motion generators

Based on the work done by Wang and Ponce [[1]](#1), we analyzed the structure of the latent space of human motion learned by VAEs, to understand if these spaces contain interpretable change directions even without explicit training.

### How to run

1. `conda env create -f environment.yml`
2. Get the code from [MotionCLIP](https://github.com/GuyTevet/MotionCLIP) [[2]](#2), [ACTOR](https://github.com/Mathux/ACTOR) [[3]](#3) and [GAN-Geometry](https://github.com/Animadversio/GAN-Geometry) [[1]](#1). To run ACTOR and MotionCLIP simultaneously, rename MotionCLIP `src` folder to `clip_src`
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


#### References
<a id="1">[1]</a>
Binxu Wang, Carlos R. Ponce (2021).
The Geometry of Deep generative image models and its Application.
ICLR 2021

<a id="2">[2]</a>
Guy Tevet, Brian Gordon, Amir Hertz, Amit H. Bermano, Daniel Cohen-Or (2022).
MotionCLIP: Exposing Human Motion Generation to CLIP Space.
ECCV 2022

 <a id="3">[3]</a>
Mathis Petrovich, Michael J. Black, Gül Varol (2021).
Action-Conditioned 3D Human Motion Synthesis with Transformer VAE.
ICCV 2021
