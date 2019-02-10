# flow-VAE
_A PyTorch implementation of the training procedure of [1] with normalizing flows for enriching the family of approximate posteriors. The first installation of normalizing flows in a variational inference framework was proposed in [2]._

## Implementation Details
This implementation supports training on four datasets, namely **MNIST**, **Fashion-MNIST**, **SVHN** and **CIFAR-10**. For each dataset, only the training split is used for learning the distribution. Labels are left untouched. Raw data (except for MNIST) is subject to dequantization, random horizontal flipping and logit transformation. Adam with default parameters are used for optimization.

Four types of flows are implemented, including two types of _general normalizing flows_ and two types of _volume-preserving flows_.

- **Planar flow** and **radial flow** are the general normalizing flows proposed in [2].
- **Householder flow** is a volume-preserving flow proposed in [3].
- **NICE** is another volume-preserving flow proposed in [4]. Although characterized as volume-preserving flow, NICE can be augmented by a scaling operation, which is included as the last step of the flow in this implementation.

Three variants of variational autoencoder (VAE) with normalizing flows are implemented. Their specifications and potential issues are summarized below.

- **VAE with dynamic flows** (see ```dynamic_flow_vae.py```). Flow parameters are treated as output of the encoder. In other words, they change with respect to input. This is a principled (and most flexible) way of parametrizing the approximate posteriors and is suggested by [2]. Unfortunately, this approach is largely impractical given the potentially huge number of flow parameters, and is extremely susceptible to numerical instability. Furthermore, it leads to little, if any, improvement on the objective value.  

-  **VAE with static flows** (see ```static_flow_vae.py```). Flow parameters are treated as learned parameters. In other words, they remain fixed with respect to input. This significantly constrains the richness of approximate posteriors, but is more practical and alleviates (but does not fully resolve) the numerical issue seen with dynamic flows. I also observed moderate improvement on the objective value under certain circumstances, particularly with radial flows. However, sampling becomes problematic since most latent codes lie outside of the most probable region under the standard normal prior (see figures below).

- **Convolutional VAE with static flow** (see ```static_flow_conv_vae.py```). Convolutional layers instead of dense layers are used in the encoder and the decoder.

Based on the experiments, I concluded that the idea of enriching approximate posteriors in VAE using normalizing flows is theoretically appealing but can easily cause more headache than benefit in real practice. Flow-based generative models may be better options if the goal is to improve data likelihood. You are welcome to check out my implementation of [NICE](https://github.com/fmu2/NICE), [realNVP](https://github.com/fmu2/realNVP) and Glow (forthcoming).

## Result
I only show results on MNIST. The models use dense layers and static flows. The obeservation carries over to other datasets. 

**MNIST** 

_model without flow_  

samples (after 10000 iterations)  

![](https://github.com/fmu2/flow-VAE/blob/master/samples/mnist_bs128_ly2_hd400_lt2_gt0__10000.png?raw=true)

latent space

![](https://github.com/fmu2/flow-VAE/blob/master/plots/mnist_bs128_ly2_hd400_lt2_gt0_latent.png?raw=true)

_model with planar flows (16 steps)_

samples (after 10000 iterations, observe highly homogenous samples) 

![](https://github.com/fmu2/flow-VAE/blob/master/samples/mnist_bs128_ly2_hd400_lt2_gt0__planar_len16_10000.png?raw=true)

latent space (observe inflation in the region with high uncertainty)

![](https://github.com/fmu2/flow-VAE/blob/master/plots/mnist_bs128_ly2_hd400_lt2_gt0__planar_len16latent.png?raw=true)

_model with radial flows (16 steps)_

samples (after 10000 iterations, observe highly homogenous samples) 

![](https://github.com/fmu2/flow-VAE/blob/master/samples/mnist_bs128_ly2_hd400_lt2_gt0__radial_len16_10000.png?raw=true)

latent space (observe inflation in the region with high uncertainty)

![](https://github.com/fmu2/flow-VAE/blob/master/plots/mnist_bs128_ly2_hd400_lt2_gt0__radial_len16latent.png?raw=true)

_model with householder flows (16 steps)_

samples (after 10000 iterations) 

![](https://github.com/fmu2/flow-VAE/blob/master/samples/mnist_bs128_ly2_hd400_lt2_gt0__householder_len16_10000.png?raw=true)

latent space (observe rotation)

![](https://github.com/fmu2/flow-VAE/blob/master/plots/mnist_bs128_ly2_hd400_lt2_gt0__householder_len16latent.png?raw=true)

_model with NICE flows (16 steps)_

samples (after 10000 iterations) 

![](https://github.com/fmu2/flow-VAE/blob/master/samples/mnist_bs128_ly2_hd400_lt2_gt0__nice_len16_10000.png?raw=true)

latent space (observe expansion of the support)

![](https://github.com/fmu2/flow-VAE/blob/master/plots/mnist_bs128_ly2_hd400_lt2_gt0__nice_len16latent.png?raw=true)

## Training
Code runs on a single GPU and has been tested with

- Python 3.7.2
- torch 1.0.0
- numpy 1.15.4

Examples:

```
python dynamic_flow_vae.py --dataset=mnist --batch_size=128 --flow=radial --length=16  
python static_flow_vae.py --dataset=mnist --batch_size=128 --flow=radial --length=16  
python static_flow_conv_vae.py --dataset=mnist --batch_size=128 --flow=radial --length=16  
```

## Reference
[1] Diederik P Kingma, Max Welling. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). ICLR 14.  
[2] Danilo Jimenez Rezende, Shakir Mohamed. [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770). ICML 15.  
[3] Jakub M. Tomczak, Max Welling. [Improving Variational Auto-Encoders using Householder Flow](https://arxiv.org/abs/1611.09630). NIPS 16 workshop.  
[4] Laurent Dinh, David Krueger, Yoshua Bengio. [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516). ICLR 15 workshop.