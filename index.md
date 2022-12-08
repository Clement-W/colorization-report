<!-- #region -->

# COLORIZED - Colorization Of Lost Or Ruined ImageZ Employing Diffusion

Clément Weinreich

Brennan Whitfield

Vivek Shome

<details>
<summary>Table of Contents</summary>
    <ul>
        <li> <a href="#Introduction">Introduction</a></li>
        <li> <a href="#Project_Management"> Project Management </a> </li>
        <li> <a href="#Instructions"> Instructions </a></li>
        <details> <summary> <a href="#Theory">Theory </a></summary> 
                <ul>
                    <li><a href="#Theory_General">General Theory</a></li>
                    <li><a href="#Notation">Notation</a></li>
                    <li><a href="#Forward_Diffusion_Theory">Forward Diffusion Theory</a></li>
                    <li><a href="#Reverse_Diffusion_Theory">Reverse Diffusion Theory</a></li>
                </ul>
        </details>
        <li><a href="#Loss_Function">Loss Function</a></li>
        <li><a href="#Algo_Changes">Changes in the Algorithms</a></li>
        <details> 
            <summary> <a href="#Implementation"> Implementation </a> </summary>
            <ul> 
                <li> <a href="#Diffusion_Basics"> Diffusion Basics </a></li>
                <li> <a href="#Forward_Process">Forward Process</a></li>
                <details> <summary> <a href="#Neural_Network">Neural Network</a> </summary>
                    <ul>
                        <li><a href="#UNet_Overview">Overview of the U-Net architecture</a> </li>
                        <li><a href="#Timestep_Embedding">Timestep Embedding</a></li>
                        <li><a href="#Conditional_input">Conditional Image Input</a></li>
                        <li><a href="#Core_Convolution_Block">Core Convolution Block</a></li>
                        <li><a href="#Resnet_Block">Resnet Block</a></li>
                        <li><a href="#Sample_Blocks">Up/Down Sampling Blocks</a></li>
                        <details> <summary> <a href="#Custom_Architecture">Our Custom UNet Architecture</a> </summary>
                            <ul>
                                <li><a href="#Encoder">Encoder</a></li>
                                <li><a href="#Bottleneck">Bottleneck</a></li>
                                <li><a href="#Decoder">Decoder</a></li>
                            </ul>
                        </details>
                    </ul>
                </details>
                <li><a href="#Reverse_Process">Reverse Process</a></li>
                <li><a href="#Training">Training</a></li>
            </ul>
            </details>
        <li><a href="#Results">Results</a></li>
        <li><a href="#Discussions">Discussions</a></li>
        <li><a href="#Installation">Installation</a></li>
        <li><a href="#Model_How_To">How to Use Our Model</a></li>
        <li><a href="#Conclusion">Conclusion</a></li>
        <li><a href="#References">References</a></li>
    </ul>
</details>

<div id="Introduction"></div>

## Introduction

The real world problem we sought to solve was the colorization of greyscale images found in the Shields library archive here in Davis. This project intended to provide the library with a means of viewing a local landmark/area from the perspective of an individual present at the original time. To achieve this, we decided to employ an image-to-image conditional diffusion model, which would be trained on a dataset ([Image Colorization on Kaggle](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code)) of 50 thousand images, consisting of 25 thousand 224 x 224 greyscale-color image pairs. 

As the information on this direct concept is fairly sparse, we took it upon ourselves to seek an understanding of the model at every level. Moreover, we recognized the value of comprehending all mathematical theory involved (as this is a math course), and have thus detailed below the theory, as well as the steps we took to grasp each idea. We would like to emphasize the amount of time which went into both the writing, as well as the thought, behind each step in the process. This project was quite ambitious, so we additionally acknowledge the tradeoff made between performance of the model, and the depth in which we understood it.

In the following sections, we dissect why a conditional diffusion model works, the mathematics behind it, and the implementation of said model.

<div id="Project_Management"></div>

## Project Management

Below, is a visualization of this project's sequencing/progress. The image displays, as requested, four solid weeks of work. Additionally, we provide [here](https://trello.com/invite/b/EfGXzV9j/ATTI5b46ac6a331db37c54495c8a4ae25e13EADADB82/colorized-project) a link to our Trello board, which further expresses our dedication to the task at hand. Work was evenly dispersed, therefore we find no need to categorize individual contributions. 

<img src="assets/Project_Management_Vis.png" >

<div id="Theory"></div>

## Theory of Conditional Diffusion Process

Brennan Whitfield - Clément Weinreich - Vivek Shome

<div id="Theory_General"></div>

### General Idea

The idea of diffusion models is to slowly destroy structure in a data distribution through an iterative forward process. Then, we learn a reverse diffusion process using an neural network, that restores structure in data. This model yield to a highly flexible generative model of the data. This can be seen as a Markov chain of diffusion steps, which slowly add random noise to the data, and then learn to reverse the diffusion process in order to construct new desired data samples from the noise. 

In the case of image colorization, we use a conditional diffusion model which, on top of requiring a noisy image and timestep, takes a grayscale image with the intent of recoloring said image. 

<div id="Notation"></div>

### Notation

The following notation will be adopted for the next parts:

-   $\mathcal{N}(x;\mu,\sigma^2)$ : sampling x from a normal distribution of mean $\mu$ and variance $\sigma^2$
-   $\mathbf{x_t}$ is the image after applying $t$ iterations of noise through the forward process
-   $\mathbf{x_0}$ is the original (color) image
-   $\mathbf{z}$ is the grayscale source image we seek to colorize
-   $\mathbf{x_T}$ is the final image of the forward process which follows an isotropic Gaussian distribution ($T$ is constant)
-   $q(\mathbf{x_t}|\mathbf{x_{t-1}})$ corresponds to the forward process, taking an image $\mathbf{x_{t-1}}$ as input, and output $\mathbf{x_t}$ which contains more noise
-   $p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})$ corresponds to the reverse process, taking an image $\mathbf{x_t}$ as input, and output $\mathbf{x_{t-1}}$ which contains less noise

<div id="Forward_Diffusion_Theory"></div>


### The Forward Diffusion Process

Let's sample an image from a real conditional data distribution $\mathbf{x_0} \sim q(\mathbf{x}|\mathbf{z})$. We define a forward diffusion process, in which a small amount of Gaussian noise is iteratively added to the image $\mathbf{x_0}$, in $T$ steps, leading to the sequence of noisy images $\mathbf{x_1},\dots,\mathbf{x_T}$. The step size is controlled by a variance schedule $\beta_t$ going from a start value to an end value defined accordingly to the scale of the pixel's values, in $T$ steps, starting at $t=1$. The noise added is sampled from a Gaussian distribution. Thus we can define:
$$q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t};\sqrt{1-\beta_t}\mathbf{x_{t-1}},\beta_t\mathbf{I})$$
Where the variance schedule scales the mean and the variance of the noise sampled from the normal distribution. Since our forward process is a Markov Chain (satisfying Markov property), we can also write:


\begin{align*}
q(\mathbf{x_{1:T}}|\mathbf{x_0}) &= q(\mathbf{x_1}, \dots, \mathbf{x_T} | \mathbf{x_0}) \\
               &= \frac{q(\mathbf{x_0}, \mathbf{x_1}, \dots, \mathbf{x_T})}{q(\mathbf{x_0})} &&\text{(Bayes' Theorem)}\\
               &= \frac{q(\mathbf{x_0})q(\mathbf{x_1}|\mathbf{x_0})\dots q(\mathbf{x_T}|\mathbf{x_{T-1}})}{q(\mathbf{x_0})} &&\text{(Markov property)}\\
               &= q(\mathbf{x_1}|\mathbf{x_0})\dots q(\mathbf{x_T}|\mathbf{x_{T-1}})\\
               &= \prod^T_{t=1}q(\mathbf{x_t}|\mathbf{x_{t-1}})
\end{align*}


<!-- Reference the reparameterization trick from lilianweng? -->
<!-- mention how using a predictable noise schedule mitigates this-->

Additionally, we can improve the forward process further, allowing us to sample a noisy image $\mathbf{x_t}$ at any particular time $t$. First, we let $\alpha_{t} = 1 - \beta_t$, and we also define $\bar{\alpha_t} = \prod\nolimits_{i=1}^t \alpha_i$. Now, we rewrite:

$$q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t};\sqrt{\alpha_t}\mathbf{x_{t-1}},(1 - \alpha_t)\mathbf{I})$$

using the reparameterization trick for Gaussian distribution $\mathcal{N}(\mathbf{x}; \mathbf{\mu},\sigma^2\mathbf{I})$,$\mathbf{x} = \mathbf{\mu} + \sigma\mathbf{\epsilon}$, where $\mathbf{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$. This gives us:


\begin{align*}
\mathbf{x_t} &= \sqrt{\alpha_t} \mathbf{x_{t-1}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
\mathbf{x_{t-1}} &= \sqrt{\alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\mathbf{\epsilon_{t-2}} && \text{(Repeat for $\mathbf{x_{t-1})}$)}
\end{align*}


Now, we combine the above equations:


\begin{align*}
\mathbf{x_t} &= \sqrt{\alpha_t} \mathbf{x_{t-1}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
             &= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\mathbf{\epsilon_{t-2}}) + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}} \\
             &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{\alpha_t(1-\alpha_{t-1})}\mathbf{\epsilon_{t-2}} + \sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}}
\end{align*}


Note that we can reverse the reparameterization trick separately on $\sqrt{\alpha_t(1-\alpha_{t-1})}\mathbf{\epsilon_{t-2}}$ and $\sqrt{1-\alpha_t}\mathbf{\epsilon_{t-1}}$, considering them to be samples from $\mathcal{N}(\mathbf{0},\alpha_t(1-\alpha_{t-1})\mathbf{I})$ and from $\mathcal{N}(\mathbf{0},(1-\alpha_t)\mathbf{I})$ respectively. Thus, we can combine these Gaussian distributions with different variance as $\mathcal{N}(\mathbf{0}, ((1-\alpha_t) + \alpha_t(1-\alpha_{t-1}))\mathbf{I}) = \mathcal{N}(\mathbf{0}, (1-\alpha_{t} \alpha_{t-1})\mathbf{I})$. Once again reparameterizing this new distribution:


\begin{align*}
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t} \alpha_{t-1}}\mathbf{\epsilon} && \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
\end{align*}


This process can be continued all the way to $\mathbf{x_0}$ resulting in:


\begin{align*}
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1-\alpha_{t} \alpha_{t-1}}\mathbf{\epsilon} \\
             &\vdots \\
\mathbf{x_t} &= \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_{1}}\mathbf{x_0} + \sqrt{1-\alpha_t \alpha_{t-1} \dots \alpha_{1}}\mathbf{\epsilon} \\
             &= \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}
\end{align*}


Since $\mathbf{x_t} = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}$, we can once more reverse the reparamaterization process to achieve:

$$q(\mathbf{x_t} | \mathbf{x_0}) = \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I})$$

It is clear that we can quickly sample a noisy image $\mathbf{x_t}$ at any timestep t. Given that we are using a conditional diffusion model, this allows us to randomly sample a timestep during training and quickly compute values as to speed up the process. Further, as we are iteravitely adding noise to the original image, we do not need to condition the forward process on our original greyscale.

<div id="Reverse_Diffusion_Theory"></div>


### The Reverse Process

To reverse the above noising process (i.e. $q(\mathbf{x_{t-1}}|\mathbf{x_t})$ ), we would need to know the entire dataset of possible noised images, which is essentially impossible. Thus, we seek to learn an _approximation_ of this reverse process. We will call this approximation $p_\theta$. Note that since we are using a conditional diffusion model, we make use of our $\mathbf{z}$ from above to condition the reverse process.

As we are describing the same distribution in the forward process, but in the opposite direction, we can use the same Markovian property to write:

<!-- Also, note that we can consider x_T to be "x_0" and traverse the process normally, just subbing back in our renamings -->

\begin{align*}
p_{\theta}(\mathbf{x_{0:T}}|\mathbf{z}) &= p_{\theta}(\mathbf{x_{T}}, \dots, \mathbf{x_{0}}|\mathbf{z})\\
                             &= p(\mathbf{x_{T}}) p_{\theta}(\mathbf{x_{T-1}} | \mathbf{x_{T}},\mathbf{z}) \dots p_{\theta}(\mathbf{x_0} | \mathbf{x_1},\mathbf{z}) && \text{(Markov Property)} \\
                             &= p_{\theta}(\mathbf{x_T})\prod_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t},\mathbf{z}) \\
\end{align*}


Also, we make mention here that conditioning our $\mathbf{\mu_{\theta}}(\mathbf{x_t}, \mathbf{z}, t)$ on $\mathbf{x_0}$ in:

$$p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{z}) = \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\mu_{\theta}}(\mathbf{x_t},\mathbf{z}, t), \beta_t \mathbf{I})$$

,allows us to work with the reverse probabilities far more easily, as conditioning on $\mathbf{x_0}$ provides us the ability to treat this distribution as the forward process ($\mathbf{x_0}$ preceeds any $\mathbf{x_t}$, and we have already discussed that $\mathbf{x_t}$ is a noised $\mathbf{x_0}$) ([2]). We will rewrite this as:

$$q(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{x_0}) = \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0}), \tilde{\beta_t}\mathbf{I})$$

Here, $\mathbf{\tilde{\mu_t}}$ and $\tilde{\beta_t}$ are the theoretical values of mean and variance. This is because we are using the forward process, wherein we are aware of the mean and variance at each step.

Before proceeding into our discussion on the loss function, we would like to make note of the absense of $\mathbf{z}$. The additional condition on $p_{\theta}$ has no (noticeable) mathematical change on the derivation. For the sake of simplicity, we refrain from conditioning on $\mathbf{z}$ in the following section.

<div id="Loss_Function"></div>

## Loss Function

Next, we seek to optimize the negative log-likelihood of $p_{\theta}(\mathbf{x_{0}})$, which is the probability of generating images coming from the original data distribution $\mathbf{x_0} \sim q(\mathbf{x} | \mathbf{z})$. We are, in essence, using a Variational Auto Encoder, wherein we want our 'probabilistic decoder' $p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_{t}})$ to closely approximate our 'probabilistic encoder' $q(\mathbf{x_{t-1}} | \mathbf{x_{t}})$ ([6]). To accomplish this, we need the ability to compare these two distributions. Thus, we use Kullback-Leibler (KL) divergence to achieve this comparison. The more the distributions are equivalent, the closer the KL divergence is to 0. It follows then that we want to minimize our KL divergence. However, we also should maximize the likelihood that we generate real samples, or $p_{\theta}(\mathbf{x_0})$. Conviently, we can employ Variational Lower Bounds (VLB)/Evidence Lower Bounds (ELBO) to achieve this concurrently ([7], [8]). We will refer to Variational Lower Bounds as VLB for the remainder of this paper. Since we want to minimize the dissimilarities between the distributions when traversing consecutive steps, we similarly want to minimize the difference between all of the steps ($\mathbf{x_{1:T}}$) which are noised samples, given the initial sample $\mathbf{x_0}$. Thus, we can write this KL divergence as $D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}}))$.

To derive the VLB, we expand the above KL divergence:

\begin{align*}
D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})) &= \mathbb{E}_{\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})}\Big )} d\mathbf{x_{1:T}} \\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) p_{\theta}(\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big )} d\mathbf{x_{1:T}} && \text{(Bayes' Rule)}\\
    &= \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) \Big [ \log{p_{\theta}(\mathbf{x_{0}})} + \log{\Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big )} \Big ] d\mathbf{x_{1:T}} \\
    &= \log{p_{\theta}(\mathbf{x_{0}})} \int q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) d\mathbf{x_{1:T}} + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big) d\mathbf{x_{1:T}}\\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}}, \mathbf{x_{0}})}\Big) d\mathbf{x_{1:T}} && \text{$\Big( \int f(x)dx = 1\Big)$}\\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) p_{\theta}(\mathbf{x_{1:T}})}\Big) d\mathbf{x_{1:T}} && \text{(Bayes' Rule)}\\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \Big [ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big ] d\mathbf{x_{1:T}} \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \int q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}}) \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})\log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) d\mathbf{x_{1:T}} \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big [ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) - \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big ] \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big [ \log \Big( \frac{q(\mathbf{x_{1:T}} \vert \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}})}\Big) \Big ] - \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big [ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big ] \\
    &= \log{p_{\theta}(\mathbf{x_{0}})}  + D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}})) - \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big [ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big ] \\
\end{align*}


If we rearrange this, we get:

$$ \log{p_{\theta}(\mathbf{x_{0}})} - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})) = \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log p_{\theta}(\mathbf{x_{0}} | \mathbf{x_{1:T}}) \Big ] - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}}))$$

The left hand side of this expression yields the desirable relationship with $\log p_{\theta}(\mathbf{x_{0}})$, and thus is our VLB, which we will call $-L_{\text{VLB}}$. Noting that KL divergence is always non-negative:

$$\log{p_{\theta}(\mathbf{x_{0}})} - D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})) \leq \log p_{\theta}(\mathbf{x_{0}})$$

As we seek to maximize the log-likelihood, and acknowledging that all maximizations can be expressed in terms of minimization, we once more rewrite the above expression in terms of said minimization:

$$-\log p_{\theta}(\mathbf{x_{0}}) \leq D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})) - \log{p_{\theta}(\mathbf{x_{0}})} $$

We may further simplify this VLB:

\begin{align*}
L_{\text{VLB}} &= D_{\text{KL}}(q(\mathbf{x_{1:T}} | \mathbf{x_{0}}) || p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})) - \log{p_{\theta}(\mathbf{x_{0}})} \\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log \Big (\frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big) \Big ] - \log{p_{\theta}(\mathbf{x_{0}})} \\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log \Big (\frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})p_{\theta}(\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{1:T}}, \mathbf{x_{0}})} \Big ) \Big ] - \log{p_{\theta}(\mathbf{x_{0}})} && \text{(Bayes' Rule)}\\
               &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log \Big (\frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}}, \mathbf{x_1}, \dots, \mathbf{x_{T}})} \Big ) + \log p_{\theta}(\mathbf{x_{0}}) \Big ] - \log{p_{\theta}(\mathbf{x_{0}})} \\
              &= \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log \Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0:T}})} \Big) \Big ] +  \mathbb{E}_ {\mathbf{x_{1:T}} \sim q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} [ \log p_{\theta}(\mathbf{x_{0}}) ] - \log{p_{\theta}(\mathbf{x_{0}})} \\
              &= \mathbb{E}_ {q(\mathbf{x_{1:T}} | \mathbf{x_{0}})} \Big[ \log \Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0:T}})} \big ) \Big ] + \log p_{\theta}(\mathbf{x_{0}}) - \log{p_{\theta}(\mathbf{x_{0}})} \\
              &= \mathbb{E}_ {q(\mathbf{x_{0:T}} | \mathbf{x_{0}})} \Big[ \log \Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0:T}})} \Big) \Big ]\\
\end{align*}

However, we cannot compute this simplified VLB, as the denominator would require us to already know the reverse conditionals. Thus, we again rewrite our VLB as:

\begin{align*}
L_{\text{VLB}} &= \mathbb{E}_ {q} \Big[ \log \Big ( \frac{q(\mathbf{x_{1:T}} | \mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0:T}})} \Big) \Big ] \\
               &= \mathbb{E}_ {q} \Big[ \log \Big ( \frac{\prod\nolimits^T_{t=1}q(\mathbf{x_t}|\mathbf{x_{t-1}})}{p_{\theta}(\mathbf{x_T})\prod\nolimits_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) \Big ] \\
               &= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \log \Big ( \frac{\prod\nolimits^T_{t=1}q(\mathbf{x_t}|\mathbf{x_{t-1}})}{\prod\nolimits_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) \Big ] \\
               &= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=1}^{T} \log \Big ( \frac{q(\mathbf{x_t}|\mathbf{x_{t-1}})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) \Big ] \\
              &= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_t}|\mathbf{x_{t-1}})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ] \\
              &= (**)
\end{align*}

Now, we use Bayes' Rule to expand $q(\mathbf{x_t}|\mathbf{x_{t-1}})$:

\begin{align*}
q(\mathbf{x_t}|\mathbf{x_{t-1}}) &= q(\mathbf{x_t}|\mathbf{x_{t-1}}, \mathbf{x_0}) && \text{($q(\mathbf{x_{t-1}}) = q(\mathbf{x_{t-1}}, \mathbf{x_0})$)} \\
                                 &= \frac{q(\mathbf{x_t},\mathbf{x_{t-1}}, \mathbf{x_0})}{q(\mathbf{x_{t-1}}, \mathbf{x_0})} && \text{(Bayes' Rule)}\\
                                 &= \frac{q(\mathbf{x_{t-1}},\mathbf{x_{t}}, \mathbf{x_0})}{q(\mathbf{x_{t-1}}, \mathbf{x_0})}\\
                                 &= \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})q(\mathbf{x_{t}}, \mathbf{x_0})}{q(\mathbf{x_{t-1}}, \mathbf{x_0})} && \text{(Bayes' Rule)}\\
                                 &= \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})q(\mathbf{x_{t}}| \mathbf{x_0})q(\mathbf{x_0})}{q(\mathbf{x_{t-1}}| \mathbf{x_0})q(\mathbf{x_0})} && \text{(Bayes' Rule)}\\
                                 &= \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})q(\mathbf{x_{t}}| \mathbf{x_0})}{q(\mathbf{x_{t-1}}| \mathbf{x_0})} && \text{(Bayes' Rule)}\\
\end{align*}

Returning to our previous system of equivalences:

\begin{align*}
(**) &= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \cdot \frac{q(\mathbf{x_{t}}| \mathbf{x_0})}{q(\mathbf{x_{t-1}}| \mathbf{x_0})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ] && \text{(Via the above substitution)}\\
       &= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) + \sum\nolimits_{t=2}^{T} \log \Big( \frac{q(\mathbf{x_{t}}| \mathbf{x_0})}{q(\mathbf{x_{t-1}}| \mathbf{x_0})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ]\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log \Big( \prod \nolimits_{t=2}^{T} \frac{q(\mathbf{x_{t}}| \mathbf{x_0})}{q(\mathbf{x_{t-1}}| \mathbf{x_0})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ]\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log \Big(  \frac{q(\mathbf{x_{2}}| \mathbf{x_0}) \dots q(\mathbf{x_{T}}| \mathbf{x_0})}{q(\mathbf{x_{1}}| \mathbf{x_0}) \cdots q(\mathbf{x_{T-1}}| \mathbf{x_0})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ] && \text{(Telescoping product)}\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log \Big(  \frac{q(\mathbf{x_{T}}| \mathbf{x_0})}{q(\mathbf{x_{1}}| \mathbf{x_0})} \Big) + \log \Big( \frac{q(\mathbf{x_1}|\mathbf{x_{0}})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big)\Big ]\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log \Big(  \frac{q(\mathbf{x_{T}}| \mathbf{x_0})q(\mathbf{x_1}|\mathbf{x_{0}})}{q(\mathbf{x_{1}}| \mathbf{x_0})p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})} \Big) \Big ]\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log \Big(  \frac{q(\mathbf{x_{T}}| \mathbf{x_0})}{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})} \Big) \Big ]\\
&= \mathbb{E}_ {q} \Big[ -\log{p_{\theta}(\mathbf{x_T})} + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) +  \log{q(\mathbf{x_{T}}| \mathbf{x_0})}-\log{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big ]\\
&= \mathbb{E}_ {q} \Big[ \log \Big( \frac{q(\mathbf{x_{T}}| \mathbf{x_0})}{p_{\theta}(\mathbf{x_T})} \Big) + \sum\nolimits_{t=2}^{T} \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big)-\log{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big ]\\
&= \mathbb{E}_ {q} \Big[ \mathbb{E}_ {q} \Big[ \log \Big( \frac{q(\mathbf{x_{T}}| \mathbf{x_0})}{p_{\theta}(\mathbf{x_T})} \Big) \Big] + \sum\nolimits_{t=2}^{T} \mathbb{E}_ {q} \Big[ \log \Big ( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})} \Big) \Big ]-\log{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big ]\\
&= \mathbb{E}_ {q} \Big[ D_\text{KL} (q(\mathbf{x_{T}}| \mathbf{x_0}) || p_{\theta}(\mathbf{x_T})) + \sum\nolimits_{t=2}^{T} D_\text{KL} ( q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})||p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})) -\log{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}\Big ]\\
\end{align*}


Now, to make sense of this VLB, we label the components of the above expression as 'loss terms' $L_{i}$ where $i = 1, \dots, T$. We have three distinguishable cases ([9]):

\begin{align*}
L_T &= D_\text{KL} (q(\mathbf{x_{T}}| \mathbf{x_0}) || p_{\theta}(\mathbf{x_T}))\\
L_{t-1} &= D_\text{KL} ( q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})||p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t}))\\
L_0 &= -\log{p_{\theta}(\mathbf{x_{0}} | \mathbf{x_1})}
\end{align*}


In the case of $L_T$, we are using a pre-defined 'noise schedule' and $q$ has no learning parameters.Thus $L_T$ should be considered constant (Recalling that $p_{\theta}(\mathbf{x_T})$ is generated Gaussian noise as well). For $L_{0}$, we might use an additional decoder to model it, and, for lack of time, we will leave this as a constant $C_0$.

As for $L_{t-1}$, we can easily rearrange the above form to achieve an equivalent result for $L_{t}$, where $1 \leq t \leq T-1$:

$$L_{t} = D_\text{KL} ( q(\mathbf{x_{t}}| \mathbf{x_{t+1}}, \mathbf{x_0})||p_{\theta}(\mathbf{x_{t}} | \mathbf{x_{t+1}}))$$

However, as we seek to optimize our VLB, we will manipulate $L_{t}$ in order to find an optimal/viable cost function. To do this, we want some way to measure the 'loss' between our predicted noisy images, and the original noisy images themselves. Thus, we manipulate the above loss component to find a usable comparison between the respective distributions. Starting with $L_{t-1}$:

\begin{align*}
L_{t-1} &=  D_\text{KL} ( q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})||p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t})) \\
        &= \mathbb{E}_ {\mathbf{x_{t-1}} \sim q} \Big[ \log \Big( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_{t}}))} \Big) \Big ] \\
         &= \mathbb{E}_ {q} \Big[ \log \Big( \frac{q(\mathbf{x_{t-1}}| \mathbf{x_{t}}, \mathbf{x_0})}{p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_{t}}))} \Big) \Big ] \\
\end{align*}

Recall from above that:

\begin{align*}
q(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{x_0}) &= \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0}), \tilde{\beta_t}\mathbf{I}) \\
p_{\theta}(\mathbf{x_{t-1}} | \mathbf{x_t}) &= \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\mu_{\theta}}(\mathbf{x_t}, t), \beta_t \mathbf{I}) \\
\end{align*}


This allows us to rewrite the expression as:

$$L_{t-1} = \mathbb{E}_ {q} \Big[ \log (\mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0}), \tilde{\beta_t}\mathbf{I})) - \log ( \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\mu_{\theta}}(\mathbf{x_t}, t), \beta_t \mathbf{I}))  \Big ]$$


Moving forward, we make use of the assumption that we have a pre-defined noise scheduler (i.e. $\tilde{\beta_t} = \beta_t$ at any timestep t). Additionally, recall that the probability density function of a Gaussian distribution $\mathcal{N}(x;\mu,\sigma^2)$ is:

$$f(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\Big({-\frac{1}{2}\Big({\frac{x-\mu}{\sigma}}\Big)^2}\Big)$$

Again, we manipulate $L_{t-1}$

\begin{align*}
L_{t-1} &= \mathbb{E}_ {q} \Big[ \log \Big( \frac{1}{\sqrt{2\pi \tilde{\beta_t}}}\exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\tilde{\mu_t}}}{\sqrt{\tilde{\beta_t}}}}\Big)^2}\Big)\Big) - \log \Big( \frac{1}{\sqrt{2\pi \beta_t}}\exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\mu_\theta}}{\sqrt{\beta_t}}}\Big)^2}\Big)\Big)  \Big ] \\
        &= \mathbb{E}_ {q} \Big[ \log \Big( \frac{1}{\sqrt{2\pi \tilde{\beta_t}}} \Big ) + \log \Big( \exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\tilde{\mu_t}}}{\sqrt{\tilde{\beta_t}}}}\Big)^2}\Big)\Big) - \Big( \log \Big( \frac{1}{\sqrt{2\pi \beta_t}} \Big) +  \log \Big(\exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\mu_\theta}}{\sqrt{\beta_t}}}\Big)^2}\Big)\Big) \Big) \Big ] \\
        &= \mathbb{E}_ {q} \Big[ \log \Big( \exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\tilde{\mu_t}}}{\sqrt{\tilde{\beta_t}}}}\Big)^2}\Big)\Big) - \log \Big(\exp\Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\mu_\theta}}{\sqrt{\beta_t}}}\Big)^2}\Big)\Big)  \Big ] && \text{($\tilde{\beta_t} = \beta_t$)} \\
&= \mathbb{E}_ {q} \Big[ \Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\tilde{\mu_t}}}{\sqrt{\tilde{\beta_t}}}}\Big)^2}\Big)- \Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\mu_\theta}}{\sqrt{\beta_t}}}\Big)^2}\Big) \Big ]\\
&= \mathbb{E}_ {q} \Big[ \Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\tilde{\mu_t}}}{\sqrt{\tilde{\beta_t}}}}\Big)^2}\Big)- \Big({-\frac{1}{2}\Big({\frac{x_{t-1}-\mathbf{\mu_\theta}}{\sqrt{\beta_t}}}\Big)^2}\Big)\Big ]\\
&= \mathbb{E}_ {q} \Big[ \Big({-\frac{1}{2}\Big({\frac{(x_{t-1}-\mathbf{\tilde{\mu_t}})^2}{\tilde{\beta_t}}}\Big)}\Big)- \Big({-\frac{1}{2}\Big({\frac{(x_{t-1}-\mathbf{\mu_\theta})^2}{\beta_t}}}\Big)\Big)\Big ]\\
&= \mathbb{E}_ {q} \Big[ {-\frac{1}{2\beta_t}\Big({(x_{t-1}-\mathbf{\tilde{\mu_t}})^2}}- (x_{t-1}-\mathbf{\mu_\theta})^2\Big)\Big ] && \text{($\tilde{\beta_t} = \beta_t$)} \\
&= (***)
\end{align*}

Using aforementioned reparameterization of $\mathbf{x_{t-1}} \sim q(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{x_0}) = \mathcal{N}(\mathbf{x_{t-1}}; \mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0}), \tilde{\beta_t}\mathbf{I})$ where $\mathbf{x_{t-1}} = \mathbf{\tilde{\mu_t}} + \sqrt{\tilde{\beta_t}}\mathbf{\epsilon}$ and $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we expand the difference of squares:

\begin{align*}
(x_{t-1}-\mathbf{\tilde{\mu_t}})^2- (x_{t-1}-\mathbf{\mu_\theta})^2 &= ((x_{t-1}-\mathbf{\tilde{\mu_t}})+(x_{t-1}-\mathbf{\mu_\theta}))((x_{t-1}-\mathbf{\tilde{\mu_t}})-(x_{t-1}-\mathbf{\mu_\theta}))\\
                                                                    &= (2x_{t-1}-\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})(\mathbf{\mu_\theta}-\mathbf{\tilde{\mu_t}})\\
                                                                    &= (2(\mathbf{\tilde{\mu_t}} + \sqrt{\tilde{\beta_t}}\mathbf{\epsilon})-\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})(\mathbf{\mu_\theta}-\mathbf{\tilde{\mu_t}})\\
                                                                    &= (2\mathbf{\tilde{\mu_t}} + 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}-\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})(\mathbf{\mu_\theta}-\mathbf{\tilde{\mu_t}})\\
                                                                    &= (2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}+\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})(\mathbf{\mu_\theta}-\mathbf{\tilde{\mu_t}})\\
                                                                    &= 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta}-2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}}+\mathbf{\tilde{\mu_t}}\mathbf{\mu_\theta}-\mathbf{\tilde{\mu_t}}\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta}\mathbf{\mu_\theta}+\mathbf{\mu_\theta}\mathbf{\tilde{\mu_t}}\\
                                                                    &= 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta}-2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}}-(\mathbf{\tilde{\mu_t}}^2-2\mathbf{\tilde{\mu_t}}\mathbf{\mu_\theta}+\mathbf{\mu_\theta}^2)\\
                                                                    &= 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta}-2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}}-(\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2\\
                                                                    &= -\Big( (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 + 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}} - 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta} \Big)\\
\end{align*}


Returning to (***):

\begin{align*}
(***) &= \mathbb{E}_ {q} \Big[ -\frac{1}{2\beta_t}\Big({(x_{t-1}-\mathbf{\tilde{\mu_t}})^2}- (x_{t-1}-\mathbf{\mu_\theta})^2\Big)\Big ]\\
&= \mathbb E_q \Big[ \frac{1}{2\beta_t}\Big( (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 + 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}} - 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta} \Big) \Big ] \\
\end{align*}

We notice that the clever reparameterization of $\mathbf{x_t}$ as $\mathbf{x_t}(\mathbf{x_0}, \mathbf{\epsilon}) = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}$, lends us:

\begin{align*}
&\mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 + 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}} - 2\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta} \Big) \Big ] \\
&= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t} (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 \Big] + \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{\beta_t}\Big(\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}} - \sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta} \Big) \Big ] \\
\end{align*}

Here, recall that $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and so $\mathbb{E}_{\mathbf{x_0}, \mathbf{\epsilon}} [\epsilon] = \mathbf{0}$. On top of this, for independent continuous random variables, $\mathbb{E} [XY] = \mathbb{E}[X]\mathbb{E}[Y]$. So, since $\mathbf{x_0}$ and $\mathbf{\epsilon}$ are independent:

\begin{align}
&\mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t} (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 \Big] + \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{\beta_t}\Big(\sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\tilde{\mu_t}} - \sqrt{\tilde{\beta_t}}\mathbf{\epsilon}\mathbf{\mu_\theta} \Big) \Big ] \\
&= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t} (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 \Big] + \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}}[\mathbf{\epsilon}] E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{\beta_t}\Big( \sqrt{\tilde{\beta_t}}\mathbf{\tilde{\mu_t}} - \sqrt{\tilde{\beta_t}}\mathbf{\mu_\theta} \Big) \Big ] \\
&= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t} (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 \Big]\\
L_{t-1} &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( \|\mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0})-\mathbf{\mu_{\theta}}(\mathbf{x_t}, t)\|^2 \Big) \Big]
\end{align}

$$\mathbb E_q \Big[ \frac{1}{2\beta_t}\Big( (\mathbf{\tilde{\mu_t}}-\mathbf{\mu_\theta})^2 \Big) \Big] = \mathbb E_q \Big[ \frac{1}{2\beta_t}\Big( \|\mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0})-\mathbf{\mu_{\theta}}(\mathbf{x_t}, t)\|^2 \Big) \Big]$$

Where $\| \cdot \|$ denotes the one-norm.

Thus, we can see that we have acquired a desired comparison between the two distributions. However, this gives that we are trying to train $\mathbf{\mu_{\theta}}$ to predict $\mathbf{\tilde{\mu_t}}$, which is intuitively difficult to justify. To find a more clear loss function, we look towards parameterizing our means. Starting with $\mathbf{\tilde{\mu_t}}$:

\begin{align*}
q(\mathbf{x_{t-1}} | \mathbf{x_t}, \mathbf{x_0}) &= \frac{q(\mathbf{x_{t-1}}, \mathbf{x_t}, \mathbf{x_0})}{q(\mathbf{x_t}, \mathbf{x_0})} && \text{(Bayes' Rule)} \\
                                                 &= \frac{q(\mathbf{x_{t}} | \mathbf{x_{t-1}}, \mathbf{x_0})q(\mathbf{x_{t-1}}, \mathbf{x_0})}{q(\mathbf{x_t}| \mathbf{x_0})q(\mathbf{x_0})} && \text{(Bayes' Rule)} \\
                                                 &= \frac{q(\mathbf{x_{t}} | \mathbf{x_{t-1}}, \mathbf{x_0})q(\mathbf{x_{t-1}}| \mathbf{x_0})q(\mathbf{x_0})}{q(\mathbf{x_t}| \mathbf{x_0})q(\mathbf{x_0})} && \text{(Bayes' Rule)} \\
                                                 &= \frac{q(\mathbf{x_{t}} | \mathbf{x_{t-1}}, \mathbf{x_0})q(\mathbf{x_{t-1}}| \mathbf{x_0})}{q(\mathbf{x_t}| \mathbf{x_0})} \\
                                                 &= (\text{****})
\end{align*}



Recall $q(\mathbf{x_{t}} | \mathbf{x_{t-1}}, \mathbf{x_0}) = q(\mathbf{x_{t}} | \mathbf{x_{t-1}})$.  Addressing the above distributions individually:

\begin{align*}
q(\mathbf{x_{t}} | \mathbf{x_{t-1}}) &= \mathcal{N}(\mathbf{x_t};\sqrt{\alpha_t}\mathbf{x_{t-1}},(1 - \alpha_t)\mathbf{I})\\
q(\mathbf{x_{t-1}}| \mathbf{x_0}) &= \mathcal{N}(\mathbf{x_{t-1}};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}, (1 - \bar{\alpha}_{t-1})\mathbf{I})\\
q(\mathbf{x_t}| \mathbf{x_0}) &= \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I})\\
\end{align*}

Again using the probability density function of a Gaussian distribution:

\begin{align*}
(\text{****}) &= \frac{\frac{1}{\sqrt{2\pi (1 - \alpha_t)}}\exp\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t}}-\sqrt{\alpha_t}\mathbf{x_{t-1}}}{\sqrt{(1 - \alpha_t)}}}\Big)^2}\Big)\frac{1}{\sqrt{2\pi (1 - \bar{\alpha}_t})}\exp\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t-1}}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{\sqrt{(1 - \bar{\alpha}_{t-1})}}}\Big)^2}\Big)}{\frac{1}{\sqrt{2\pi (1 - \bar{\alpha}_t})}\exp\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t}}-\sqrt{\bar{\alpha_t}}\mathbf{x_0}}{\sqrt{(1 - \bar{\alpha_t})}}}\Big)^2}\Big)}\\
              &\propto \exp\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t}}-\sqrt{\alpha_t}\mathbf{x_{t-1}}}{\sqrt{\beta_t}}}\Big)^2}\Big)+\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t-1}}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{\sqrt{(1 - \bar{\alpha}_{t-1})}}}\Big)^2}\Big)-\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t}}-\sqrt{\bar{\alpha_t}}\mathbf{x_0}}{\sqrt{(1 - \bar{\alpha_t})}}}\Big)^2}\Big)\\
              &= \exp\Big({-\frac{1}{2}\Big({\frac{(\mathbf{x_{t}}-\sqrt{\alpha_t}\mathbf{x_{t-1}})^2}{\beta_t}}}+{\frac{(\mathbf{x_{t-1}}-\sqrt{\bar{\alpha_{t-1}}}\mathbf{x_0})^2}{(1 - \bar{\alpha}_{t-1})}}-{\frac{(\mathbf{x_{t}}-\sqrt{\bar{\alpha_t}}\mathbf{x_0})^2}{(1 - \bar{\alpha_t})}}\Big)\Big)\\
              &= \exp\Big({-\frac{1}{2}\Big({\frac{\mathbf{x_{t}}^2-2\mathbf{x_{t}}\sqrt{\alpha_t}\mathbf{x_{t-1}} + \alpha_t\mathbf{x_{t-1}}^2}{\beta_t}}}+{\frac{\mathbf{x_{t-1}}^2-2\mathbf{x_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0} + \bar{\alpha}_{t-1}\mathbf{x_0}^2}{(1 - \bar{\alpha}_{t-1})}}-{\frac{(\mathbf{x_{t}}-\sqrt{\bar{\alpha_t}}\mathbf{x_0})^2}{(1 - \bar{\alpha_t})}}\Big)\Big)\\
              &= \exp\Big({-\frac{1}{2}\Big({-\frac{2\mathbf{x_{t}}\sqrt{\alpha_t}\mathbf{x_{t-1}}}{\beta_t} -\frac{2\mathbf{x_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})} + \frac{\alpha_t\mathbf{x_{t-1}}^2}{\beta_t}}}+{\frac{\mathbf{x_{t-1}}^2}{(1 - \bar{\alpha}_{t-1})} + \frac{\bar{\alpha}_{t-1}\mathbf{x_0}^2}{(1 - \bar{\alpha}_{t-1})}}+\frac{\mathbf{x_{t}}^2}{\beta_t}-{\frac{(\mathbf{x_{t}}-\sqrt{\bar{\alpha_t}}\mathbf{x_0})^2}{(1 - \bar{\alpha_t})}}\Big)\Big)\\
              &= \exp\Big({-\frac{1}{2}\Big(\frac{\alpha_t\mathbf{x_{t-1}}^2}{\beta_t}}+\frac{\mathbf{x_{t-1}}^2}{(1 - \bar{\alpha}_{t-1})} {-\frac{2\mathbf{x_{t}}\sqrt{\alpha_t}\mathbf{x_{t-1}}}{\beta_t} -\frac{2\mathbf{x_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})} + C(\mathbf{x_t}, \mathbf{x_0})}\Big)\Big)\\
              &= \exp\Big({-\frac{1}{2}\Big(\Big(\frac{\alpha_t}{\beta_t}}+\frac{1}{(1 - \bar{\alpha}_{t-1})} \Big)\mathbf{x_{t-1}}^2 {-2\Big(\frac{\mathbf{x_{t}}\sqrt{\alpha_t}}{\beta_t} +\frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})} \Big)\mathbf{x_{t-1}} + C(\mathbf{x_t}, \mathbf{x_0})}\Big)\Big)\\
\end{align*}


Here $C(\mathbf{x_t}, \mathbf{x_0})$ is function which does not involve $\mathbf{x_{t-1}}$. The reason we choose not to deal with this function involves, again, the probability density function of a normal distribution:

\begin{align*}
f(x) &= \frac{1}{\sqrt{2\pi \sigma^2}}\exp\Big({-\frac{1}{2}\Big({\frac{x-\mu}{\sigma}}\Big)^2}\Big) \\
     &\propto \exp\Big({-\frac{1}{2}\Big({\frac{x-\mu}{\sigma}}\Big)^2}\Big) \\
     &= \exp\Big({-\frac{1}{2}\Big({\frac{x^2-2\mu x + \mu^2}{\sigma^2}}\Big)}\Big) \\
     &= \exp\Big({-\frac{1}{2}\Big({\frac{1}{\sigma^2}x^2-\frac{2\mu}{\sigma^2}x + \frac{\mu^2}{\sigma^2}}\Big)}\Big) \\
\end{align*}

Where $\frac{\mu^2}{\sigma^2}$ does not involve $x$.

To continue, we can use the fact that two polynomials are equals if their coefficients are equal. Returning to the above expression, we can correspond $\Big(\frac{\alpha_t}{\beta_t}+\frac{1}{(1 - \bar{\alpha}_{t-1})} \Big)\mathbf{x_{t-1}}^2$ to $\frac{1}{\sigma^2}x^2$. Similarly, $2\Big(\frac{\mathbf{x_{t}}\sqrt{\alpha_t}}{\beta_t} +\frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})} \Big)\mathbf{x_{t-1}}$ corresponds to $\frac{2\mu}{\sigma^2}x$. With this information, we parameterize $\tilde{\mu}_{t}(\mathbf{x_t}, \mathbf{x_0})$:

\begin{align*}
\tilde{\beta}_{t} &= \Big(\frac{\alpha_t}{\beta_t}+\frac{1}{(1 - \bar{\alpha}_{t-1})} \Big)^{-1} \\
                  &= \Big(\frac{\alpha_t(1 - \bar{\alpha}_{t-1})+\beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})} \Big)^{-1} \\
                  &= \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{\alpha_t(1 - \bar{\alpha}_{t-1})+\beta_t} \\
                  &= \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{\alpha_t - \alpha_t\bar{\alpha}_{t-1}+(1-\alpha_t)} && \text{($\beta_t = 1 - \alpha_t$)}\\
                  &= \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1- \bar{\alpha}_{t}} && \text{($\bar{\alpha_t} = \prod\nolimits_{i=1}^t \alpha_i = \alpha_t\bar{\alpha}_{t-1}$)}\\
\end{align*}

\begin{align*}
\tilde{\mu}_{t}(\mathbf{x_t}, \mathbf{x_0}) &= \Big(\frac{\mathbf{x_{t}}\sqrt{\alpha_t}}{\beta_t} +\frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})} \Big) \Big( \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1- \bar{\alpha}_{t}} \Big) \\
                                            &= \frac{\mathbf{x_{t}}\sqrt{\alpha_t}}{\beta_t}\Big( \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1- \bar{\alpha}_{t}} \Big)+\frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x_0}}{(1 - \bar{\alpha}_{t-1})}\Big( \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1- \bar{\alpha}_{t}} \Big) \\
                                            &= \mathbf{x_{t}}\Big( \frac{\sqrt{\alpha_t}\beta_t(1 - \bar{\alpha}_{t-1})}{\beta_t(1- \bar{\alpha}_{t})}\Big)+\mathbf{x_0}\Big( \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_{t-1})(1- \bar{\alpha}_{t})}\Big) \\
                                            &= \mathbf{x_{t}}\Big( \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{(1- \bar{\alpha}_{t})}\Big)+\mathbf{x_0}\Big( \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1- \bar{\alpha}_{t})}\Big) \\
\end{align*}

We can actually perform a substitution here involving the reparameterization of samples from normal distribution from above: $\mathcal{N}(\mathbf{x}; \mathbf{\mu}, \sigma^2\mathbf{I})$, $\mathbf{x} = \mathbf{\mu} + \sigma\mathbf{\epsilon}$, where $\mathbf{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$.

\begin{align*}
q(\mathbf{x_t} | \mathbf{x_0}) &= \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I}) \\
\mathbf{x_t} &= \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t} \\
\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t} &= \sqrt{\bar{\alpha_t}}\mathbf{x_0}\\
(\sqrt{\bar{\alpha_t}})^{-1}(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t}) &= \mathbf{x_0}\\
\end{align*}

Returning to $\tilde{\mu}_{t}(\mathbf{x_t}, \mathbf{x_0})$:

\begin{align*}
\tilde{\mu}_{t}(\mathbf{x_t}, \mathbf{x_0}) &= \mathbf{x_{t}}\Big( \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{(1- \bar{\alpha}_{t})}\Big)+\mathbf{x_0}\Big( \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1- \bar{\alpha}_{t})}\Big) \\
                                            &= \mathbf{x_{t}}\Big( \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{(1- \bar{\alpha}_{t})}\Big)+(\sqrt{\bar{\alpha_t}})^{-1}(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t})\Big( \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1- \bar{\alpha}_{t})}\Big) \\
                                            &= \frac{\mathbf{x_{t}}\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})+(\sqrt{\bar{\alpha_t}})^{-1}(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t})\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1- \bar{\alpha}_{t})} \\
                                            &= \frac{\mathbf{x_{t}}\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})(\sqrt{\bar{\alpha_t}})+(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t})\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(\sqrt{\bar{\alpha_t}})(1- \bar{\alpha}_{t})} \\
                                            &= \frac{\mathbf{x_{t}}\sqrt{\alpha_t}\sqrt{\bar{\alpha_t}} - \mathbf{x_{t}}\sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\sqrt{\bar{\alpha_t}}+(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t})\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(\sqrt{\bar{\alpha_t}})(1- \bar{\alpha}_{t})} && \text{$\bar{\alpha}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\sqrt{\bar{\alpha}_{t-1}}$}\\
                                            &= \frac{\mathbf{x_{t}}\sqrt{\alpha_t}\sqrt{\alpha_{t}}\sqrt{\bar{\alpha}_{t-1}} - \mathbf{x_{t}}\sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\sqrt{\bar{\alpha_t}}+(\mathbf{x_t} - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t})\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(\sqrt{\bar{\alpha}_{t}}\sqrt{\bar{\alpha}_{t-1}})(1- \bar{\alpha}_{t})} && \text{($\alpha_{t} = \alpha_{t}\bar{\alpha}_{t-1}$)}\\
                                            &= \frac{\mathbf{x_{t}}\alpha_t - \mathbf{x_{t}}\sqrt{\bar{\alpha}_{t}}\sqrt{\bar{\alpha_t}}+\mathbf{x_t}\beta_t - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t}\beta_t}{\sqrt{\alpha_{t}}(1- \bar{\alpha}_{t})}\\
                                            &= \frac{\mathbf{x_{t}}\alpha_t - \mathbf{x_{t}}\bar{\alpha}_{t}+\mathbf{x_t}\beta_t - \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t}\beta_t}{\sqrt{\alpha_{t}}(1- \bar{\alpha}_{t})} \\
                                            &= \frac{\mathbf{x_{t}}\alpha_t - \mathbf{x_{t}}\bar{\alpha}_{t}+\mathbf{x_t}(1 - \alpha_t)}{\sqrt{\alpha_{t}}(1- \bar{\alpha}_{t})} - \frac{\mathbf{\epsilon_t}(1 - \alpha_t)}{\sqrt{\alpha_{t}}\sqrt{1 - \bar{\alpha_t}}} && \text{($ \beta_t = 1 - \alpha_t $)} \\
                                            &= \frac{1}{\sqrt{\alpha_{t}}} \Big( \frac{\mathbf{x_{t}}\alpha_t - \mathbf{x_{t}}\bar{\alpha}_{t}+\mathbf{x_t} -\mathbf{x_t} \alpha_t}{1- \bar{\alpha}_{t}} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big) \\
                                            &= \frac{1}{\sqrt{\alpha_{t}}} \Big( \frac{\mathbf{x_t} - \mathbf{x_{t}}\bar{\alpha}_{t}}{1- \bar{\alpha}_{t}} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big) \\
                                            &= \frac{1}{\sqrt{\alpha_{t}}} \Big( \frac{\mathbf{x_t}(1 - \bar{\alpha}_{t})}{1- \bar{\alpha}_{t}} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big) \\
                                            &= \frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big) \\
\end{align*}

Via a different manipulation of $\mathbf{x_t} = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon_t}$, we see we can write $\mathbf{\epsilon}$ parameterized by $\mathbf{x_t}$ and t. Thus, when $\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t)$, we can perform the above reparameterization to $\mu_{\theta}( \mathbf{x_t}, t)$:

\begin{align*}
\mu_{\theta}( \mathbf{x_t}, t) &=\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t) \Big) \\
\end{align*}

From here, we can return to:

\begin{align*}
L_{t-1} &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( \|\mathbf{\tilde{\mu_t}}(\mathbf{x_t}, \mathbf{x_0})-\mathbf{\mu_{\theta}}(\mathbf{x_t}, t)\|^2 \Big) \Big]
\end{align*}

Now, we may rewrite this using our new reparameterizations of $\tilde{\mu}_{t}$ and $\mathbf{\mu_{\theta}}$:

\begin{align*}
 &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( \|\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big)-\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t) \Big)\|^2 \Big) \Big] \\
 &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( \| -\Big(\frac{1 - \alpha_t}{\sqrt{\alpha_{t}}\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_t} \Big) + \frac{1 - \alpha_t}{\sqrt{\alpha_{t}}\sqrt{1 - \bar{\alpha_t}}}\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t) \|^2 \Big) \Big] \\
 &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{1}{2\beta_t}\Big( \| \Big(\frac{1 - \alpha_t}{\sqrt{\alpha_{t}}\sqrt{1 - \bar{\alpha_t}}}\Big) (\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t)-\mathbf{\epsilon_t})\|^2 \Big) \Big] \\
 &= \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \frac{(1 - \alpha_t)^2}{2\beta_t\alpha_{t}(1 - \bar{\alpha_t})}\| \mathbf{\epsilon_t} -(\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t))\|^2 \Big] \\
\end{align*}

As mentioned in other works on this topic, ([9]), finds that removing the leading weight term and redefining $L_{t-1}$ as:

$$L_{t-1} := \mathbb E_{\mathbf{x_0}, \mathbf{\epsilon}} \Big[ \| \mathbf{\epsilon_t} -(\mathbf{\epsilon_{\theta}}(\mathbf{x_t}, t))\|^2 \Big]$$

, yields better performance during training.

Intuitively, we have turned our model $\mathbf{\epsilon}_{\theta}$ into a noise predictor. Essentially, we are training said model (employing a neural network) to closely approximate/reproduce the level of noise present in an image. For our purposes, the noise is "subtracted" as to reveal a colorized version of the input greyscale image. See the Implementation section for more details.

We simplify our entire Loss of $L = L_{0} + L_t + C_t$ down to:

$$L' = L_t + C $$

where $C = L_0 + C_t$ is a constant which do not depends on $\theta$.

Thus, our loss function is complete.

<div id="Algo_Changes"></div>

## Algorithmic Changes in Conditional Diffusion Model

As mentioned prior to the Loss Function section, we have not conditioned our $p_{\theta}$ on $\mathbf{z}$. This has no effect on the derivation of our loss. However, the algorithms we use during training and inference will differ slightly from a 'standard' diffusion model. This is characterized below:

<img src="assets/Algorithm_1.png" align="center"/>

For line 2, we have no conditioned variables. As our dataset is in the form of pairs (input-output images [10]), we sample this pair to eventually have a point of reference for our model ($\mathbf{x_0}$ is the starting colored image, while $\mathbf{z}$ is the same image greyscaled).

<img src="assets/Algorithm_2.png" align="center"/>

In both algorithms, note that $\mathbf{\epsilon}_\theta$ has an additional parameter $\mathbf{z}$. In essence, this added parameter allows us to both gather more information during the diffusion process, as well as provides a base image to concatenate the reversed noise onto. A more detailed explanation can be found in the Implementation section below.

<div id="Implementation"></div>

## Implementation

```
you may argue to use a build-in implementation if the algorithm involved is
overly complicated but necessary to the task. After solving the task with your own
implementation, you may also, in a separate notebook, use a built-in implementation
and compare performance.
```

As our project is quite ambitious, the most complex part was to understand how diffusion models work. This is why, most of our work and of our research is explained in the theory section. Concerning the implementation, most of the papers we are reffering to, published their code in github ([9],[11]). As exploring and understanding the theory behind diffusion models took us 3 to 4 weeks in total, we have chosen to draw inspiration from these github repository, as well as articles going through these implementations like [the one from Hugging Face](https://huggingface.co/blog/annotated-diffusion). Even though we got inspired from these codes, we can note that some algorithms had to be modified to adapt to conditional diffusion.

As our project requires to use a neural network with an encoder-decoder architecture with complex types of layers, we are using the python library PyTorch. Thus, most of the objects that we are manipulating are not numpy arrays, but PyTorch tensors. These tensors behaves like numpy arrays, but are more convenient to track the operations and therefore compute the gradients required by the backpropagation step. To do so, PyTorch uses automatic differenciation, which consists of building an implicit computational graph when performing mathematical operations. By knowing the derivative of elementary mathematical operations, it is then possible to compute efficiently the gradient of every functions. 

<div id="Diffusion_Basics"></div>

### Diffusion basics

In the theory section, we often make reference to $\bar{\beta}_t$ or $\bar{\alpha}_t$. To compute the noise scheduler, we implemented two different schedules:
- The cosine schedule which is implemented the same way as proposed in [11]:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, f(t) = cos\Big(\frac{t/T + s}{1 + s}  \frac{\pi}{2}\Big)^2$$
$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$
- The linear schedule which is implemented the same way as proposed in [9], with the same constants. Thus, it just a list of evenly spaced numbers starting from 0.0001 to 0.02.

In order to improve the speed of the algorithms, the values of many constants are computed, and stored in the class `ConstantDiffusionTerms`. This consists of lists containing all the values from $t=0$ to $t=T$ of:
$$\beta_t, \alpha_t, \bar{\alpha}_t, \bar{\alpha}_{t-1}, \frac{1}{\sqrt{\alpha_t}}, \sqrt{\bar{\alpha}_t}, \sqrt{1 - \bar{\alpha}_t}, \tilde{\beta}_t, \sqrt{\tilde{\beta}_t}$$

Thanks to the method `extract_t_th_value_of_list`, we can then extract the $t$ element of these lists, with respect to the batch size.

<div id="Forward_Process"></div>

### Forward process

As all the closed forms are computed, it is now easy to implement the forward process, which corresponds to 
$$q(\mathbf{x_t} | \mathbf{x_0}) = \mathcal{N}(\mathbf{x_t};\sqrt{\bar{\alpha_t}}\mathbf{x_0}, (1 - \bar{\alpha_t})\mathbf{I})$$
or using the reparametrization trick
$$\mathbf{x_t} = \sqrt{\bar{\alpha_t}}\mathbf{x_0} + \sqrt{1 - \bar{\alpha_t}}\mathbf{\epsilon}$$

Thus, we only need to compute a random noise $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ of the same size as the image, and compute $\mathbf{x_t}$ thanks to the closed forms computed earlier.

<div id="Neural_Network"></div>

### The neural network

<div id="UNet_Overview"></div>

#### Overview of the U-Net architecture

The most complex part of the implementation, is designing the neural network. This neural network has an encoder-decoder architecture as used in the Variational Auto Encoder. More precisely, the U-Net architecture was used in [9], and offers good results. The same type of architecture was used in [11], [12] and [10]. Here is the idea of this architecture:

<img src="assets/u-net-architecture.png" alt="unet architecture picture" style="height: 400px" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 1</b>: Base U-Net architecture </div>


This architecture first downsamples the input in term of spatial resolution, but with more and channels because of the convolutions (64 -> 128 -> 256 -> 512 -> 1024). It also has a bottleneck in the middle that allow the network to learn the most important information to perform the task. Then, an upsampling is performed in order to output an image of the same shape as the input. Between the layers of identical size of the encoder and decoder, there is also residual connections (or skip connections), which improves the gradient flow.

In recent work on Diffusion models ([9],[11],[12],[10]), it has been shown that the self-attention layers greatly improves the results of diffusion models. In this first implementation, we made the choice of not implementing attention layers. As we don't have the computing power to train a model with hundreds of millions of parameters, we also reduced the size of the network in terms of depth and number of convolution kernels.

In this section, each main blocks are described, in order to present the architecture of our custom U-Net.

<div id="Timestep_Embedding"></div>

#### Timestep embedding

As seen in the theory part, the neural network must also take the timestep $t$ (indication on the noise level) as input. The authors of [9] employed the sinusoidal position embedding to encode $t$. The sinusoidal position embedding has been proposed by the famous Transformer architecture [13]. Sinusoidal positional embedding aims to generate an embedding that take into account the order of the information, using the sin and cos functions. So it takes a integer $t$ as input, and output a vector in $\mathbb{R}^d$, $d$ being the desired embedding dimension.

The sinusoidal positional embedding is defined by a function $f: \mathbb{N} \to \mathbb{R}^d$:

$$f(t)^{(i)} = \begin{cases}
      sin(\omega_k t), & \text{if}\ i=2k \\
      cos(\omega_k t), & \text{if}\ i=2k+1
    \end{cases}$$ 

$$\text{For } 1 \leq i \leq d, \forall k \in \mathbb{N} \text{ and } \omega_k = \frac{1}{1000^{2k/d}}$$
    
This positional encoding forms a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. In the end, we obtain a vector containing pairs of sin and cos for each frequency. This also implies that $d$ must be divisible by 2.

<div id="Conditional_input"></div>

#### Conditional image input

The dataset consists of input-output image pairs $\{z_i,x_i\}^N_{i=1}$ where the samples are sampled from an unknown conditional distribution $p(x|z)$. Here, $x$ is the color image and $z$ is the grayscale image. This conditional input $z$ is given to the denoising model $\epsilon_\theta$, in addition to the noisy target image $x_t$ and the timestep $t$. In practice, to condition the model we followed the work of [10]. The input $z$ is concatenated with $x_t$ along the channel dimension before entering the first layer of the U-Net. 

<div id="Core_Convolution_Block"></div>

#### Core convolution block

The core convolution block is the elementary block of the bigger blocks described below. This convolution block simply apply a 2D convolution with kernels of size 3x3 and a padding of 1. A group normalization is applied to the output of the convolution, and then the SiLU (Sigmoid Linear Unit) activation function is applied.

<div id="Resnet_Block"></div>

#### Resnet Block

In our implementations, we have chosen to use the residual block as employed by [9]. The resnet block take as input an image but also the timestep embedded with sinusoidal embedding. At first, the input image goes through the first core convolution block. At the same time, the embedded timestep is sent through a one layer perceptron that output a vector of the same size as the number of kernels applied in the first core convolution block. Then, the timestep is reshaped in order to be compatible in number of dimensions with the output of the first core convolution block. This allows us to add the output of the MLP, with the output of the first convolution block. The result is then sent through another core convolution block.

The last step consists in adding the residual connection, which corresponds to add the input image of the resnet block, with the last result. If the case where the channel dimension is incompatible between those two, we apply a convolution with a kernel of size 1 in order to make them compatible.

<div id="Sample_Blocks"></div>

#### Up-sample and Down-sample blocks

As we can see in Figure 1, a downsample and an upsample operation is applied after 2 main blocks of convolutions (which are resnet blocks in our case). For the downsample operation, we are using a convolution layer with a kernel of size 4x4, a stride of 2 and a padding of 1. For the upsample operation, we use a transposed convolution layer with a kernel of size 4x4, a stride of 2 and a padding of 1. A transposed convolution generate an output feature map, with has a greater spatial dimension than the input feature map. The transposed convolution broadcasts each input elements via the kernel, which leads to an output that is larger than the input. 

We can express the convolution operation as a matrix multiplication between a sparse matrix containing the information of the kernel $W_{sparse}$, and a column vector which is the flattened (by row) version of the input matrix $X_{flattened}$. This result of this operation gives a column vector $Z_{flattened}$ which can then be reshaped to produce the same result as a classic convolution operation $Z$. Now, if we take an input matrix that has the same shape of $Z$, and perform a matrix multiplication with the transposed sparse kernel matrix $W_{sparse}^T$, we obtain a result that has the same shape as $X_{flattened}$. Then, we just have to reshape it to produce a result of the same shape as $X$. Thus, we performed an upsampling of $Z$. By increasing the stride and the kernel size, we can generate outputs of greater sizes.


<div id="Custom_Architecture"></div>

#### Architecture of our custom U-Net

Now that we defined the main building blocks of our custom U-Net, let's expand its architecture. The following table does not include the MLP of the sinusoidal position embedding as it is done at first only for the timestep $t$. As mentioned previously, our U-net is much smaller than the other u-net employed for state of the art diffusion models. In the end, we get a network having only ~6.9 Millions of parameters.

<div id="Encoder"></div>

##### Encoder

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| Conv2d            	| img_channels          	| 32                     	|
| ResNet block       	| 32                    	| 64                    	|
| ResNet block       	| 64                    	| 64                    	|
| Conv2d (downsample)  	| 64                    	| 64                    	|
| ResNet block       	| 64                    	| 128                    	|
| ResNet block       	| 128                   	| 128                    	|
| Conv2d (downsample) 	| 128                   	| 128                    	|
| ResNet block       	| 128                   	| 256                    	|
| ResNet block       	| 256                   	| 256                    	|

   The encoder downscale the image while adding more and more feature maps.


##### Bottleneck

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| ResNet block       	| 256                    	| 256                    	|
| ResNet block       	| 256                   	| 256                    	|
    
The bottleneck is the part of the network with the lowest image dimension, which compress all the important information of the image.

<div id="Decoder"></div>

##### Decoder

| **Layer**          	| **No Input channels** 	| **No Output channels** 	|
|--------------------	|:---------------------:	|------------------------	|
| ResNet block       	| 256                    	| 128                    	|
| ResNet block       	| 128                    	| 128                    	|
| ConvTranspose2D (up) 	| 128                    	| 128                    	|
| ResNet block       	| 128                    	| 64                    	|
| ResNet block       	| 64                    	| 64                    	|
| ConvTranspose2D (up) 	| 64                    	| 64                    	|
| ResNet block       	| 64                    	| 32                    	|
| ResNet block       	| 32                    	| 32                    	|
| ResNet block       	| 32                    	| 32                    	|
| Conv2d            	| 32                    	| img_channels             	|

The decoder upsample the input features from the bottlenck, in order to output an image of the same size as the input in terms of spatial dimenions, and number of channels.

<div id="Reverse_Process"></div>

### Reverse process

The reverse process (or 'inference' for our purposes) follows exactly from <a href="#Algo_Changes">Algorithm 2</a> of the <a href="#Theory">Theory</a> section above. We begin by sampling a completely noisy image $\mathbf{x_T}$ from a Gaussian distribution $\mathcal{N}(\mathbf{0},\mathbf{I})$. As our goal is to subtract noise from this image until we have the desired $\mathbf{x_0}$, at each timestep $0 \leq t \leq T$, we begin by sampling random noise $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Next, since each step can be seen as calculating $\mathbf{x_{t-1}}$ from $\mathbf{x_t}$, we perform the reparameterization trick to achieve:

$$\mathbf{x_{t-1}} = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta (\mathbf{x_t}, \mathbf{z},t) \Big) + \sqrt{\beta_t}\mathbf{\epsilon}$$

Thanks to our ```ConstantDiffusionTerms``` class, many of these closed forms are already computed.

Thus, we use our model $\mathbf{\epsilon}_{\theta}$ to predict the noise between steps, and feed this predicted noise into the reparameterization of the <a href="#Loss_Function">predicted mean</a>:

$$\mu_{\theta}(\mathbf{x_t}, t) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x_t} - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta (\mathbf{x_t}, \mathbf{z},t) \Big)$$

This gives us $\mathbf{x_{t-1}}$. Repeating this process T times yields us $\mathbf{x_0}$, which is a colorized version of the greyscale input.

<div id="Training"></div>

### Training

To train this model, we look to <a href="#Algo_Changes">Algorithm 1</a>. As diffusion models seek to produce images closely correlated with the data they are trained on, we sample a color-greyscale image pair $(\mathbf{x_0}, \mathbf{z})$ from our data distribution $p(\mathbf{x}, \mathbf{z})$. Then, we sample a random timestep t from a uniform distribution of timesteps $\{ 1, \dots, T \}$, and as discussed in the theory section above, we can use the Markov Property to produce a noisy image $\mathbf{x_{t}}$ at said timestep:

$$\mathbf{x_t} = \sqrt{\bar{\alpha}_t}\mathbf{x_0}+\sqrt{1 - \bar{\alpha}_t}\epsilon$$

As before, we need to sample $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ to perform this. 

Now, we take a gradient descent step using our loss function:

$$\nabla_\theta \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x_0}+\sqrt{1 - \bar{\alpha}_t}\epsilon, \mathbf{z} ,t) \|^2$$

We repeat this until converged. Note, again, that our loss function is essentially the normed-difference between the theoretical noise, and our noise predicting model.

<div id="Results"></div>

## Results

We trained the network for 5 epochs with 2000 images (see `notebook/train.ipynb`) in order to be able to produce results. The results of our project can be found in the notebook in the folder `demo`. First, the forward process is working well. Here is the forward process using the linear schedule on 300 timesteps ($T = 300$):
<img src="assets/linearschedule.png" alt="Linear schedule" style="width: 800px" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 2</b>: Forward process with linear schedule </div>

Now the results of the reverse process are not satisfying. This not suprising as the network has only be trained for 5 epochs. Here, we try to colorize this image:

<img src="assets/grayscale.png" alt="Grayscale image" style="width: 100" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 3</b>: Grayscale image that we seek to colorize </div>


Then, here is 10 evenly separated noisy images from $\mathbf{x_T}$ to $\mathbf{x_0}$, using $T=300$:

<img src="assets/reverse.png" alt=" image" style="width: 800" align="center"/>
<br>
<div style='text-align:center'> <b>Figure 4</b>: Reverse process</div>

We notice that the noisy image is changing across the timesteps. We also notice some sort of structure in the noisy image, which slightly resembles the branch on the right of the image. Finally, this model output a (non accurate) colorized version of the grayscale image.

<div id="Discussions"></div>

## Discussions

As stated previously, we spent most of the time doing research and working on the theory side of conditional diffusion models. Thus, the time left for the implementation part was shorter as expected. It was still important for us to have the whole pipeline working, so the project can be improved by adding features. This is why we trained a small model on a small amount of data. A first limit of this training is that only 2000 images was used. This is mostly due to a lack of RAM from our computers, but we also wanted to use a subset of the dataset to make the training faster. We also did not created a test and validation set, as we knew that results won't be interpretable. At first, we planned to evaluate the model using the FID (Fréchet inception distance) score, which is a metrics used to evaluate generative models. For lack of time, we decided to not implement the `eval` method present in `scripts/eval.py`, and focus on this readme file. Finally, the results we obtained are far from being the colorized input image. As the entire pipeline is already written, the model needs to be improved, and the dataset needs to be better handled. Thereafter, we could expect better results.

We can also note that, diffusion models are a very recent research topic, we are working on research papers from 2 years ago or even from 2022. This makes the project really challenging when it comes to research.

....

<div id="Installation"></div>

## Installation

First clone the repository, and navigate to the colorized folder:

```sh
git clone git@github.com:Clement-W/MAT_180_ML_Projects.git
cd MAT_180_ML_Projects/colorized
```

To install the required libraries, there is 2 solutions. You can either install the requirements directly on your python installation, or install them into a virtual environement.

To install the requirements, you can just do:

```sh
pip3 install -r requirements.txt
```

To use a virtual environement:

* Install virtualenv if you don't already have it

```sh
pip3 install virtualenv
```

* Create a virtual environment

```sh
python3 -m venv colorized-env
```

* Activate the virtual environment

```sh
source colorized-env/bin/activate
```

* Install the required libraries into the environment
```sh
pip3 install -r requirements.txt
```

If you whish to leave the virtual environment, you can just do:
```sh
deactivate
```

<div id="Model_How_To"></div>

## Using This Model

To use our model, navigate to the ```data``` folder. Then, select ```init.ipynb```, executing all cells sequentially. From here, the user may open both ```forward_process.ipynb``` and ```reverse_process.ipynb```. To perform either process, enter the respective notebook and execute each cell sequentially. This will result in a demonstration of our model.

Alternatively, if the user seeks to test our model on their own image, put the grayscale image to colorize in the demo/img folder. Our demo will then use our trained diffusion model to generate the colorized version of your grayscale image.
Additionally, the command ```python3 colorize.py imgfile``` can be used to generate and show the colorized version of ```imgfile```.

<div id="Conclusion"></div>

## Conclusion

Through the course of this project, we found ourselves enjoying every step. Although extremely ambitiuous for the time/experience between us, the difficulty was worth it. Both in regards to mathematics, and to deep learning, we learned far more than any of expected to. At the end of the process, we find ourselves ecstatic to declare a working pipeline, which we individually seek to optimize. 

To conclude, we would like to bring attention back to the proposal of this work:

"We seek to recolor grayscale photographs..."

_Technically_ we are completely inline with our objective, the colors are just not the right ones, or in the right place.

<div id="References"></div>

## References
[1] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[2] Jascha Sohl-Dickstein et al. “Deep Unsupervised Learning using Nonequilibrium Thermodynamics.” ICML 2015. https://arxiv.org/abs/1503.03585

[3] J. Chang, February 2, 2007. "Markov chains
" http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MarkovChains.pdf

[4] Feller, W. On the theory of stochastic processes, with particular reference to applications. In Proceedings of the
[First] Berkeley Symposium on Mathematical Statistics
and Probability. The Regents of the University of California, 1949.

[5] Deep Learning (Ian J. Goodfellow, Yoshua Bengio and Aaron Courville), MIT Press, 2016. https://www.deeplearningbook.org/

[6] Weng, Lilian. 2018. From Autoencoder to Beta-VAE. "https://lilianweng.github.io/posts/2018-08-12-vae/"


[7] Feller, W. (1966). An introduction to probability theory and its applications (Vol. 2). John Wiley & Sons.

[8] Kingma, Diederik P.; Welling, Max (2014-05-01). "Auto-Encoding Variational Bayes". https://arxiv.org/abs/1312.6114.

[9] Ho, Jonathan, et al. (2020) "Denoising Diffusion Probabilistic Models" https://arxiv.org/pdf/2006.11239.pdf with associated github (https://github.com/lucidrains/denoising-diffusion-pytorch)

[10] Saharia, Chitwan, et al. "Image Super-Resolution via Iterative Refinement
". https://arxiv.org/pdf/2104.07636.pdf.

[11] Alex Nichol and Prafulla Dhariwal (2021), Improved denoising diffusion probabilistic models https://arxiv.org/pdf/2102.09672.pdf) + associated github (https://github.com/openai/improved-diffusion)

[12] Chitwan Saharia, et al. (2022) Palette: Image-to-Image Diffusion Models https://arxiv.org/pdf/2111.05826.pdf

[13] Vaswani, A., et al. (2017) Attention is all you need https://arxiv.org/abs/1706.03762
<!-- #endregion -->
