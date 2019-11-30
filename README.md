# Cartoon Face Generation Using Conditional GAN

In theis project, ACGAN and VAE were implemented for the cartoon face generation. Morevoer, I also did serveral experiments on the model architectures to verify the capabilities of the models.

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/diagram.gif" width="60%" height="60%">

## Preprocessed Cartoon Set

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/Preprocess.png" width="60%" height="60%">

## Usage

The detailed usage is listed in each directory. Note that the pretrained models can be downloaded [here](https://drive.google.com/open?id=1BnhKlJb73f67i0kO8ECw6XJvSTAwv4YA). If you want to reproduce the results, rememeber to download the model files and place them in the right places.


## Algorithm

### 1. ACGAN

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/ACGAN-Alg.png" width="60%" height="60%">

In detail, the class label is divided into four categories (4 one-hots), including the hair style, eye color, face, and eyeglasses. Therefore, the dimension of the class label is multi-hot and the dimension is 15 (6+4+3+2). And because of that, the number of outputs from the discriminator becomes 5. 1 for true/fake label, 4 for each one-hot class label. Moreover, different from ACGAN, I replace batch normalization with spectral normalization on every layer in the discriminator. The other details are the same as ACGAN.

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/ACGAN-Loss.png" width="60%" height="60%">

The loss terms including two parts. The first part is the adversarial loss that is used in every GAN model. As for the second part, it is the classification loss that includes 4 binary cross entropy(BCE) loss. Aside from dividing classification loss into four BCEloss, other implementation details are the same as ACGAN.

### 2. VAE

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/VAE-Alg.png" width="60%" height="60%">

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/VAE-Loss.png" width="60%" height="60%">

* Encoder:4-layer CNN(conv + batch_norm + ReLU), and the last layer is divided into two embeddings (sigma and mean)

* Decoder:5-layer CNN(de-conv + batch_norm + ReLU)

* Batch size = 128, epoch = 1000

* Learning rate = 0.0001

* Loss = MSE_loss + KL_divergence_loss

* Images are normalized to ((0.5,0.5,0.5), (0.5,0.5,0.5))

## Results

### 1. ACGAN (FID = 76.445)

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/original.png" width="40%" height="40%">

### 2. ACGAN + batch normalization (FID score = 198.033)

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/exp1.png" width="40%" height="40%">

This experiment setting use batch normalization in discriminator instead of spectral normalization, which is the same setting in ACGAN.

We can find that the result is worse (mode collapse is more obvious and the resolution is worse), and I observe that the discriminator performs worse as well. I thought that the reason is that spectral normalization is more suitable for this case than batch normalization, which has the same results in the corresponding paper.

### 3. ACGAN + multi-hot generation (FID score = 110.305)

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/exp2.png" width="40%" height="40%">

This experiment directly drives the discriminator to generator multi- hot representation instead of generating 4 one-hot. Therefore, the number of outputs of the discriminator is 2, where one is for real/fake label, and the other is 15-dim vector for class label prediction.

From the result, we can find that the performance is much worse than the original model. By observing the training process, I found that the model training process is much more unstable, and I thought that the reason is that it is much harder for discriminator to generate multi-hot label than one-hot label, which directly influenced the performance of model.


### 4. ACGAN + gradient penalty (FID score = 93.982)

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/exp3.png" width="40%" height="40%">

In this experiment, I add gradient penalty term when training the discriminator and the lambda term is 5 here.

We can find that the training process is the most stable among these methods. I attribute this progress to gradient penalty because it definitely makes the gradient smoother which directly makes the whole training process more stable. This model generates images with more mode collapse but with higher resolution.

### 5. VAE (Unsupervised Conditional Generation)

* Generate by random noise

<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/random.jpg" width="40%" height="40%">

* Interpolation with two random noises
 
<img src="https://github.com/PierreSue/Cartoon-Face-Generation-Using-Conditional-GAN/blob/master/image/interpolation.jpg" width="40%" height="40%">

By this two result demo, we can find that the model can generate different faces with different embeddings. Moreover, via interpolation experiment, we can know that the model somehow acquires some conditional knowledge.