**Report: Image Generation from Textual Descriptions**

**Introduction:**
In this report, we present a comprehensive process of generating images(Facial images) from textual descriptions using GAN. Our approach involves using pre-trained models for data set generation, and custom neural networks to translate textual descriptions into latent vectors suitable for image generation.
We used styleGAN3 architecture from NVIDIA Labs in this work, and pretrained model trained with facial images in this architecture.

**Dataset Generation:**
We generated a dataset consisting of images paired with textual descriptions. We generated face images with the pretrained model from NVLabs styleGAN3 architecture. The textual descriptions were generated using another pre-trained image captioning model, specifically the Salesforce/blip-image-captioning-base model. This model is capable of generating descriptive captions for images, providing us with a diverse dataset for training our image generation model.

**StyleGAN3 Setup:**
To generate realistic images from latent vectors, we utilized StyleGAN3, a generative adversarial network (GAN) architecture from NVLabs NVIDIA. The StyleGAN3 model can produce high-quality, diverse images with realistic details. We obtained the pre-trained StyleGAN3 model trained on the FFHQ dataset, enabling us to generate images with resolutions up to 1024x1024 pixels. later we upscaled the generated image in the end-to-end step to 2048x2048 pixels.

**Language Encoder Setup:**
Textual descriptions were encoded into numerical vectors using a language encoder model. Specifically, we employed the Sentence Transformers library and used the all-MiniLM-L6-v2 model for encoding textual descriptions into fixed-size vector representations. This encoding process converts the descriptive text into a format suitable for processing by neural networks.

**Model Training:**
We designed and trained a custom neural network model with same architecture as an earlier project, named LaTran, to translate the encoded textual descriptions into latent vectors that can be fed into the StyleGAN3 model. LaTran consists of a sequential pipeline of linear layers followed by activation functions, aimed at transforming the input textual embeddings into latent vectors suitable for image generation. The training process involved optimizing the model parameters using the mean squared error (MSE) loss function, minimizing the discrepancy between predicted and target latent vectors.

**End-to-End Image Generation:**
Upon successful training of the LaTran model, we proceeded to generate images from textual descriptions in an end-to-end manner. This process involved encoding the input text using the language encoder, translating it into a latent vector using the trained LaTran model, and finally generating an image using the StyleGAN3 model. By integrating these components seamlessly, we were able to generate realistic and diverse images corresponding to a wide range of textual attributes.

**Image Generation Examples:**
To demonstrate the effectiveness of our approach, we provided examples of image generation based on both simple and complex textual descriptions. These examples showcase capability to generate diverse images corresponding to various attributes such as gender, age, facial expressions, and combinations thereof.

**Conclusion:**
In conclusion, our study highlights the potential of deep learning techniques for generating images from textual descriptions. By leveraging pre-trained models and custom neural networks, we were able to achieve realistic and diverse image generation results.
Due to lack of computing capability, the model was trained on a comparitively small number size of data set(20k).
