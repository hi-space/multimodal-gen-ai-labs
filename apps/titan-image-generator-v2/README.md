# [Amazon Titan Image Generator v2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-image-models.html)

Amazon Titan Image Generator G1 is an image generation model. It enables users to generate and edit images in versatile ways. The model provides precise control over the prompt, reference images and color palette.

## Features

### Text-to-image (T2I) generation

Input a text prompt and generate a new image as output. The generated image captures the concepts described by the text prompt.

| Input                                                                                                                           | Output                          |
|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| Modern, high-end beach house interior with panoramic views of the sand and sea, emphasizing sleek design and uncluttered spaces | ![output](./assets/output.webp) |

### Image variation

Uses 1 to 5 images and an optional prompt as input. It generates a new image that preserves the content of the input image(s), but variates its style and background.

| Input                                          |                Output                 |
|------------------------------------------------|:-------------------------------------:|
| ![input](./sample/robots.png) **[text]** Tiger | ![output](./assets/output-robot.webp) |

### Inpainting

Uses an image and a segmentation mask as input (either from the user or estimated by the model) and reconstructs the region within the mask. Use inpainting to remove masked elements and replace them with background pixels.

| Input                                                                                           |                Output                |
|-------------------------------------------------------------------------------------------------|:------------------------------------:|
| ![input](./sample/food.png) **[mask prompt]** bowl of salsa <br> **[text]** bowl of black beans | ![output](./assets/output-food.webp) |

### Outpainting

Uses an image and a segmentation mask as input (either from the user or estimated by the model) and generates new pixels that seamlessly extend the region. Use precise outpainting to preserve the pixels of the masked image when extending the image to the boundaries. Use default outpainting to extend the pixels of the masked image to the image boundaries based on segmentation settings.

| Input                                                                                                 |                 Output                 |
|-------------------------------------------------------------------------------------------------------|:--------------------------------------:|
| ![input](./sample/laptop.png) **[mask prompt]** laptop <br> **[text]** laptop on a granite countertop | ![output](./assets/output-laptop.webp) |

### Image conditioning (V2 only)

Uses an input reference image to guide image generation. The model generates output image that aligns with the layout and the composition of the reference image, while still following the textual prompt.

| Input                                                                   |                Output                 |
|-------------------------------------------------------------------------|:-------------------------------------:|
| ![input](./sample/house.jpg) **[text]** A fairytale house in the forest | ![output](./assets/output-house.webp) |

### Color guided content (V2 only)

You can provide a list of hex color codes along with a prompt. A range of 1 to 10 hex codes can be provided. The image returned by Titan Image Generator G1 V2 will incorporate the color palette provided by the user.

| Input                                                                                                                                                                                                                                                                                                                                                                                                             |                Output                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------:|
| Modern, high-end beach house interior with panoramic views of the sand and sea, emphasizing sleek design and uncluttered spaces <br> **[colors]** <span style="display:inline-block;width:70px;height:20px;background-color:#ff8080;color:#000;text-align:center;">#ff8080</span> <span style="display:inline-block;width:70px;height:20px;background-color:#e5ff80;color:#000;text-align:center;">#e5ff80</span> | ![output](./assets/output-color.webp) |

### Background removal (V2 only)

Automatically identifies multiple objects in the input image and removes the background. The output image has a transparent background.

| Input                         |                 Output                 |
|-------------------------------|:--------------------------------------:|
| ![input](./sample/falcon.png) | ![output](./assets/output-falcon.webp) |
