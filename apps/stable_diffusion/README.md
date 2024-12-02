# Bedrock Stable Diffusion

You can now use Stability AI's Text-to-Image models on Amazon Bedrock (as of September 4, 2024).

Stable Image Ultra, Stable Diffusion 3 Large, and Stable Image Core models significantly improve performance in multi-subject prompts, image quality, and typography, and can be used to quickly generate high-quality visuals for various use cases such as marketing, advertising, media, entertainment, and retail. ([Link](https://aws.amazon.com/blogs/aws/stability-ais-best-image-generating-models-now-in-amazon-bedrock/))

Compared to Stable Diffusion XL (SDXL), one of the key improvements of Stable Image Ultra and Stable Diffusion 3 Large is the enhanced text quality in generated images and reduced spelling and printing errors, thanks to the innovative Diffusion Transformer architecture. This architecture implements two separate weight sets for images and text while enabling information flow between the two modalities.

### Request Field

- `mode` (str)
  - `text-to-image`, `image-to-image` (only use in Stable Image 3)
- `aspect_ration` (str)
  - `16:9` / `1:1` / `21:9` / `2:3` / `3:2` / `4:5` / `5:4` / `9:16` / `9:21`
- `negative_prompt`
- `seed`
- `output_format` (str)
  - `png` / `jpeg` / `webp`
