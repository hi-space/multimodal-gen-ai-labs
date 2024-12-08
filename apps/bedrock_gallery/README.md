
## Model Details

### Amazon Titan Image Generator v2

| **Attribute**                                                                          | **Details**                                                                                                   |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Model ID**                                                                           | `amazon.titan-image-generator-v2:0`                                                                           |
| **Max Input Characters**                                                               | 512 characters                                                                                                |
| **Max Input Image Size**                                                               | 5 MB (only specific resolutions supported)                                                                    |
| **Max Image Size (Inpainting, Background Removal, Image Conditioning, Color Palette)** | 1,408 x 1,408 px                                                                                              |
| **Max Image Size (Image Variation)**                                                   | 4,096 x 4,096 px                                                                                              |
| **Languages**                                                                          | English                                                                                                       |
| **Output Type**                                                                        | Image                                                                                                         |
| **Supported Image Types**                                                              | JPEG, JPG, PNG                                                                                                |
| **Inference Types**                                                                    | On-Demand, Provisioned Throughput                                                                             |
| **Supported Use Cases**                                                                | - Image Generation<br>- Image Editing<br>- Image Variations<br>- Background Removal<br>- Color Guided Content |

---

### Amazon Nova Canvas

| **Attribute**                                | **Details**                                                                                                              |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Model ID**                                 | `amazon.nova-canvas-v1:0`                                                                                                |
| **Max Input Characters**                     | 1024 characters                                                                                                          |
| **Max Output Resolution (Generation Tasks)** | 4.19 million pixels (e.g., 2048x2048, 2816x1536)                                                                         |
| **Max Output Resolution (Editing Tasks)**    | - Longest Side: Up to 4096 pixels<br>- Aspect Ratio: Between 1:4 and 4:1<br>- Total Pixel Count: 4.19 million or smaller |
| **Languages**                                | English                                                                                                                  |
| **Supported Input Image Types**              | PNG, JPEG                                                                                                                |
| **Regions**                                  | US East (N. Virginia)                                                                                                    |
| **Invoke Model API**                         | Yes                                                                                                                      |

---

### Amazon Nova Real

| **Attribute**                   | **Details**             |
|---------------------------------|-------------------------|
| **Model ID**                    | `amazon.nova-reel-v1:0` |
| **Input Modalities**            | Text, Image             |
| **Output Modalities**           | Video                   |
| **Input Context Window (Text)** | 512 characters          |
| **Languages**                   | English                 |
| **Video Resolution**            | 1280x720                |
| **Frames Per Second**           | 24                      |
| **Video Duration**              | 6 seconds               |
| **Regions**                     | US East (N. Virginia)   |
| **Async Invoke Model API**      | Yes                     |
