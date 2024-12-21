# Applied Deep Learning, WS24 

Author: Anastasia Cissa, 11937948 

## Assignment 1 - Initiate

Image colorization is a fascinating problem that's being tackled using deep learning. It's all about taking black-and-white photos and adding realistic colors to them. By using machine learning (ML)algorithms, we can make old pictures look vibrant and alive again. The idea of reviving photos from personal and town archive by applying existing ML models is a reason of choosing this topic for the project. 
Previously, image colorization models weren't used in Moldova (my homecountry); all the work was done by hand. Now, I will try to implement a solution to automate the process, reducing it from a month's work to just a few clicks.

In this project I focus on collecting own data test set, so type of the project is "Bring your own data". It will contain scanned photos from National Archive of Moldova, photos from personal collections and archives and scraped from open source internet resources.  Later on, they will be processed and adjusted, so they could be used by existing models. For training one of these datasets will be used: [Places](http://places.csail.mit.edu/index.html) or [ImageNet](https://image-net.org/index.php). They are free for non-commercial research purposes and provide sufficient amount of the data to train the model. 

Based on research of the existing models, I will try to implement either conditional Generative Adversarial Networks (GAN) or U-NET models. 

### References to scientific papers related to the chosen topic

Task  | Effort in hours | Status
------------- | ------------- | -------------
Dataset collection | 25 hours | Completed
Implementing the model architecture and debugging/testing it | 15 hours | Completed
Training the model | 10 hours | Completed
Running the model | 4 hours | Completed
Implementing tests, documenting the code | 4 hours | Completed
Build application to present results | 7 hours | To-do
Writing the final report | 3 hours | To-do
Preparing presentation of the work | 2 hours | To-do

Most likely time spent on the work will be prolonged due to author's experience. 

## Assignment 2 

**Task:** Build the dataset and train a model on it

### 1. Repository structure

```bash
ADL-WS-2024/
│
├── data/                                       # Folder for data collected
│   ├── combined_data                           # Data used to extend training data
│   └── combined_data_bw                        # Data used as a test set
│
├── wandb_results/                              # Wandb Log Screenshots
│
├── data_web_scraper.ipynb                      # Notebook used to scrape part of the data
├── Image_Colorization_with_U_Net_and_GAN.ipynb # Image Colorization Model 
├── README.md                                   # Project description, setup
├── requirements.txt                            # Python dependencies
└── .gitignore                                  # Ignore unnecessary files (e.g., parts of 'data' folder)

```
### 2. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```
All models can be found and tested using `notebooks` section.

------------

### 3. Dataset Collection

#### **Problem Setting**

For this project, we aim to analyze historical photos related to Moldova. That is done using publicly available data from two main sources:

1. **PastVu**, a [platform](https://pastvu.com/) that curates historical photographs and provides an API for structured access.
2. **Instagram**, leveraging publicly accessible content from the official account of the [**National Agency of Archives (Agenția Națională a Arhivelor)**](https://www.instagram.com/arhiva.gov.md/) and of the 20th-century photographer [**Zaharia Cusnir**] (https://www.instagram.com/zahariacusnir/). To scrape data from Instagram Chrome extension was used. 

#### **Methodology: Scraping Data from the PastVu API**

The **PastVu API** provides a structured way to access historical photo data from its platform. This section describes the API's capabilities and how it was utilized in this study to retrieve and analyze data for Moldova.

**Overview of PastVu API**

The PastVu API supports several methods for retrieving photos and related metadata. For this study, the primary focus was on the following methods:

1. **`photo.getByBounds`**: 
   - Retrieves photos and clusters based on a specified geographical region (bounding box).
   - Returns data at different zoom levels (`z`) for finer granularity.
   
2. **`photo.giveNearestPhotos`**:
   - Fetches photos closest to specified coordinates, sorted by distance.

3. **`photo.giveForPage`**:
   - Provides detailed information about a specific photo using its unique `cid`.

**Method Used: `photo.getByBounds`**

This method was chosen as it allows retrieving all photos within a defined geographical region, making it ideal for covering Moldova's boundaries. The query requires a *GeoJSON* object to specify the bounding box and a zoom level (`z`) for granularity.

**Request Format**

The `photo.getByBounds` method is called using the following URL format: *https://pastvu.com/api2?method=photo.getByBounds&params={...}*

**Parameters**

| Parameter     | Type     | Description                                                                                     |
|---------------|----------|-------------------------------------------------------------------------------------------------|
| `z`           | `int`    | Zoom level (>= 17 for photos only; < 17 returns both photos and clusters).                      |
| `geometry`    | `object` | GeoJSON object specifying the bounding box (Polygon or MultiPolygon).                           |
| `year`        | `int`    | (Optional) Lower limit for the year of the photo.                                               |
| `year2`       | `int`    | (Optional) Upper limit for the year of the photo.                                               |
| `isPainting`  | `bool`   | (Optional) If `true`, only paintings are returned.   

**Defining the Bounding Box for Moldova**

To define the region of interest, Moldova's bounding box was created using its extreme geographical points:

- **Southwest (lon, lat):** `[26.616475, 45.466549]`
- **Northeast (lon, lat):** `[30.163753, 48.491593]`

This bounding box was specified in GeoJSON format:
```json
{
  "type": "Polygon",
  "coordinates": [
    [
      [26.616475, 45.466549],  // SW
      [30.163753, 45.466549],  // SE
      [30.163753, 48.491593],  // NE
      [26.616475, 48.491593],  // NW
      [26.616475, 45.466549]   // Close the polygon
    ]
  ]
}
```

---

### 4. Summary of the Architecture

**Proposed Model and Architecture**
The image colorization model combines a **U-Net** generator and a **PatchGAN** discriminator within a Conditional Generative Adversarial Network (cGAN) framework. The U-Net, built with a ResNet-18 backbone pretrained on ImageNet, uses skip connections to fuse low-level and high-level features. It receives grayscale images (L channel from the L*a*b* color space) as input and predicts the color channels (a*b*). The PatchGAN discriminator, with a 70×70 receptive field, evaluates the realism of these colorized patches, outputting a grid of probabilities indicating whether each patch is real or fake. This architecture balances pixel-wise reconstruction and perceptual realism. 

<img src="/files/unet.png" width="65%">
<p style="text-align:center;"> Generator proposed by the paper</p>


#### Loss Functions
The model optimizes a combined loss function, including an adversarial loss for realism and an L1 loss for pixel-wise accuracy. The adversarial loss encourages the generator to produce realistic outputs, while the L1 loss minimizes the difference between predicted and actual a*b* channels. A hyperparameter (λ) balances the contributions of these losses.

#### Training Strategy
The training process involved two stages. First, the generator was pretrained with L1 loss to stabilize adversarial training and avoid the "blind leading the blind" phenomenon. This pretraining step significantly reduced the required training epochs during adversarial fine-tuning. In the second stage, the generator and discriminator were jointly trained, excluding dropout layers since the grayscale input provided sufficient noise for adversarial learning.

#### Evaluation Metrics
Model performance was assessed using visual comparisons, loss metrics (L1 and adversarial loss), and receptive field analysis. The PatchGAN discriminator ensured local fidelity in colorization, while the U-Net’s skip connections preserved global structure.

#### Related Work
This project draws on the **pix2pix framework** from the paper [_Image-to-Image Translation with Conditional Adversarial Networks_](https://arxiv.org/abs/1611.07004). The U-Net and adversarial training strategies were adapted to the image colorization task.

---

