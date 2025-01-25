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
Build application to present results | 7 hours | Completed
Writing the final report | 3 hours | Completed
Preparing presentation of the work | 2 hours | Completed

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
├── report/                                     # Report for the 3rd assignment
│
├── files/                                      # Pictures used in README.md
│
├── results/                                    # Model results for the test set
│
├── wandb_results/                              # Wandb Log Screenshots
│
├── data_web_scraper.ipynb                      # Notebook used to scrape part of the data
├── Image_Colorization_with_U_Net_and_GAN.ipynb # Image Colorization Model 
├── README.md                                   # Project description, setup
├── requirements.txt                            # Python dependencies
├── model.py                                    # Model architecture needed for streamlit application
├── streamlit_app.py                            # Streamlit application
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
2. **Instagram**, leveraging publicly accessible content from the official account of the [**National Agency of Archives (Agenția Națională a Arhivelor)**](https://www.instagram.com/arhiva.gov.md/) and of the 20th-century photographer [**Zaharia Cusnir**](https://www.instagram.com/zahariacusnir/). To scrape data from Instagram Chrome extension was used. 

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

### 5. Results

Firstly, let's take a look at one of the results we got from running trainbed model at the test set. 

<img src="/results/download (9) - 5.png" width="65%">

First line represents pictures turned into the same greyscale palette, the second row is output of the model, and the third one are original pictures. 

If we take a look at all other pictures in `results` folder, we can that model uses lots of magenta and green colors in unexpected places, and it performs worse on pictures that show specific scenarios connected to traditions, customs and history represented. 

Secondly, let's take a look on results we got during the training. 

<table>
  <caption>Results of the first training </caption>
  <tr>
    <td><img src="wandb_results\train1_1.svg" alt="Image 1" width="300"></td>
    <td><img src="wandb_results\train1_2.svg" alt="Image 2" width="300"></td>
  </tr>
  <tr>
    <td><img src="wandb_results\train1_3.svg" alt="Image 3" width="300"></td>
    <td><img src="wandb_results\train1_4.svg" alt="Image 4" width="300"></td>
  </tr>
  <tr>
    <td><img src="wandb_results\train1_5.svg" alt="Image 5" width="300"></td>
    <td><img src="wandb_results\train1_6.svg" alt="Image 6" width="300"></td>
  </tr>
</table>

The training graphs reveal key insights into the behavior of the generator and discriminator during the first trainer's progress.

The **Loss_D_fake** tracks the discriminator's ability to identify fake images. It initially rises as the generator improves, then stabilizes as the discriminator adapts. Similarly, **Loss_D_real** reflects the discriminator’s ability to classify real images. Its early increase suggests the generator is producing better outputs, but it eventually decreases as the discriminator improves.

The **Loss_D**, representing the overall discriminator loss, combines both tasks. It follows a similar trend, with initial fluctuations stabilizing as the models reach equilibrium.

The **Loss_G_L1**, the pixel-wise generator loss, decreases steadily, indicating the generator is improving its accuracy in predicting realistic color channels. The **Loss_G_GAN**, which measures how well the generator fools the discriminator, drops sharply early on, showing rapid improvement, and later stabilizes. Lastly, **Loss_G**, the total generator loss, steadily declines, reflecting overall progress in both accuracy and realism.

In summary, the generator and discriminator initially struggle, but their performance stabilizes as training progresses, with the generator producing increasingly realistic and accurate outputs.

<table>
  <caption>Results of the second training </caption>
  <tr>
    <td><img src="wandb_results\train2-3.svg" alt="Image 7" width="300"></td>
    <td><img src="wandb_results\train2-1.svg" alt="Image 8" width="300"></td>
  </tr>
  <tr>
    <td><img src="wandb_results\train2-2.svg" alt="Image 9" width="300"></td>
    <td><img src="wandb_results\train2-4.svg" alt="Image 10" width="300"></td>
  </tr>
  <tr>
    <td><img src="wandb_results\train2-5.svg" alt="Image 11" width="300"></td>
    <td><img src="wandb_results\train2-6.svg" alt="Image 12" width="300"></td>
  </tr>
</table>

We can see that results are significantly improved.

## Assignment 3 - Deliver 

During this stage we had to deploy our model online. For this purposes I decided to create a ´streamlit´ application. You can run it by installing streamlit library and then run the following code. 

```bash
streamlit run streamlit_app.py 
```

To ensure that everyone could try the application, I decided to use HuggingFace Spaces, so it becomes accessible right from link. This link will lead you to the application - [HuggingsFace Spaces Apllication link]([https://image-net.org/index.php](https://huggingface.co/spaces/augustowski/image_colorization_app)). 
The report for tha Assignment 3 can be found in the folder ´report´. 
