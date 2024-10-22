# Applied Deep Learning, WS24 

Author: Anastasia Cissa, 11937948 

## Table of Contents

1. [Short description and approach](#short-description-and-approach)
2. [References to scientific papers related to the chosen topic](#references-to-scientific-papers-related-to-the-chosen-topic)
3. [Work estimates](#work-estimates)

## Short description and approach

Image colorization is a fascinating problem that's being tackled using deep learning. It's all about taking black-and-white photos and adding realistic colors to them. By using machine learning (ML)algorithms, we can make old pictures look vibrant and alive again. The idea of reviving photos from personal and town archive by applying existing ML models is a reason of choosing this topic for the project. 
Previously, image colorization models weren't used in Moldova (my homecountry); all the work was done by hand. Now, I will try to implement a solution to automate the process, reducing it from a month's work to just a few clicks.

In this project I focus on collecting own data test set, so type of the project is "Bring your own data". It will contain scanned photos from National Archive of Moldova, photos from personal collections and archives and scraped from open source internet resources.  Later on, they will be processed and adjusted, so they could be used by existing models. For training one of these datasets will be used: [Places](http://places.csail.mit.edu/index.html) or [ImageNet](https://image-net.org/index.php). They are free for non-commercial research purposes and provide sufficient amount of the data to train the model. 

Based on research of the existing models, I will try to implement either conditional Generative Adversarial Networks (GAN) or U-NET models. 

## References to scientific papers related to the chosen topic

Papers mentioned below describe existing approaches to the problem of Image Colorization. This broad overview of current models influences the core of the future project. 

* [S. Jiang, Q. Luo and H. Bi, "Image Colorization : GAN and Autoencoder Models’ Performance Comparison," 2022](https://ieeexplore.ieee.org/document/9943130) : Comparison of two well-performing models of Image Colorization: GAN and Autoencoder;  
* [Wang, N., Chen, G.-D., Tian Y., "Image Colorization Algorithm Based on Deep Learning", 2022.](https://doi.org/10.3390/sym14112295) :
Paper describes a new improved approach of U-NET model - CU-NET, which focuses on collecting more feature information and reducing preformance time for training; 
* [Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa, "Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification", 2016.](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/) :
CNN approach to Image Colorization problem;
* [Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks", 2017.](https://arxiv.org/abs/1611.07004) :
Conditional GAN approach - the one meant to be implemented in this works;
* [K. Nazeri, E. Ng, M. Ebrahimi, "Image Colorization using Generative Adversarial Networks", 2018](https://link.springer.com/chapter/10.1007/978-3-319-94544-6_9) : 
Thorough description of conditional GAN implementation. 

## Work estimates 

Task  | Effort in hours
------------- | -------------
Dataset collection | 25 hours
Implementing the model architecture and debugging/testing it | 15 hours
Training the model | 10 hours
Running the model | 4 hours
Implementing tests, documenting the code | 4 hours
Build application to present results |  hours
Writing the final report | 3 hours
Preparing presentation of the work | 2 hours

Most likely time spent on the work will be prolonged due to author's experience. 
