# Applied Deep Learning, WS24 

Author: Anastasia Cissa, 11937948 

## Table of Contents

1. [Short description and approach](#short-description-and-approach)
2. [References to scientific papers related to the chosen topic](#references-to-scientific-papers-related-to-the-chosen-topic)
3. [Work estimates](#work-estimates)

## Short description and approach

Recommender systems are an essential part of modern applications, from personalized product recommendations to content suggestions. This project will focus on using deep learning approaches within the **RecSys library** to train an efficient recommender system, which  will help solving one of the most routine tasks every person encounters in everyday life, such as picking recipes to use on daily basis.

The goal of this project is to train a personalized recommendation system using deep learning methods, with a focus on collecting a diverse dataset via **surveys, scraping websites**, and other user behavior data. This dataset will include user preferences, item features, and contextual data..

The recommender system will be picked from one of the deep learning architectures (e.g., Neural Collaborative Filtering, RNN-based models, and context-aware models), which will be implemented and fine-tuned using existing frameworks within the [RecSys library](https://github.com/recommenders-team/recommenders/tree/main).

The main objective is to train a **context-aware recommender system** that can suggest recipes based on user preferences and additional contextual information. The use of **deep learning** will enable capturing complex relationships between users and items, as well as dynamic contextual factors.

This project is classified as **"Bring your own data"**, as the core focus will be on collecting a **custom dataset** through surveys, scraping online resources, and aggregating user interaction data. This includes:
- User preferences via **survey forms** (e.g., Google forms or other resources that are relevant and more UX frinedly).
- Scraping **publicly available** datasets (e.g., Spoonacular, Edamam).
- Collecting real-time contextual data (e.g., season, time of day, available ingredients).

## References to scientific papers related to the chosen topic

Papers listed below discuss the current state-of-the-art in deep learning-based recommender systems, providing a strong foundation for the project's approach.

- **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017)**. [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031): Introduces NCF, a deep learning-based collaborative filtering method that uses embeddings and multilayer perceptrons for recommendation.
- **Rendle, S. (2010)**. [Factorization Machines](https://ieeexplore.ieee.org/document/5694074): Describes factorization machines, a technique often combined with deep learning to model pairwise interactions between features.
- **Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, L., Gai, K. (2018)**. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978): Discusses using attention mechanisms to model user interest over time, an approach that can be applied to context-aware recommender systems.
- **Wu, C.-Y., Ahmed, A., Beutel, A., Smola, A. J., & Jing, H. (2017)**. [Recurrent Recommender Networks](https://dl.acm.org/doi/10.1145/3038912.3052632): Demonstrates the use of RNNs to capture user preferences over time in recommendation systems.
- **Wang, N., Chen, G.-D., Tian Y. (2022)**. [Context-Aware Recommender Systems](https://www.mdpi.com/2076-3417/10/9/3022): A discussion of various context-aware models and deep learning techniques for enhancing recommendation accuracy.
- **Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019)**. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations](https://arxiv.org/abs/1904.06814): Applies transformer-based models for sequential recommendation, a method that could be explored in the project.

## Work estimates

| Task                                             | Effort in hours |
|--------------------------------------------------|-----------------|
| Dataset collection via surveys & web scraping    | 20 hours        |
| Implementing and tuning deep learning models     | 15 hours        |
| Training and validation of models                | 10 hours        |
| Testing and evaluating the recommender system    | 5 hours         |
| Application development and integration          | 10 hours        |
| Documentation and code review                    | 4 hours         |
| Writing the final report                         | 4 hours         |
| Presentation preparation                         | 2 hours         |

Total project effort: **91 hours**. However, the complexity of data collection (scraping, survey creation, and gathering responses) could add some additional hours based on user participation and the number of external sources scraped.
