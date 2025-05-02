# Movielens Recommendation System
This project uses the MovieLens‑100K dataset. The dataset contains 100,000 ratings from 943 users on 1,682 movies, it is lightweight, fast to load and iterate.


Movie recommendation system : https://youtu.be/t34LVX_d5P8


![image](https://github.com/user-attachments/assets/063affa1-5077-41c6-b1da-f439d3af125b)

## Overview

This project explores several recommendation strategies on the MovieLens dataset (100K/1M). Each model is trained to predict user ratings, and performance is evaluated using RMSE and MAE metrics.

---

## Data

- **Dataset**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)  
- **Preprocessing**:  
  1. Load ratings (`userId`, `movieId`, `rating`, `timestamp`).  
  2. Filter movies with fewer than 50 ratings for the popularity baseline.  
  3. Split into train/test (e.g., 80/20 random split).

---

## Models

| Category                | Model                         | Key Idea                                                                                 |
|-------------------------|-------------------------------|------------------------------------------------------------------------------------------|
| **Baseline**            | Popularity-Based              | Rank movies by overall average rating (min. 50 votes).                                   |
| **Collaborative Filtering** | User–User Collaborative Filtering | KNN with cosine similarity on user-rating vectors.                                       |
|                         | Item–Item Collaborative Filtering | KNN with cosine similarity on item-rating vectors.                                       |
| **Matrix Factorization**| SVD                           | Latent-factor model via Surprise’s SVD implementation.                                   |
| **Deep Learning**       | Neural CF (NCF-Deep)          | Keras MLP on concatenated user/item embeddings.                                          |

---

## Installation and Usage

**Clone repository**  
   ```bash
   git clone https://github.com/yourusername/movielens-recsys.git
   cd movielens-recsys

  set FLASK_APP=app.py
  flask run


