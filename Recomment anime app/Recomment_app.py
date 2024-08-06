import os
import numpy as np
import pandas as pd
import pickle

anime_feature = pd.read_pickle('anime_feature.pkl')
anime_data = pd.read_csv('anime.csv')

anime_pivot=anime_feature.pivot_table(index='anime_title',columns='user_id',values='user_rating').fillna(0)


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

anime_matrix = csr_matrix(anime_pivot.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(anime_matrix)

query_index = np.random.choice(anime_pivot.shape[0])

distances, indices = model_knn.kneighbors(anime_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(anime_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i]))

from difflib import get_close_matches

anime_names = list(anime_pivot.index)

def find_similar_animename(search_name, anime_names = anime_names):
    close_matches = get_close_matches(search_name.title(), anime_names, n=10, cutoff=0.5)

    #print(close_matches)
    if close_matches:
        return close_matches[0]
    else:
        return None

name = "initial d"

result = find_similar_animename(name)


def get_recommendations(anime, rows = 10):
    anime = find_similar_animename(anime)
    distances, indices = model_knn.kneighbors(anime_pivot.loc[anime].values.reshape(1, -1), n_neighbors = rows+1)

    similarity = 1 - distances

    recommendations = pd.DataFrame({
        "Anime": anime_pivot.index[indices.flatten()[1:rows+1]],
        "Similarity": similarity.flatten()[1:rows+1]

    })

    return recommendations

import streamlit as st
st.title('Recommend Anime with Colaborative Filter')

# Chọn anime từ danh sách
selected_anime = st.selectbox('Chọn anime:', anime_data['Name'].values)

# Hiển thị thông tin anime đã chọn
st.write(f'Thông tin về anime: {selected_anime}')
st.write(anime_data[anime_data['Name'] == selected_anime])

# Hiển thị các đề xuất
recommendations = get_recommendations(selected_anime)
st.write('Các anime được đề xuất:')
st.write(recommendations)
