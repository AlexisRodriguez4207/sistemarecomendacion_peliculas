# import pandas as pd
# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import requests

# app = Flask(__name__)

# # Función para realizar solicitudes a la API de TMDb
# def tmdb_request(endpoint, params=None):
#     base_url = 'https://api.themoviedb.org/3/'
#     api_key = '2af2da82d49b988704b95e0a53661965'  # Reemplaza con tu propia API key

#     if params is None:
#         params = {}

#     params['api_key'] = api_key

#     response = requests.get(base_url + endpoint, params=params)
#     if response.status_code == 200:
#         return response.json()
#     return None

# # Realiza una solicitud para obtener películas populares de TMDb
# def get_tmdb_movies():
#     tmdb_endpoint = 'discover/movie'
#     params = {
#         'page': 1,  # Puedes ajustar los parámetros según tus necesidades
#         'primary_release_date.gte': '2015-01-01',  # Movies released from 2010
#         'primary_release_date.lte': '2023-12-31',  # to the end of 2010
#         'sort_by': 'revenue.desc'  # Sort by popularity
#     }
#     return tmdb_request(tmdb_endpoint, params)

# def get_random_backdrop():
#     tmdb_endpoint = 'discover/movie'
#     params = {
#         'primary_release_date.gte': '2015-01-01',
#         'primary_release_date.lte': '2023-12-31',
#         'with_backdrop': True,  # Asegúrate de que la API devuelve películas con backdrop
#         'sort_by': 'popularity.desc'  # Ordena por popularidad para obtener películas populares
#     }
#     data = tmdb_request(tmdb_endpoint, params)
#     if data and data['results']:
#         random_movie = random.choice(data['results'])
#         return random_movie['backdrop_path']
#     return None

# # Obtener datos de películas desde TMDb
# tmdb_data = get_tmdb_movies()

# # Comprobar si se obtuvieron datos válidos desde TMDb
# if tmdb_data:
#     # Convierte los datos de TMDb en un DataFrame de pandas
#     movies = pd.DataFrame(tmdb_data['results'])
#     movies['features'] = movies['title'] + ' ' + movies['overview']  # Utiliza el título y la descripción (overview)
# else:
#     # En caso de no obtener datos válidos de TMDb, utiliza datos locales
#     movies = pd.read_csv('data/moviesn.csv').head(100)

# # Vectorización de características usando TF-IDF
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])

# # Cálculo de similitud del coseno
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# # Ruta para mostrar recomendaciones al usuario
# @app.route('/')
# def index():
#     # Selecciona 20 películas aleatorias
#     random_movies = movies.sample(n=20)
    
    

#     # Simula la selección del usuario de 5 películas
#     user_selection = random_movies.sample(n=5)['title']

#     return render_template('index.html', random_movies=random_movies, user_selection=user_selection)

# # Ruta para mostrar recomendaciones
# @app.route('/recommendations', methods=['POST'])
# def recommendations():
#     selected_movies = request.form.getlist('selected_movies')

#     # Función para obtener recomendaciones
#     def get_recommendations(movie_title, cosine_sim=cosine_sim):
#         idx = movies[movies['title'] == movie_title].index[0]
#         sim_scores = list(enumerate(cosine_sim[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         sim_scores = sim_scores[1:11]
#         movie_indices = [i[0] for i in sim_scores]
#         return movies['title'].iloc[movie_indices]

#     # Recomendaciones basadas en las películas seleccionadas por el usuario
#     recommendations = []
#     for movie_title in selected_movies:
#         recommendations.extend(get_recommendations(movie_title))

#     # Elimina duplicados y las películas ya seleccionadas
#     recommendations = list(set(recommendations) - set(selected_movies))

#     # Obtiene las 10 películas recomendadas finales
#     recommended_movies = recommendations[:10]

#     return render_template('recommendations.html', recommended_movies=recommended_movies)

# if __name__ == '__main__':
#     app.run(debug=True)

#hola
import pandas as pd
import random
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

app = Flask(__name__)

# Función para realizar solicitudes a la API de TMDb
def tmdb_request(endpoint, params=None):
    base_url = 'https://api.themoviedb.org/3/'
    api_key = '2af2da82d49b988704b95e0a53661965'  # Reemplaza con tu propia API key

    if params is None:
        params = {}

    params['api_key'] = api_key

    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    return None

# Función para obtener una imagen de fondo aleatoria
def get_random_backdrop():
    tmdb_endpoint = 'discover/movie'
    params = {
        'primary_release_date.gte': '2015-01-01',
        'primary_release_date.lte': '2023-12-31',
        'with_backdrop': 'true',  # Asegúrate de que la API devuelve películas con backdrop
        'sort_by': 'revenue.desc'  # Ordena por popularidad para obtener películas populares
    }
    data = tmdb_request(tmdb_endpoint, params)
    if data and data['results']:
        random_movie = random.choice(data['results'])
        return random_movie['backdrop_path']
    return None

# Realiza una solicitud para obtener películas populares de TMDb
def get_tmdb_movies():
    tmdb_endpoint = 'discover/movie'
    params = {
        'page': 1,
        'primary_release_date.gte': '2010-01-01',
        'sort_by': 'revenue.desc'
    }
    return tmdb_request(tmdb_endpoint, params)

# Obtener datos de películas desde TMDb
tmdb_data = get_tmdb_movies()

# Comprobar si se obtuvieron datos válidos desde TMDb
if tmdb_data:
    # Convierte los datos de TMDb en un DataFrame de pandas
    movies = pd.DataFrame(tmdb_data['results'])
    movies['features'] = movies['title'] + ' ' + movies['overview']
else:
    # En caso de no obtener datos válidos de TMDb, utiliza datos locales
    movies = pd.read_csv('data/movies.csv').head(100)

# Vectorización de características usando TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])

# Cálculo de similitud del coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Ruta para mostrar recomendaciones al usuario
@app.route('/')
def index():
    random_movies = movies.sample(n=20)
    user_selection = random_movies.sample(n=5)['title']
    backdrop_url = get_random_backdrop()
    full_backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop_url}" if backdrop_url else None
    
    return render_template('index.html', random_movies=random_movies, user_selection=user_selection, backdrop_url=full_backdrop_url)

# Ruta para mostrar recomendaciones
@app.route('/recommendations', methods=['POST'])
def recommendations():
    selected_movies = request.form.getlist('selected_movies')

    def get_recommendations(movie_title, cosine_sim=cosine_sim):
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices]

    recommendations = []
    for movie_title in selected_movies:
        recommendations.extend(get_recommendations(movie_title))

    recommendations = list(set(recommendations) - set(selected_movies))
    recommended_movies = recommendations[:10]

    return render_template('recommendations.html', recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
#prueba sabino

