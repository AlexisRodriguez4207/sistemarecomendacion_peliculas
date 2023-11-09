# import pandas as pd
# import random
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

# # Función para obtener una imagen de fondo aleatoria
# def get_random_backdrop():
#     tmdb_endpoint = 'discover/movie'
#     params = {
#         'primary_release_date.gte': '2015-01-01',
#         'primary_release_date.lte': '2023-12-31',
#         'with_backdrop': 'true',  # Asegúrate de que la API devuelve películas con backdrop
#         'sort_by': 'revenue.desc'  # Ordena por popularidad para obtener películas populares
#     }
#     data = tmdb_request(tmdb_endpoint, params)
#     if data and data['results']:
#         random_movie = random.choice(data['results'])
#         return random_movie['backdrop_path']
#     return None

# # Realiza una solicitud para obtener películas populares de TMDb
# def get_tmdb_movies():
#     tmdb_endpoint = 'discover/movie'
#     params = {
#         'page': 1,
#         'primary_release_date.gte': '2010-01-01',
#         'sort_by': 'revenue.desc'
#     }
#     return tmdb_request(tmdb_endpoint, params)

# # Obtener datos de películas desde TMDb
# tmdb_data = get_tmdb_movies()

# # Comprobar si se obtuvieron datos válidos desde TMDb
# if tmdb_data:
#     # Convierte los datos de TMDb en un DataFrame de pandas
#     movies = pd.DataFrame(tmdb_data['results'])
#     movies['features'] = movies['title'] + ' ' + movies['overview']
# else:
#     # En caso de no obtener datos válidos de TMDb, utiliza datos locales
#     movies = pd.read_csv('data/movies.csv').head(100)

# # Vectorización de características usando TF-IDF
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])

# # Cálculo de similitud del coseno
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# # Ruta para mostrar recomendaciones al usuario
# @app.route('/')
# def index():
#     global full_backdrop_url  # Indica que vas a utilizar la variable global

#     random_movies = movies.sample(n=20)
#     user_selection = random_movies.sample(n=5)['title']
#     backdrop_url = get_random_backdrop()
#     full_backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop_url}" if backdrop_url else None
    
#     return render_template('index.html', random_movies=random_movies, user_selection=user_selection, backdrop_url=full_backdrop_url)


# # Función para obtener recomendaciones de películas similares
# def get_recommendations(movie_title, cosine_sim=cosine_sim):
#     idx = movies[movies['title'] == movie_title].index[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]
#     movie_indices = [i[0] for i in sim_scores]
#     return movies['title'].iloc[movie_indices]

# # Ruta para mostrar recomendaciones
# @app.route('/recommendations', methods=['POST'])
# def recommendations():
#     selected_movies = request.form.getlist('selected_movies')
    
#     # Obtener el género de la película seleccionada
#     selected_movie = selected_movies[0]  # Suponiendo que solo se selecciona una película
#     selected_movie_info = tmdb_request('search/movie', {'query': selected_movie})
    
#     if selected_movie_info and 'results' in selected_movie_info:
#         # Obtener el primer resultado (asumimos que es el correcto)
#         selected_movie_genre_ids = selected_movie_info['results'][0]['genre_ids']
        
#         # Filtrar películas por género
#         genre_filtered_movies = movies[movies['genre_ids'].apply(lambda x: any(genre_id in selected_movie_genre_ids for genre_id in x))]

#         # Resto del código para obtener recomendaciones
#         recommendations = []
#         for movie_title in selected_movies:
#             recommendations.extend(get_recommendations(movie_title))
        
#         recommendations = list(set(recommendations) - set(selected_movies))
#         recommended_movies = recommendations[:10]

#         # Pasar datos a la plantilla HTML
#         return render_template('recommendations.html', recommended_movies=recommended_movies, backdrop_url=full_backdrop_url, selected_movie=selected_movie, genre_filtered_movies=genre_filtered_movies)

# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
import random
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

app = Flask(__name__)

# Función para realizar solicitudes a la API de TMDb
def tmdb_request(endpoint, params=None, language='es-ES'):
    base_url = 'https://api.themoviedb.org/3/'
    api_key = '2af2da82d49b988704b95e0a53661965'  # Reemplaza con tu propia API key

    if params is None:
        params = {}

    params['api_key'] = api_key
    params['language'] = language

    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    return None


# Función para obtener los nombres de los géneros basados en los IDs de género
def get_genre_names(genre_ids):
    genre_names = []
    genre_info = tmdb_request('genre/movie/list')  # Hacer una única solicitud fuera del bucle
    if genre_info:
        for genre_id in genre_ids:
            for genre in genre_info['genres']:
                if genre['id'] == genre_id:
                    genre_names.append(genre['name'])
                    break
    return genre_names

# Función para obtener una imagen de fondo aleatoria
def get_random_backdrop():
    tmdb_endpoint = 'discover/movie'
    params = {
        'primary_release_date.gte': '2015-01-01',
        'primary_release_date.lte': '2023-12-31',
        'with_backdrop': 'true',
        'sort_by': 'revenue.desc'
    }
    data = tmdb_request(tmdb_endpoint, params)
    if data and data['results']:
        random_movie = random.choice(data['results'])
        return random_movie['backdrop_path']
    return None

# Función para obtener películas populares de TMDb
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

if tmdb_data and 'results' in tmdb_data:
    movies = pd.DataFrame(tmdb_data['results'])
    movies['features'] = movies['title'] + ' ' + movies['overview']
    movies['genre_names'] = movies['genre_ids'].apply(get_genre_names)  # Añade nombres de géneros al DataFrame
else:
    movies = pd.read_csv('data/movies.csv').head(100)

# Vectorización de características usando TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['features'])

# Cálculo de similitud del coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Ruta para mostrar recomendaciones al usuario
@app.route('/')
def index():
    # Obtener películas al azar
    random_movies = movies.sample(n=20)
    # Seleccionar una película al azar para la sección del héroe
    hero_movie = random_movies.sample(n=1).iloc[0]

    # Obtener la sinopsis en español de la película del héroe
    movie_details = tmdb_request(f'movie/{hero_movie["id"]}', language='es-ES')
    hero_movie_synopsis = movie_details['overview'] if movie_details else ''

    hero_movie_backdrop_url = f"https://image.tmdb.org/t/p/w1280{hero_movie['backdrop_path']}" if hero_movie['backdrop_path'] else None
    hero_movie_poster_path = f"https://image.tmdb.org/t/p/w500{hero_movie['poster_path']}"
    
    return render_template('index.html', 
                           random_movies=random_movies, 
                           hero_movie_title=hero_movie['title'], 
                           hero_movie_synopsis=hero_movie_synopsis, 
                           hero_movie_poster_path=hero_movie_poster_path,
                           hero_movie_backdrop_url=hero_movie_backdrop_url)



# Función para obtener recomendaciones de películas similares
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Función para buscar películas en TMDb por género
def search_movies_by_genre(genre_ids):
    tmdb_endpoint = 'discover/movie'
    params = {
        'with_genres': ','.join(map(str, genre_ids)),  # Convertir la lista de IDs de género a una cadena separada por comas
        'sort_by': 'popularity.desc',  # Ordenar por popularidad (también puedes usar 'revenue.desc' o cualquier otro criterio)
        'page': 1
    }
    return tmdb_request(tmdb_endpoint, params)

# Ruta para mostrar recomendaciones
@app.route('/recommendations', methods=['POST'])
def recommendations():
    selected_movies = request.form.getlist('selected_movies')
    selected_movie = selected_movies[0]

    selected_movie_info = tmdb_request('search/movie', {'query': selected_movie})
    unique_movies = set()
    recommendations_list = []

    if selected_movie_info and 'results' in selected_movie_info:
        selected_movie_data = selected_movie_info['results'][0]
        selected_movie_genre_ids = selected_movie_data['genre_ids'][:3]  # Tomamos los primeros 3 géneros si hay más
        selected_movie_poster_path = selected_movie_data['poster_path']
        selected_movie_genres = get_genre_names(selected_movie_genre_ids)

        # Buscar películas con al menos 2 géneros en común
        for i in range(len(selected_movie_genre_ids)):
            for j in range(i + 1, len(selected_movie_genre_ids)):
                genre_pair = [selected_movie_genre_ids[i], selected_movie_genre_ids[j]]
                genre_movies = search_movies_by_genre(genre_pair)
                if genre_movies and 'results' in genre_movies:
                    for movie in genre_movies['results']:
                        movie_id = movie['id']
                        if movie_id not in unique_movies:
                            unique_movies.add(movie_id)
                            recommendations_list.append(movie)

        # Ordenar por popularidad y tomar los primeros 10
        recommendations_list = sorted(recommendations_list, key=lambda x: x['popularity'], reverse=True)[:10]

        return render_template('recommendations.html', recommended_movies=recommendations_list, selected_movie_genres=selected_movie_genres, selected_movie_poster_path=f"https://image.tmdb.org/t/p/w500{selected_movie_poster_path}")


if __name__ == '__main__':
    app.run(debug=True)
