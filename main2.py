import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

inf = pd.read_csv('peliculas_data.csv')

app = FastAPI()

@app.get("/peliculas_idioma/{idioma}")
def peliculas_idioma(idioma:str):
    cantidad = int(inf[inf['original_language'] == idioma].shape[0])
    return {'La cantidad de películas producidas en idioma': idioma, 'es de ': cantidad}

@app.get("/peliculas_duracion/{Pelicula}")
def peliculas_duracion( Pelicula: str ):
    pelicula = inf[inf['title'] == Pelicula]
    duracion = int(pelicula['runtime'].values[0])
    año = int(pelicula['release_year'].values[0])
    return {'La película': Pelicula, 'tiene una duracion de': duracion , 'minutos y fué estrenada en el año': año}

@app.get("/franquicia/{Franquicia}")
def franquicia( Franquicia: str ):
    fran = inf[inf['belongs_to_collection'].str.contains(Franquicia, na=False)]
    cantidad = int(fran.shape[0])
    ganancia = float(fran['revenue'].sum())
    ganancia_prom = float(fran['revenue'].mean())
    return {'La cantidad de peliculas pertenecientes a la franquicia': Franquicia, 'es de': cantidad, 'recaudando una ganacia de': ganancia, 'siendo el promedio de': ganancia_prom}

@app.get("/peliculas_pais/{Pais}")
def peliculas_pais( Pais: str ):
    can = inf[inf['production_countries'].str.contains(Pais, na=False)]
    cantidad = int(can['title'].count())
    return {'En el país': Pais, 'la cantidad de películas producidas es de': cantidad}

@app.get("/productoras_exitosas/{Productora}")
def productoras_exitosas( Productora: str ):
    alt = inf[inf['production_companies'].str.contains(Productora, na=False)]
    ganancia = int(alt['revenue'].sum())
    cantidad = int(alt['title'].count())
    return {'La productora': Productora, 'ha tenído una ganancia de': ganancia, 'habiendo producido una cantidad de películas de': cantidad}

@app.get("/get_director/{nombre_director}")
def get_director( nombre_director ):
    alt = inf[inf['crew'].str.contains(nombre_director, na=False)]
    exito = float(alt['return'].sum())
    peliculas = alt[['title', 'release_year', 'return', 'budget', 'revenue']].values.tolist()
    return {'El director': nombre_director, 'tuvo un éxito de': exito, 'teniendo en cuenta las siguientes peliculas': peliculas}

@app.get("/recomendacion/{titulo}")
def recomendacion( titulo ):
    # Convertir la columna de popularidad a tipo numérico
    inf['popularity'] = pd.to_numeric(inf['popularity'], errors='coerce')

    # Reemplazar valores NaN en la columna de sinopsis por una cadena vacía
    inf['cast'] = inf['cast'].fillna('')

    # Reemplazar valores NaN en la columna de belongs_to_collection por una cadena vacía
    inf['belongs_to_collection'] = inf['belongs_to_collection'].fillna('')
    
    # Reemplazar valores NaN en la columna de crew por una cadena vacía
    inf['crew'] = inf['crew'].fillna('')

    # Filtrar películas por popularidad
    valor = 5
    df_filtrado = inf[inf['popularity'] > valor]

    # Resetear los índices del DataFrame filtrado
    df_filtrado = df_filtrado.reset_index(drop=True)

    # Obtener las características de las películas
    directores = df_filtrado['crew'].tolist()
    casting = df_filtrado['cast'].tolist()
    genres = df_filtrado['genres'].tolist()
    collection = df_filtrado['belongs_to_collection'].tolist()

    n_components = min(50, len(genres[0].split('|')))

    # Vectorizar los castings de las películas utilizando TF-IDF
    vectorizer_casting = TfidfVectorizer()
    casting_vectors = vectorizer_casting.fit_transform(casting)
    # Reducción de dimensionalidad con LSA
    lsa_model_casting = TruncatedSVD(n_components=n_components)
    casting_vectors_reduced = lsa_model_casting.fit_transform(casting_vectors)

    vectorizer_collection = TfidfVectorizer()
    collection_vectors = vectorizer_collection.fit_transform(collection)
    lsa_model_collection = TruncatedSVD(n_components=n_components)
    collection_vectors_reduced = lsa_model_collection.fit_transform(collection_vectors)

    vectorizer_genres = TfidfVectorizer()
    genres_vectors = vectorizer_genres.fit_transform(genres)
    lsa_model_genres = TruncatedSVD(n_components=n_components)
    genres_vectors_reduced = lsa_model_genres.fit_transform(genres_vectors)
    
    vectorizer_directores = TfidfVectorizer()
    directores_vectors = vectorizer_directores.fit_transform(directores)
    lsa_model_directores = TruncatedSVD(n_components=n_components)
    directores_vectors_reduced = lsa_model_directores.fit_transform(directores_vectors)
    
    feature_vectors = np.concatenate((casting_vectors_reduced, collection_vectors_reduced, genres_vectors_reduced, directores_vectors_reduced), axis=1)
    column_names = ['Feature_' + str(i+1) for i in range(4 * n_components)]
    df_feature_vectors = pd.DataFrame(data=feature_vectors, columns=column_names)
    feature_vectors = df_feature_vectors.values

    n_neighbors = 6
    metric = 'cosine'
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(feature_vectors)

    movie_index = df_filtrado[df_filtrado['title'] == titulo].index[0]

    s, indices = model.kneighbors(feature_vectors[movie_index].reshape(1, -1))

    recommended_movies = df_filtrado.loc[indices.flatten()].copy()

    return recommended_movies[['title']].head(5)