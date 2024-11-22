from django.shortcuts import render
import os
import pickle as pk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests as rq

# Create your views here.
def home(request):
    try:
        vw = ''
        if request.method == 'GET':
            n1 = request.GET.get('section1', '')  # Use `get` for safety
            vw = n1
    except:
        vw = "Not Available"

    # Path to the pickle file
    pk_path = os.path.join(os.path.dirname(__file__), 'D:\\Work & Study\\Projects\\Movie Recomandation\\MovieR\\movies.pkl')

    with open(pk_path, 'rb') as file:
        model = pk.load(file)
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(model['tags']).toarray()
        similarity = cosine_similarity(vectors)

    def fetch_poster(movie_title):
        """
        Fetch the poster URL for a given movie using the TMDB API.
        """
        try:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key=6f282a339411a930be81b41f3dfd87c1&query={movie_title}"
            response = rq.get(search_url)
            data = response.json()
            if data['results']:
                poster_path = data['results'][0]['poster_path']
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
            else:
                return None  # Return None if no poster is found
        except Exception as e:
            print(f"Error fetching poster for {movie_title}: {e}")
            return None

    def rec(movie):
        """
        Recommend movies based on the input movie.
        """
        try:
            # Get the index of the movie
            movie_index = model[model['title'] == movie].index[0]
            distances = similarity[movie_index]
            movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
            
            # Get recommended movie titles
            recom_movies = [model.iloc[i[0]].title for i in movie_list]
            
            # Get their posters
            recom_movies_posters = [fetch_poster(title) for title in recom_movies]
            
            return recom_movies, recom_movies_posters
        except IndexError:
            return (["Movie not found in dataset. Please try another movie."], [None])

    # Get recommendations and posters for the user input or default movie
    recommended_movies, recommended_posters = rec(vw if vw else 'Alien')

    # Prepare data for rendering
    data = {
        'movie1': recommended_movies[0] if len(recommended_movies) > 0 else "N/A",
        'poster1': recommended_posters[0] if len(recommended_posters) > 0 else None,
        'movie2': recommended_movies[1] if len(recommended_movies) > 1 else "N/A",
        'poster2': recommended_posters[1] if len(recommended_posters) > 1 else None,
        'movie3': recommended_movies[2] if len(recommended_movies) > 2 else "N/A",
        'poster3': recommended_posters[2] if len(recommended_posters) > 2 else None,
        'movie4': recommended_movies[3] if len(recommended_movies) > 3 else "N/A",
        'poster4': recommended_posters[3] if len(recommended_posters) > 3 else None,
        'movie5': recommended_movies[4] if len(recommended_movies) > 4 else "N/A",
        'poster5': recommended_posters[4] if len(recommended_posters) > 4 else None,
        'movie6': recommended_movies[5] if len(recommended_movies) > 5 else "N/A",
        'poster6': recommended_posters[5] if len(recommended_posters) > 5 else None,
    }

    return render(request, "Home.html", data)
    

def about(request):
    return render(request, "about.html")
        
