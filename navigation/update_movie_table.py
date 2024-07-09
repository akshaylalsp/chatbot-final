from .models import Movies  # Update with your actual app name
from .scrapper.get_movie import get_movie_detail

def update_movie(placename):
    Movies.objects.all().delete()
    movies = get_movie_detail(placename)

    for movie_dict in movies:
        movie, created = Movies.objects.get_or_create(
            name=movie_dict['name'],
            defaults={
                'genre': movie_dict['genre'],
                'image': movie_dict['image'],
                'inLanguage': movie_dict['inLanguage'],
                'duration': movie_dict['duration'],
                'datePublished': movie_dict['datePublished'],
                'movie_detail_link': movie_dict['movie_detail_link'],
                'summary': movie_dict['summary'],
                'cast': movie_dict['cast'],
                'rating': movie_dict['rating'],
            }
        )
        if created:
            print(f"Movie '{movie.name}' was created successfully.")
        else:
            print(f"Movie '{movie.name}' already exists.")

