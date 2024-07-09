import sqlite3
from .get_movie import get_movie_detail
# Connect to the SQLite database (or create it if it doesn't exist)

def insert_movie_into_db(placename):
    conn = sqlite3.connect('movie.db')  # Replace 'movies.db' with your database name
    cursor = conn.cursor()

    movies = get_movie_detail(placename)

    # movies = [
    #     {
    #         'name': 'Nadikar',
    #         'genre': 'drama',
    #         'image': 'https://assetscdn1.paytm.com/images/cinema/Nadikar--705x750-dcaab8e0-03b3-11ef-90dc-a7ba56aa725d.jpg',
    #         'inLanguage': 'Malayalam',
    #         'duration': 'PT144M',
    #         'datePublished': '2024-05-03',
    #         'movie_detail_link': '/movies/nadikar-movie-detail-172443',
    #         'summary': 'The story follows David, a superstar, when he is forced to look within to find his real self and go beyond the facade of the stardom.',
    #         'casts': 'Tovino Thomas',
    #         'rating': 6.0
    #     },
    #     # Add the rest of your movies here...
    # ]

    # Parameterized insert statement to prevent SQL injection
    insert_query = """
    INSERT INTO Movies (name, genre, image, inLanguage, duration, datePublished, movie_detail_link, summary, casts, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Insert each movie into the database
    for movie in movies:
        cursor.execute(
            insert_query,
            (
                movie['name'],
                movie['genre'],
                movie['image'],
                movie['inLanguage'],
                movie['duration'],
                movie['datePublished'],
                movie['movie_detail_link'],
                movie['summary'],
                movie['cast'],
                movie['rating'],
            )
        )

    # Commit the transaction to save the changes
    conn.commit()
