import sqlite3
from .get_theatre import get_theatre_detail
# Connect to the SQLite database

def insert_theatre_into_db(placename):
    conn = sqlite3.connect('movie.db')  # Change to your actual database name
    cursor = conn.cursor()

    theatre_st_data = get_theatre_detail(placename)


    insert_theater_query = """
    INSERT INTO Theaters (name, address)
    VALUES (?, ?)
    """

    # Insert each theater, using a try-except block to avoid IntegrityErrors for duplicate entries
    for theater in theatre_st_data:
        try:
            cursor.execute(insert_theater_query, (theater['theatre_name'], theater['theatre_address']))
        except sqlite3.IntegrityError:
            # If the theater already exists, skip insertion
            pass

    insert_showtime_query = """
    INSERT INTO Showtimes (theater, movie, showtime)
    VALUES (?, ?, ?)
    """

    # Insert each showtime for every movie in every theater
    for theater_showtimes in theatre_st_data:
        theater_name = theater_showtimes['theatre_name']
        showtimes = theater_showtimes['showtime']

        for movie, times in showtimes.items():
            for showtime in times:
                try:
                    cursor.execute(
                        insert_showtime_query,
                        (theater_name, movie, showtime)
                    )
                except sqlite3.IntegrityError:
                    pass


    # Commit the changes to the database
    conn.commit()

    # Close the database connection
    conn.close()