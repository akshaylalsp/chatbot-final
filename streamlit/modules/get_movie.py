from bs4 import BeautifulSoup
import requests
import json
import re

def get_cast_summary(link,moviedetail):
    html_content = requests.get(f'https://paytm.com{link}')
    soup1 = BeautifulSoup(html_content.content, 'html.parser')
    divs = soup1.find_all('div',class_=re.compile(r'^MovieDetail'))
    
    summary = ''
    for div in divs:
        try:
            summary = div.find('p').text.strip()
            break
        except:
            pass
    moviedetail['summary'] = summary
    
    lead_cast = ""

    divs = soup1.find_all('div',class_=re.compile(r'^MovieCast_celebName'))
    try:
        lead_cast = divs[0].text
    except:
        lead_cast = ''
    moviedetail['cast'] = lead_cast


def get_movie_detail(placename):
    place_endpoint = f'https://paytm.com/movies/{placename}'
    response = requests.get(place_endpoint)
    soup = BeautifulSoup(response.content, 'html.parser')
    movies = []
    elements = soup.find_all(class_=re.compile(r'^DesktopRunningMovie_movieCard'))
    for element in elements:
        script = element.find('script', type='application/ld+json')
        link = element.find('a')['href']

        if script:
            # Extract JSON-LD data from the script tag
            json_data = script.string
            json_obj = json.loads(json_data)
            json_obj['movie_detail_link'] = link
            movies.append(json_obj)

    print(len(movies))


    for movie in movies:
        get_cast_summary(movie.get('movie_detail_link'),movie)

        try:
            rating = movie.get('aggregateRating').get('ratingValue')
        except Exception:
            rating = 7  ## hehe
        movie['rating'] = rating
        genres = movie['genre'] 

        if genres == '':
            genre = 'drama'
        else:
            genre_list = [genre.strip() for genre in genres.split(",")]
            try:
                genre = genre_list[0]
            except:
                genre = ""
        movie['genre'] = genre
        keys_to_remove = ['@context', '@type', 'releasedEvent','aggregateRating','url']
        for key in keys_to_remove:
            print(f'deleting {key}')
            try:
                del movie[key]
            except Exception:
                pass
            
    return movies
    

