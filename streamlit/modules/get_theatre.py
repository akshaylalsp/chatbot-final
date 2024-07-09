import re
from bs4 import BeautifulSoup
import requests
import json

def extract_time(s):
    if ':' in s:
        p = s.index(':')
        return convert_to_sqlite_time(s[p-2:p+6])
    else:
        None


def convert_to_sqlite_time(time_str):
    # Regular expression to validate time format (HH:MM AM/PM)
    time_pattern = re.compile(r'^((0?[1-9]|1[0-2]):([0-5]\d)\s([AP]M))$', re.IGNORECASE)

    # Check if the input time string matches the expected format
    if not re.match(time_pattern, time_str):
        raise ValueError("Invalid time format. Time must be in 'HH:MM AM/PM' format.")

    # Split the time string into hours, minutes, and AM/PM
    match = re.match(time_pattern, time_str)
    hours, minutes, am_pm = match.group(2, 3, 4)

    # Convert hours to 24-hour format if necessary
    if am_pm.upper() == 'PM' and hours != '12':
        hours = str(int(hours) + 12)
    elif am_pm.upper() == 'AM' and hours == '12':
        hours = '00'

    # Format the time string in SQLite time format
    sqlite_time = "{:02d}:{:02d}:00".format(int(hours), int(minutes))

    return sqlite_time

def get_theatre_detail(placename):
    results = []  # This list will store each unique theatre's data
    
    endpoint_for_theatre_name = f"https://paytm.com/movies/{placename}/cinema-halls-and-movie-theatre"
    response_for_theatre_name = requests.get(endpoint_for_theatre_name)
    soup = BeautifulSoup(response_for_theatre_name.content, 'html.parser')
    
    # Gather theatre links
    theatre_links = [
        li_tag.a['href']
        for li_tag in soup.find_all(class_=re.compile(r'^CinemaListItem_cinemaListItemCon'))
    ]
    
    for theatre_link in theatre_links:
        # Initialize a new dictionary for each theatre
        json_result = {}  # Important: create a new dictionary for each theatre
        
        endpoint_for_theatre_name = f"https://paytm.com{theatre_link}"
        response = requests.get(endpoint_for_theatre_name)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        theatre_name = soup.find('h2').text.strip()
        theatre_address = soup.find('p').text.strip()
        
        json_result['theatre_name'] = theatre_name
        json_result['theatre_address'] = theatre_address
        
        # Create a dictionary for showtimes
        show_time_json = {}
        
        movie_sections = soup.find_all('div', class_='MovieSessionsListingDesktop_movieSessions__KYv1d')
        
        # Iterate through movie sections to get movie names and showtimes
        for movie_section in movie_sections:
            movie_name = movie_section.find('div', class_='MovieSessionsListingDesktop_movieDetailsDivHeading__INXv0').text.strip()
            
            showtimes_section = movie_section.find_all('div', class_='MovieSessionsListingDesktop_time__r6FAI')
            
            # Collect showtimes for each movie
            showtime_list = []
            for showtime in showtimes_section:
                if showtime:
                    time = extract_time(showtime.get_text(strip=True))
                    showtime_list.append(time)
            
            show_time_json[movie_name] = showtime_list
        
        json_result['showtime'] = show_time_json
        
        results.append(json_result)  # Store the result for each theatre
    
    return results