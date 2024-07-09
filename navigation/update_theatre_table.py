from .models import Theatre  # Import your model
from django.db import transaction  # Optional, to ensure atomicity
from .scrapper.get_theatre import get_theatre_detail
# Given data in a list of dictionaries
# data = [
#     {
#         'theatre_name': 'President Cinemas, Adarsh Nagar',
#         'theatre_address': 'President Cinemas, Adarsh nagar, GT Road , , Near President Hotel , Dasuya, 144205.',
#         'showtime': {
#             'Shayar': ['15:00:00', '20:00:00'],
#             'Pind Aala School': ['17:50:00'],
#         },
#     },
# ]
def update_theatre(placename):
    Theatre.objects.all().delete()
    data = get_theatre_detail(placename)

    # Optional: Use transaction.atomic() to ensure all-or-nothing for the database operations
    with transaction.atomic():
        for entry in data:
            theatre_name = entry['theatre_name']
            theatre_address = entry['theatre_address']
            
            # Create or get the Theatre instance
            theatre, created = Theatre.objects.get_or_create(
                name=theatre_name,
                defaults={'address': theatre_address},
            )

            if created:
                print(f"Theatre '{theatre.name}' was created.")
            else:
                print(f"Theatre '{theatre.name}' already exists.")