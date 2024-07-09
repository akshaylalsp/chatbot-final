from django.db import models

# Create your models here.
class Movies(models.Model):
    name = models.CharField(max_length=50,primary_key=True)
    image = models.CharField(max_length=50)
    inLanguage = models.CharField(max_length=50)
    duration = models.CharField(max_length=50)
    datePublished = models.CharField(max_length=50)
    movie_detail_link = models.CharField(max_length=50)
    summary = models.CharField(max_length=50)
    genre = models.CharField(max_length=50)
    cast = models.CharField(max_length=50)
    rating = models.IntegerField()

class Theatre(models.Model):
    name = models.CharField(max_length=50)
    address = models.CharField(max_length=50)

{ 'name': 'Nadikar',
  'genre': 'drama',
  'image': 'https://assetscdn1.paytm.com/images/cinema/Nadikar--705x750-dcaab8e0-03b3-11ef-90dc-a7ba56aa725d.jpg',
  'inLanguage': 'Malayalam',
  'duration': 'PT144M',
  'datePublished': '2024-05-03',
  'movie_detail_link': '/movies/nadikar-movie-detail-172443',
  'summary': 'Aalparambil Gopi is an unemployed young person who lives his life mother and younger sister. The story revolves around the Life of Aalparambil Gopi and his dear friend Malgosh.K.P and the little surprises that life has in store for them',
  'cast': 'Nivin Pauly',
  'rating': 6}