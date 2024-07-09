from django.shortcuts import render,redirect,HttpResponse
from django.contrib.auth.decorators import login_required
# from .update_movie_table import update_movie
from .update_movie_table import update_movie
from .update_theatre_table import update_theatre
from .models import Theatre,Movies

# Create your views here.
@login_required
def home(request):
    movies = Movies.objects.all()
    return render(request,'home.html',{'movies':movies})

@login_required
def theatre(request):
    theatres = Theatre.objects.all()
    return render(request,'theatre.html', {'theatres': theatres})

@login_required
def location(request,placename):
    update_movie(placename)
    update_theatre(placename)
    return redirect('home')

@login_required
def askme(request):
    return redirect('http://localhost:8501/')

@login_required
def showtime(request):
    return render(request,'home.html')