from django.urls import path 
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    path('theatre/',views.theatre,name='theatre'),
    path('showtime/',views.showtime,name='showtime'),
    path('askme/',views.askme,name='askme'),
    path('location/<str:placename>',views.location,name='location')
]