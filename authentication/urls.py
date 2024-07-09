from . import views
from django.urls import path 
 
urlpatterns = [
    path('login/',view=views.login_view,name='login'),
    path('register/',view=views.register_view,name='register'),
    path('logout/',view=views.logout_view,name='logout')
]