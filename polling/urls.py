from django.urls import path
from . import views

app_name = 'polling'

urlpatterns = [
    path('login/', views.poll_worker_login, name='login'),
    path('', views.polling_home, name='home'),
    path('home/', views.polling_home, name='home_explicit'),
    path('verify-nic/', views.verify_nic, name='verify_nic'),
    path('verify-face/', views.verify_face, name='verify_face'),
]
