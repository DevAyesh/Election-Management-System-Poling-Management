from django.urls import path
from . import views

urlpatterns = [
    path('api/register_voter/', views.register_voter, name='register_voter'),
    path('api/verify_voter/', views.verify_voter, name='verify_voter'),
    path('verify/', views.verification_page, name='verification_page'),
]
