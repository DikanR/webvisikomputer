from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("tugas2", views.tugas2, name="tugas2")
]