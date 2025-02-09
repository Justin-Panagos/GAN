from django.urls import path

from pages.views import HomePageView, image_gen

app_name = "pages"

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("generate/", image_gen, name="generate_image"),
]
