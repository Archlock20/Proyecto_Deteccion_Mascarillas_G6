# accounts/urls.py
from django.urls import path
from . import views
from .views import SignUpView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    path('live_mask_detection/', views.live_mask_detection, name='live_mask_detection'),
    path('deteccion/', views.live_mask_detection_view, name='deteccion'),
    path('deteccion_img/', views.img_mask_detection, name='img_mask_detection'),
]