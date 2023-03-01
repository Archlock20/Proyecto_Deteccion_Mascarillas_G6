# accounts/urls.py
from django.urls import path
from . import views
from .views import SignUpView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    #path("deteccion/", mask_detection_view.as_view(), name="mask_detection_view"),
    path('live_mask_detection/', views.live_mask_detection, name='live_mask_detection'),
]