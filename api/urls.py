from django.urls import path
from api.views import login, user, file, admin_data, score

urlpatterns = [
    path('login/', login.LoginView.as_view()),
    path('sign/', login.SignView.as_view()),
    path('userlur/', user.UserLurView.as_view()),
    path('file/', file.AvatarView.as_view()),
    path('score/', score.ScoreView.as_view()),
    path('get_seven_data/', admin_data.get_seven_data),  # 获取七日用户注册
]