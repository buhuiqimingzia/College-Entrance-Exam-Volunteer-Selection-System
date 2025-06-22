import datetime
import random
from django.contrib import auth
from django.shortcuts import render, redirect
from app01.models import *


# Create your views here.
def index(request):
    return render(request, 'index.html')


def logout(request):
    auth.logout(request)
    return redirect('/')


def home(request):
    score = request.GET.get('score')
    ranking = int(request.GET.get('ranking'))
    choose = request.GET.get('choose')
    profession = request.GET.get('profession')
    type_info = request.GET.get('type')
    score_query = Score.objects.order_by('ranking')
    score_list = []
    for score in score_query:
        if ranking - score.ranking > 3000 or score.ranking - ranking > 6000:
             continue
        if 0 < ranking - score.ranking < 3000:
            print(123)
            probability = random.randint(25, 50)
        if 0 < score.ranking - ranking < 3000:
            probability = random.randint(50, 85)
        if score.ranking - ranking > 3000:
            probability = random.randint(85, 100)

        score_list.append({
            'nid': score.nid,
            'university': score.university,
            'ranking': score.ranking,
            'probability': str(probability) + '%'
        })
    user_info = UserInfo.objects.filter(id=request.user.id).first()
    return render(request, 'home.html', locals())


def user(request):
    user_info = UserInfo.objects.filter(id=request.user.id).first()
    return render(request, 'user.html', locals())


def userinfo(request):
    user_info = UserInfo.objects.filter(id=request.user.id).first()
    return render(request, 'userinfo.html', locals())

def admin_home(request):

    user_count = UserInfo.objects.count()

    sample_count = Score.objects.count()

    solvent_count = 3

    appointment_count = 3

    data_count = 2

    now = datetime.date.today()
    # 今日注册
    today_sign = UserInfo.objects.filter(
        date_joined__gte=now
    ).count()

    #
    theme_count = 28

    online_count = 1  # 在线人数
    return render(request, 'admin_home.html', locals())


def input_score(request):
    user_info = UserInfo.objects.filter(id=request.user.id).first()
    return render(request, 'input_score.html', locals())