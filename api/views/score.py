from django.views import View
from django.http import JsonResponse


class ScoreView(View):
    def post(self, request):
        res = {
            'code': 500,
            'msg': '查询成功',
            'data': {}
        }

        return JsonResponse(res)