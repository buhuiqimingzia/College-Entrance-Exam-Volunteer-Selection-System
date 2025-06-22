from django.views import View
from django.http import JsonResponse
from app01.models import Avatars
from django.core.files.uploadedfile import InMemoryUploadedFile


class AvatarView(View):
    def post(self, request):
        res = {
            'code': 412,
            'msg': '文件上传不合法！'
        }
        file: InMemoryUploadedFile = request.FILES.get('file')

        Avatars.objects.create(url=file)
        res['code'] = 0
        res['msg'] = '上传成功'
        return JsonResponse(res)


