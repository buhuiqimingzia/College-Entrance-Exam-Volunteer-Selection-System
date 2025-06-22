from django.contrib.auth.models import AbstractUser
from django.db import models


class UserInfo(AbstractUser):
    id = models.AutoField(primary_key=True)
    tel = models.CharField(verbose_name='手机号', max_length=12, null=True, blank=True)
    name = models.CharField(max_length=128, verbose_name='姓名')
    sex = models.CharField(max_length=128, verbose_name='性别')
    age = models.CharField(max_length=128, verbose_name='年龄')
    school = models.CharField(max_length=128, verbose_name='学校')
    major = models.CharField(max_length=128, verbose_name='专业')

    def __str__(self):
        return self.username

    class Meta:
        verbose_name_plural = '用户信息'


class Avatars(models.Model):
    nid = models.AutoField(primary_key=True)
    url = models.FileField(verbose_name='文件地址', upload_to='file/')

    def __str__(self):
        return str(self.url)

    class Meta:
        verbose_name_plural = '文件管理'



class Score(models.Model):
    nid = models.AutoField(primary_key=True)
    university = models.CharField(verbose_name='大学', max_length=32)
    type = models.CharField(verbose_name='大学类型', max_length=64, null=True)
    year = models.CharField(verbose_name='年份', max_length=64, null=True)
    piecewise = models.CharField(verbose_name='大学分段', max_length=64, null=True)
    goal = models.CharField(verbose_name='最低录取分数', max_length=64, null=True)
    ranking = models.IntegerField(verbose_name='最低录取排名', null=True)

    def __str__(self):
        return str(self.university)

    class Meta:
        verbose_name_plural = '大学录取信息'