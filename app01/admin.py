from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from import_export import resources
from import_export.admin import ImportExportModelAdmin
from app01.models import *
from django.contrib.auth.admin import UserAdmin


# Register your models here.
class UserInfoAdmin(UserAdmin):
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (_("Personal info"),
         {"fields": ("name", "tel", "sex", "age", "school", "major")}),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )

    list_display = ("username", "name", "tel", "sex", "age", "school", "major")

    list_filter = ("is_staff", "is_superuser", "is_active",)

    search_fields = ("username", "name", "tel")

    ordering = ("username",)

    filter_horizontal = (
        "groups",
        "user_permissions",
    )



admin.site.register(UserInfo, UserInfoAdmin)

class ScoreResource(resources.ModelResource):

    class Meta:
        model = Score
        exclude = ['id']
        import_id_fields = ['university']

class ScoreAdmin(ImportExportModelAdmin):



    list_display = ['nid', 'university', 'type', 'year', 'piecewise', 'goal', 'ranking']

    list_filter = ['university']

    search_fields = ['ranking']

    resource_class = ScoreResource


admin.site.register(Score, ScoreAdmin)


admin.site.site_header = '高考报考管理系统'

admin.site.site_title = '高考报考管理系统'
