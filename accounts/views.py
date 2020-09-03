from django.shortcuts import render

from django.contrib.auth.decorators import login_required
# Create your views here.


@login_required
def profil(request):
    return render(request, 'profile/changer_dataset.html', {})