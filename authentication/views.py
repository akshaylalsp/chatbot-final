from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegistrationForm, LoginForm

# User registration view
def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()  # Save the new user
            messages.success(request, "Registration successful!")
            return redirect('login')  # Redirect to login page after successful registration
    else:
        form = RegistrationForm()
    return render(request, 'register/index.html', {'form': form})

# User login view
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, "Login successful!")
                return redirect('home')  # Redirect to some protected page
            else:
                print('error in credentils')
                messages.error(request, "Invalid credentials.")
    else:
        print('not post')
        form = LoginForm()
    return render(request, 'login/index.html',{'form':form})

# User logout view
@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('login')  # Redirect to login page after logout