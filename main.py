import speech_recognition as sr
import pyttsx3
import webbrowser
import datetime
import os
import random
import psutil
import json
import pickle
import numpy as np
import requests
import yt_dlp as youtube_dl  # Use yt-dlp instead of youtube_dl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Load chatbot model and resources
model = load_model("chat_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("intents.json") as f:
    intents = json.load(f)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}\n")
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            speak("Sorry, I didn't understand that.") 
            return ""
        except sr.RequestError:
            print("Request failed; check your internet connection.")
            speak("Request failed; check your internet connection.")
            return ""
        return command.lower()

def open_app(app_name):
    app_paths = {
        'notepad': "notepad.exe",
        'calculator': "calc.exe",
        'word': "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        'excel': "C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
        'powerpoint': "C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
        'chrome': "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        'cmd': "cmd.exe"
    }
    if app_name in app_paths:
        os.startfile(app_paths[app_name])
        speak(f"{app_name} is opened.")
    else:
        speak(f"Sorry, I don't know how to open {app_name}.")

def play_music(command):
    if 'play' in command and 'song' in command:
        song_name = command.replace('play', '').replace('song', '').strip()
        if song_name:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'noplaylist': True,
                'default_search': 'ytsearch',
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(song_name, download=False)
                    video_url = info['entries'][0]['webpage_url']
                    webbrowser.open(video_url)
                    speak(f"Playing {song_name} on YouTube.")
                except Exception as e:
                    speak("Sorry, I couldn't find that song.")
                    print(e)
        else:
            speak("Sorry, I didn't catch the song name.")
    else:
        playlist_dir = r'C:\Users\jhaaa\Music\Playlists'  # Directory where the playlist is stored
        playlist_file = 'HINDI.m3u8'  # Replace with your actual playlist file name
        playlist_path = os.path.join(playlist_dir, playlist_file)
        if os.path.exists(playlist_path):
            try:
                os.startfile(playlist_path)  # This should open the playlist in the default media player
                speak(f"Playing music from {playlist_file}")
            except Exception as e:
                speak("Error opening the playlist.")
                print(f"Error details: {e}")
        else:
            speak("Playlist file not found.")

def get_weather():
    api_key = "62fd163d86fb1c22cca3963f43cb86c2"  # Replace with your OpenWeatherMap API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    speak("Please tell me the city name.")
    city_name = take_command()
    if city_name:
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        data = response.json()
        if data["cod"] != "404":
            weather = data["main"]
            temp = weather["temp"] - 273.15  # Convert Kelvin to Celsius
            humidity = weather["humidity"]
            description = data["weather"][0]["description"]
            speak(f"The temperature in {city_name} is {temp:.2f} degrees Celsius with {description}. The humidity is {humidity} percent.")
        else:
            speak("City not found, please try again.")
    else:
        speak("Sorry, I couldn't get the city name.")

def system_condition():
    usage = str(psutil.cpu_percent())
    speak(f"CPU is at {usage}% usage.")
    battery = psutil.sensors_battery()
    speak(f"Battery is at {battery.percent}%.")

def wish_me():
    hour = datetime.datetime.now().hour
    if 0 <= hour < 12:
        speak("Good morning!")
    elif 12 <= hour < 18:
        speak("Good afternoon!")
    else:
        speak("Good evening!")

def chat(query):
    sequences = tokenizer.texts_to_sequences([query])
    padded_sequences = pad_sequences(sequences, maxlen=20)
    prediction = model.predict(padded_sequences)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    if tag == "datetime":
        get_date_time()
    else:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                speak(random.choice(intent['responses']))
                break

def get_date_time():
    now = datetime.datetime.now()
    date_time_str = now.strftime("%A, %d %B %Y, %H:%M:%S")
    speak(f"The current date and time is {date_time_str}.")

def execute_command(command):
    if 'open' in command:
        if 'notepad' in command:
            open_app('notepad')
        elif 'calculator' in command:
            open_app('calculator')
        elif 'word' in command:
            open_app('word')
        elif 'excel' in command:
            open_app('excel')
        elif 'powerpoint' in command:
            open_app('powerpoint')
        elif 'chrome' in command:
            open_app('chrome')
        elif 'cmd' in command:
            open_app('cmd')
        elif 'google' in command:
            speak("What should I search on Google?")
            search_query = take_command()
            if search_query:
                webbrowser.open(f"https://www.google.com/search?q={search_query}")
                speak("Here are the results.")
            else:
                speak("No search query provided.")
        else:
            speak("Application not recognized.")
    elif 'system' in command and 'condition' in command:
        system_condition()
    elif 'play music' in command or 'play song' in command:
        play_music(command)
    elif 'weather' in command:
        get_weather()
    elif 'date' in command or 'time' in command:
        get_date_time()
    elif any(keyword in command for keyword in ["what", "who", "how", "hi", "thanks", "hello"]):
        chat(command)
    elif 'exit' in command:
        speak("Goodbye!")
        exit()
    else:
        speak("Sorry, I didn't understand that command.")

if __name__ == "__main__":
    wish_me()
    while True:
        command = take_command()
        if command:
            execute_command(command)
