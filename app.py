import streamlit as st
import requests
import datetime
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import st_folium
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

API_KEY="d0f4215f39312e5de368ee8edad554b8"

# ---------------- CONFIG ----------------

ICONS={
"clear":"icons/clear.png",
"cloud":"icons/clouds.png",
"drizzle":"icons/drizzle.png",
"snow":"icons/snow.png",
"mist":"icons/mist.png",
"thunder":"icons/thunderstorm.png",
"wind":"icons/wind.png",
"humidity":"icons/humidity.png",
"pressure":"icons/pressure.png",
"temp":"icons/high_temp.png"
}

WALLPAPERS={
"clear":"wallpapers/bg2.png",
"cloud":"wallpapers/bg3.png",
"rain":"wallpapers/bg6.png",
"default":"wallpapers/bg1.png"
}

# ---------------- UTIL ----------------

def encode(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def icon_img(path,w=60):
    return f'<img src="data:image/png;base64,{encode(path)}" width="{w}">'

# ---------------- WEATHER ICON SELECTOR ----------------

def get_icon(desc):

    desc=desc.lower()

    if "clear" in desc:
        return ICONS["clear"]

    if "cloud" in desc:
        return ICONS["cloud"]

    if "rain" in desc:
        return ICONS["drizzle"]

    if "snow" in desc:
        return ICONS["snow"]

    if "mist" in desc or "fog" in desc:
        return ICONS["mist"]

    if "thunder" in desc:
        return ICONS["thunder"]

    return ICONS["clear"]

# ---------------- BACKGROUND ----------------

def set_bg(path):

    st.markdown(
    f"""
    <style>
    .stApp {{
    background-image:url("data:image/png;base64,{encode(path)}");
    background-size:cover;
    }}

    .weather-card {{
    background:#13151f;
    border:1px solid #00ffff;
    border-radius:14px;
    padding:20px;
    text-align:center;
    color:white;
    }}

    .card {{
    background:#13151f;
    border:1px solid #00ffff;
    border-radius:14px;
    padding:20px;
    text-align:center;
    color:white;
    }}

    .ask-btn {{
    position:fixed;
    bottom:30px;
    right:30px;
    background:#00ffff;
    color:black;
    border-radius:50px;
    padding:14px 22px;
    font-weight:bold;
    z-index:999;
    }}

    </style>
    """,
    unsafe_allow_html=True
    )

set_bg("wallpapers/bg1.png")

# ---------------- WEATHER API ----------------

@st.cache_data(ttl=600)
def get_weather(city):

    w=f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    f=f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={API_KEY}"

    return requests.get(w).json(),requests.get(f).json()

# ---------------- ML FORECAST ----------------

def ml_forecast(temps):

    X=np.arange(len(temps)).reshape(-1,1)
    y=np.array(temps)

    model=LinearRegression()
    model.fit(X,y)

    future=np.arange(len(temps),len(temps)+5).reshape(-1,1)

    return model.predict(future)

# ---------------- LLM ----------------

@st.cache_resource
def load_llm():

    model_name="HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

    return tokenizer,model

tokenizer,model=load_llm()

# ---------------- UI ----------------

st.title("WEATHER_SYS_V2")

city=st.text_input("Search city","Bengaluru")

if st.button("Search"):

    weather,forecast=get_weather(city)

    if "weather" in weather:

        st.session_state.weather=weather
        st.session_state.forecast=forecast

# ---------------- DASHBOARD ----------------

if "weather" in st.session_state:

    weather=st.session_state.weather
    forecast=st.session_state.forecast

    desc=weather["weather"][0]["description"]
    temp=weather["main"]["temp"]
    humidity=weather["main"]["humidity"]
    pressure=weather["main"]["pressure"]
    wind=weather["wind"]["speed"]

    st.subheader(weather["name"])

    icon_status=get_icon(desc)

    c1,c2,c3,c4,c5=st.columns(5)

    with c1:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(icon_status)}" width="70"><h4>Status</h4>{desc}</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["temp"])}" width="70"><h4>Temp</h4>{temp}°C</div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["humidity"])}" width="70"><h4>Humidity</h4>{humidity}%</div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["pressure"])}" width="70"><h4>Pressure</h4>{pressure} hPa</div>',unsafe_allow_html=True)

    with c5:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["wind"])}" width="70"><h4>Wind</h4>{wind} m/s</div>',unsafe_allow_html=True)

# ---------------- 5 DAY FORECAST ----------------

    st.subheader("5 Day Forecast")

    cols=st.columns(5)

    days=set()
    i=0

    for item in forecast["list"]:

        dt=datetime.datetime.fromtimestamp(item["dt"])
        day=dt.strftime("%a")

        if day not in days and dt.hour>=12:

            days.add(day)

            t=item["main"]["temp"]

            cols[i].markdown(
                f'<div class="weather-card">{day}<br>{t}°C</div>',
                unsafe_allow_html=True
            )

            i+=1

            if len(days)==5:
                break

# ---------------- HOURLY FORECAST ----------------

    st.subheader("24 Hour Forecast")

    hours=[]
    temps=[]

    for item in forecast["list"][:8]:

        hours.append(datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M"))
        temps.append(item["main"]["temp"])

    df=pd.DataFrame({"Hour":hours,"Temp":temps})

    st.plotly_chart(px.line(df,x="Hour",y="Temp",markers=True),width="stretch")

# ---------------- RADAR ----------------

    st.subheader("Radar Map")

    lat=weather["coord"]["lat"]
    lon=weather["coord"]["lon"]

    m=folium.Map(location=[lat,lon],zoom_start=6)

    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeather"
    ).add_to(m)

    st_folium(m,width=900,height=500)

# ---------------- FLOATING AI BUTTON ----------------

st.markdown('<div class="ask-btn">Ask AI →</div>',unsafe_allow_html=True)

# ---------------- CHAT PANEL ----------------

st.sidebar.title("AI Weather Assistant")

def build_context():

    if "weather" not in st.session_state:
        return "Weather unavailable"

    weather=st.session_state.weather
    forecast=st.session_state.forecast

    desc=weather["weather"][0]["description"]
    temp=weather["main"]["temp"]
    humidity=weather["main"]["humidity"]

    rain=forecast["list"][0].get("pop",0)*100

    return f"""
Temperature: {temp}°C
Condition: {desc}
Humidity: {humidity}%
Rain probability tomorrow: {rain}%
"""

if "messages" not in st.session_state:
    st.session_state.messages=[]

for m in st.session_state.messages:

    with st.sidebar.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:=st.sidebar.chat_input("Ask weather questions"):

    st.session_state.messages.append({"role":"user","content":prompt})

    context=build_context()

    full_prompt=f"""
You are a weather assistant.

Weather data:
{context}

Question: {prompt}

Give helpful advice.
"""

    inputs=tokenizer(full_prompt,return_tensors="pt")

    outputs=model.generate(**inputs,max_new_tokens=100)

    generated=outputs[0][inputs["input_ids"].shape[-1]:]

    response=tokenizer.decode(generated,skip_special_tokens=True)

    st.session_state.messages.append({"role":"assistant","content":response})

    with st.sidebar.chat_message("assistant"):
        st.markdown(response)
