import streamlit as st
import requests
import datetime
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import st_folium

# ---------------- CONFIG ----------------

API_KEY="d0f4215f39312e5de368ee8edad554b8"

DEFAULT_BG="wallpapers/bg1.png"

WALLPAPERS={
"clear":"wallpapers/bg2.png",
"cloud":"wallpapers/bg3.png",
"rain":"wallpapers/bg6.png",
"snow":"wallpapers/bg4.png",
"night":"wallpapers/bg5.png",
"default":"wallpapers/bg1.png"
}

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
"high_temp":"icons/high_temp.png",
"low_temp":"icons/low_temp.png"
}

ACCENT="#00ffff"
CARD_COLOR="#13151f"

st.set_page_config(page_title="WEATHER_SYS_V2",layout="wide")

# ---------------- IMAGE ENCODER ----------------

def encode_image(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------- BACKGROUND ----------------

def set_background(path):
    encoded=encode_image(path)

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:url("data:image/png;base64,{encoded}");
        background-size:cover;
        background-position:center;
    }}
    .card {{
        background:{CARD_COLOR};
        border:1px solid {ACCENT};
        border-radius:16px;
        padding:20px;
        text-align:center;
        color:white;
    }}
    .title {{
        text-align:center;
        font-size:42px;
        color:{ACCENT};
        font-weight:bold;
    }}
    </style>
    """,unsafe_allow_html=True)

set_background(DEFAULT_BG)

# ---------------- API ----------------

@st.cache_data(ttl=600)
def get_weather(city):

    weather_url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    forecast_url=f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={API_KEY}"

    weather=requests.get(weather_url).json()
    forecast=requests.get(forecast_url).json()

    return weather,forecast

# ---------------- HELPERS ----------------

def choose_background(desc):

    desc=desc.lower()

    if "clear" in desc:
        return WALLPAPERS["clear"]

    if "cloud" in desc:
        return WALLPAPERS["cloud"]

    if "rain" in desc:
        return WALLPAPERS["rain"]

    if "snow" in desc:
        return WALLPAPERS["snow"]

    return WALLPAPERS["default"]

def get_icon(desc):

    desc=desc.lower()

    if "thunder" in desc:
        return ICONS["thunder"]

    if "snow" in desc:
        return ICONS["snow"]

    if "rain" in desc:
        return ICONS["drizzle"]

    if "cloud" in desc:
        return ICONS["cloud"]

    if "mist" in desc:
        return ICONS["mist"]

    return ICONS["clear"]

# ---------------- GAUGES ----------------

def wind_gauge(speed):

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=speed,
        title={"text":"Wind Speed"},
        gauge={"axis":{"range":[0,20]}}
    ))

    fig.update_layout(height=260)

    return fig

def uv_meter():

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=5,
        title={"text":"UV Index"},
        gauge={"axis":{"range":[0,12]}}
    ))

    fig.update_layout(height=260)

    return fig

def sun_moon_gauge(sunrise,sunset,timezone):

    now=datetime.datetime.now(datetime.UTC).timestamp()+timezone

    progress=(now-sunrise)/(sunset-sunrise)
    progress=max(0,min(progress,1))

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=[0,0.5,1],y=[0,1,0],mode="lines"))

    x=progress
    y=1-abs(progress-0.5)*2

    icon="☀" if sunrise<=now<=sunset else "🌙"

    fig.add_trace(go.Scatter(x=[x],y=[y],mode="text",text=[icon],textfont=dict(size=40)))

    fig.update_layout(height=260,xaxis_visible=False,yaxis_visible=False)

    return fig

# ---------------- UI ----------------

st.markdown('<div class="title">WEATHER_SYS_V2</div>',unsafe_allow_html=True)

city=st.text_input("LOC","Bengaluru")

if st.button("Search"):

    weather,forecast=get_weather(city)

    if "weather" in weather:

        st.session_state.weather=weather
        st.session_state.forecast=forecast
        st.session_state.city=city

# ---------------- DISPLAY ----------------

if "weather" in st.session_state:

    weather=st.session_state.weather
    forecast=st.session_state.forecast
    city=st.session_state.city

    desc=weather["weather"][0]["description"]
    temp=weather["main"]["temp"]
    humidity=weather["main"]["humidity"]
    pressure=weather["main"]["pressure"]
    wind=weather["wind"]["speed"]

    sunrise=weather["sys"]["sunrise"]
    sunset=weather["sys"]["sunset"]
    timezone=weather["timezone"]

    lat=weather["coord"]["lat"]
    lon=weather["coord"]["lon"]

    set_background(choose_background(desc))

    icon_status=get_icon(desc)

    st.subheader(f"// {city.upper()}")

# ---------------- WEATHER CARDS ----------------

    c1,c2,c3,c4,c5=st.columns(5)

    with c1:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(icon_status)}" width="70"><h4>Status</h4>{desc}</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["high_temp"])}" width="70"><h4>Temp</h4>{temp}°C</div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["humidity"])}" width="70"><h4>Humidity</h4>{humidity}%</div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["pressure"])}" width="70"><h4>Pressure</h4>{pressure} hPa</div>',unsafe_allow_html=True)

    with c5:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["wind"])}" width="70"><h4>Wind</h4>{wind} m/s</div>',unsafe_allow_html=True)

# ---------------- GAUGES ----------------

    g1,g2,g3=st.columns(3)

    g1.plotly_chart(wind_gauge(wind),width="stretch")
    g2.plotly_chart(uv_meter(),width="stretch")
    g3.plotly_chart(sun_moon_gauge(sunrise,sunset,timezone),width="stretch")

# ---------------- 5 DAY FORECAST ----------------

    st.subheader("5 Day Forecast")

    days=set()
    cols=st.columns(5)
    i=0

    for item in forecast["list"]:

        dt=datetime.datetime.fromtimestamp(item["dt"])
        day=dt.strftime("%a")

        if day not in days and dt.hour>=12:

            days.add(day)

            t=item["main"]["temp"]
            icon=get_icon(item["weather"][0]["description"])

            cols[i].markdown(
                f'<div class="card"><img src="data:image/png;base64,{encode_image(icon)}" width="40"><br>{day}<br>{t}°C</div>',
                unsafe_allow_html=True
            )

            i+=1

            if len(days)==5:
                break

# ---------------- FULL DAY FORECAST ----------------

    st.subheader("24 Hour Forecast")

    hours=[]
    temps=[]

    for item in forecast["list"][:8]:

        hours.append(datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M"))
        temps.append(item["main"]["temp"])

    df=pd.DataFrame({"Hour":hours,"Temp":temps})

    st.plotly_chart(px.line(df,x="Hour",y="Temp",markers=True),width="stretch")
