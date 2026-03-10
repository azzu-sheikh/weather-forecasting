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
"high_temp":"icons/high_temp.png"
}

ACCENT="#00ffff"
CARD="#13151f"

st.set_page_config(page_title="WEATHER_SYS_V2",layout="wide")

# ---------------- IMAGE ENCODER ----------------

def encode(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------- BACKGROUND ----------------

def set_bg(path):

    img=encode(path)

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:url("data:image/png;base64,{img}");
        background-size:cover;
    }}

    .card {{
        background:{CARD};
        border:1px solid {ACCENT};
        border-radius:15px;
        padding:20px;
        text-align:center;
        color:white;
    }}

    .title {{
        text-align:center;
        font-size:40px;
        color:{ACCENT};
        font-weight:bold;
    }}
    </style>
    """,unsafe_allow_html=True)

set_bg(DEFAULT_BG)

# ---------------- WEATHER API ----------------

@st.cache_data(ttl=600)
def get_weather(city):

    weather_url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    forecast_url=f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={API_KEY}"

    return requests.get(weather_url).json(),requests.get(forecast_url).json()

@st.cache_data(ttl=600)
def get_aqi(lat,lon):

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    return requests.get(url).json()

# ---------------- HELPERS ----------------

def bg_from_weather(desc):

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

def icon(desc):

    desc=desc.lower()

    if "rain" in desc:
        return ICONS["drizzle"]

    if "cloud" in desc:
        return ICONS["cloud"]

    if "snow" in desc:
        return ICONS["snow"]

    if "mist" in desc:
        return ICONS["mist"]

    return ICONS["clear"]

# ---------------- GAUGES ----------------

def wind_gauge(speed):

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=speed,
        title={"text":"Wind"},
        gauge={"axis":{"range":[0,20]}}
    ))

    fig.update_layout(height=250)

    return fig

def uv_meter():

    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=5,
        title={"text":"UV Index"},
        gauge={"axis":{"range":[0,12]}}
    ))

    fig.update_layout(height=250)

    return fig

def sun_arc(sunrise,sunset,timezone):

    now=datetime.datetime.now(datetime.UTC).timestamp()+timezone

    progress=(now-sunrise)/(sunset-sunrise)

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=[0,0.5,1],y=[0,1,0],mode="lines"))

    x=progress
    y=1-abs(progress-0.5)*2

    icon="☀" if sunrise<=now<=sunset else "🌙"

    fig.add_trace(go.Scatter(x=[x],y=[y],mode="text",text=[icon],textfont=dict(size=40)))

    fig.update_layout(height=250,xaxis_visible=False,yaxis_visible=False)

    return fig

# ---------------- ML FORECAST ----------------

def ml_forecast(temps):

    X=np.arange(len(temps)).reshape(-1,1)
    y=np.array(temps)

    model=LinearRegression()
    model.fit(X,y)

    future=np.arange(len(temps),len(temps)+5).reshape(-1,1)

    return model.predict(future)

# ---------------- LLM MODEL ----------------

@st.cache_resource
def load_llm():

    checkpoint="HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(checkpoint)

    model=AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32
    ).to("cpu")

    return tokenizer,model

tokenizer,model=load_llm()

# ---------------- UI ----------------

st.markdown('<div class="title">WEATHER_SYS_V2</div>',unsafe_allow_html=True)

city=st.text_input("Search City","Bengaluru")

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

    set_bg(bg_from_weather(desc))

    st.subheader(city.upper())

# ---------------- WEATHER CARDS ----------------

    c1,c2,c3,c4,c5=st.columns(5)

    with c1:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(icon(desc))}" width="60"><br>Status<br>{desc}</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["high_temp"])}" width="60"><br>Temp<br>{temp}°C</div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["humidity"])}" width="60"><br>Humidity<br>{humidity}%</div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["pressure"])}" width="60"><br>Pressure<br>{pressure}</div>',unsafe_allow_html=True)

    with c5:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode(ICONS["wind"])}" width="60"><br>Wind<br>{wind}</div>',unsafe_allow_html=True)

# ---------------- GAUGES ----------------

    g1,g2,g3=st.columns(3)

    g1.plotly_chart(wind_gauge(wind),width="stretch")
    g2.plotly_chart(uv_meter(),width="stretch")
    g3.plotly_chart(sun_arc(sunrise,sunset,timezone),width="stretch")

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

            cols[i].markdown(f'<div class="card">{day}<br>{t}°C</div>',unsafe_allow_html=True)

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

# ---------------- PRECIPITATION ----------------

    rain=[item.get("pop",0)*100 for item in forecast["list"][:8]]

    df2=pd.DataFrame({"Hour":hours,"Rain%":rain})

    st.subheader("Rain Probability")

    st.plotly_chart(px.bar(df2,x="Hour",y="Rain%"),width="stretch")

# ---------------- AI WEATHER ASSISTANT ----------------

st.divider()
st.header("AI Weather Assistant")

def build_context():

    if "weather" not in st.session_state:
        return "Weather data not loaded."

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

    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:=st.chat_input("Ask about weather..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context=build_context()

    full_prompt=f"""
You are a weather assistant.

Weather data:
{context}

User question: {prompt}

Give helpful advice.
"""

    inputs=tokenizer(full_prompt,return_tensors="pt")

    outputs=model.generate(**inputs,max_new_tokens=120)

    response=tokenizer.decode(outputs[0],skip_special_tokens=True)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant","content":response})
