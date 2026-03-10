import streamlit as st
import requests
import datetime
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import folium
from streamlit_folium import st_folium

API_KEY="d0f4215f39312e5de368ee8edad554b8"

st.set_page_config(page_title="WEATHER_SYS_V2",layout="wide")

# ---------------- ICONS ----------------

ICONS={
"clear":"icons/clear.png",
"cloud":"icons/clouds.png",
"rain":"icons/drizzle.png",
"snow":"icons/snow.png",
"mist":"icons/mist.png",
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

# ---------------- IMAGE UTILS ----------------

def img64(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def icon(path,w=50):
    return f'<img src="data:image/png;base64,{img64(path)}" width="{w}">'

# ---------------- BACKGROUND ----------------

def set_bg(path):

    st.markdown(f"""
    <style>
    .stApp {{
    background-image:url("data:image/png;base64,{img64(path)}");
    background-size:cover;
    }}

    .card {{
    background:#13151f;
    border:1px solid #00ffff;
    border-radius:14px;
    padding:18px;
    text-align:center;
    color:white;
    }}

    .ai-panel {{
    position:fixed;
    top:0;
    right:0;
    width:360px;
    height:100%;
    background:#0f1116;
    border-left:2px solid #00ffff;
    padding:20px;
    overflow-y:auto;
    z-index:9999;
    }}
    </style>
    """,unsafe_allow_html=True)

set_bg("wallpapers/bg1.png")

# ---------------- API ----------------

@st.cache_data(ttl=600)
def get_weather(city):

    weather=f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    forecast=f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={API_KEY}"

    return requests.get(weather).json(),requests.get(forecast).json()

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
def load_model():

    name="HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name).to("cpu")

    return tokenizer,model

tokenizer,model=load_model()

# ---------------- SEARCH ----------------

st.title("WEATHER_SYS_V2")

city=st.text_input("Search City","Bangalore")

if st.button("Search"):

    weather,forecast=get_weather(city)

    if "weather" in weather:

        st.session_state.weather=weather
        st.session_state.forecast=forecast
        st.session_state.messages=[]

# ---------------- DASHBOARD ----------------

if "weather" in st.session_state:

    weather=st.session_state.weather
    forecast=st.session_state.forecast

    desc=weather["weather"][0]["description"]
    temp=weather["main"]["temp"]
    humidity=weather["main"]["humidity"]
    pressure=weather["main"]["pressure"]
    wind=weather["wind"]["speed"]

# background
    if "clear" in desc:
        set_bg(WALLPAPERS["clear"])
    elif "cloud" in desc:
        set_bg(WALLPAPERS["cloud"])
    elif "rain" in desc:
        set_bg(WALLPAPERS["rain"])
    else:
        set_bg(WALLPAPERS["default"])

# weather cards
    c1,c2,c3,c4,c5=st.columns(5)

    c1.markdown(f'<div class="card">{icon(ICONS["clear"])}<br>{desc}</div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="card">{icon(ICONS["temp"])}<br>{temp}°C</div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="card">{icon(ICONS["humidity"])}<br>{humidity}%</div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="card">{icon(ICONS["pressure"])}<br>{pressure}</div>',unsafe_allow_html=True)
    c5.markdown(f'<div class="card">{icon(ICONS["wind"])}<br>{wind} m/s</div>',unsafe_allow_html=True)

# wind gauge
    g1,g2,g3=st.columns(3)

    fig=go.Figure(go.Indicator(mode="gauge+number",value=wind,title={'text':'Wind'},gauge={'axis':{'range':[0,20]}}))
    g1.plotly_chart(fig,width="stretch")

# uv meter
    fig=go.Figure(go.Indicator(mode="gauge+number",value=5,title={'text':'UV Index'},gauge={'axis':{'range':[0,12]}}))
    g2.plotly_chart(fig,width="stretch")

# sun arc
    sunrise=weather["sys"]["sunrise"]
    sunset=weather["sys"]["sunset"]
    timezone=weather["timezone"]

    now=datetime.datetime.now(datetime.UTC).timestamp()+timezone
    progress=(now-sunrise)/(sunset-sunrise)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,0.5,1],y=[0,1,0],mode="lines"))
    fig.add_trace(go.Scatter(x=[progress],y=[0.5],mode="text",text=["☀"],textfont=dict(size=40)))
    fig.update_layout(height=250,xaxis_visible=False,yaxis_visible=False)

    g3.plotly_chart(fig,width="stretch")

# 5 day forecast
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
            f'<div class="card">{icon(ICONS["clear"],40)}<br>{day}<br>{t}°C</div>',
            unsafe_allow_html=True)

            i+=1

            if len(days)==5:
                break

# hourly
    st.subheader("24 Hour Forecast")

    hours=[]
    temps=[]

    for item in forecast["list"][:8]:

        hour=datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M")
        t=item["main"]["temp"]

        hours.append(hour)
        temps.append(t)

    df=pd.DataFrame({"Hour":hours,"Temp":temps})

    st.plotly_chart(px.line(df,x="Hour",y="Temp",markers=True),width="stretch")

# precipitation
    rain=[item.get("pop",0)*100 for item in forecast["list"][:8]]

    st.subheader("Rain Probability")

    df2=pd.DataFrame({"Hour":hours,"Rain":rain})

    st.plotly_chart(px.bar(df2,x="Hour",y="Rain"),width="stretch")

# radar
    lat=weather["coord"]["lat"]
    lon=weather["coord"]["lon"]

    st.subheader("Radar Map")

    m=folium.Map(location=[lat,lon],zoom_start=6)

    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeather"
    ).add_to(m)

    st_folium(m,width=900,height=500)

# ---------------- AI CHAT ----------------

if "ai_open" not in st.session_state:
    st.session_state.ai_open=False

if st.button("Ask AI"):
    st.session_state.ai_open=not st.session_state.ai_open

def context():

    w=st.session_state.weather
    f=st.session_state.forecast

    desc=w["weather"][0]["description"]
    temp=w["main"]["temp"]
    humidity=w["main"]["humidity"]
    wind=w["wind"]["speed"]
    rain=f["list"][0].get("pop",0)*100

    return f"""
Temperature: {temp}°C
Condition: {desc}
Humidity: {humidity}%
Wind: {wind} m/s
Rain tomorrow: {rain}%
"""

if st.session_state.ai_open:

    st.markdown('<div class="ai-panel">',unsafe_allow_html=True)

    st.markdown("### AI Weather Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt:=st.chat_input("Ask weather questions"):

        st.session_state.messages.append({"role":"user","content":prompt})

        text=f"""
You are a helpful weather assistant.

Weather data:
{context()}

Question: {prompt}

Answer briefly with advice.
"""

        inputs=tokenizer(text,return_tensors="pt")

        outputs=model.generate(**inputs,max_new_tokens=80)

        gen=outputs[0][inputs["input_ids"].shape[-1]:]

        reply=tokenizer.decode(gen,skip_special_tokens=True)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role":"assistant","content":reply})

    st.markdown("</div>",unsafe_allow_html=True)
