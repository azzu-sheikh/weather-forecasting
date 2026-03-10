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
"rain":"icons/drizzle.png",
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

st.set_page_config(page_title="WEATHER_SYS_V2",layout="wide")

# ---------------- IMAGE HELPERS ----------------

def encode(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def icon_img(path,w=50):
    return f'<img src="data:image/png;base64,{encode(path)}" width="{w}">'

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
    padding:18px;
    text-align:center;
    color:white;
    }}

    .ai-button {{
    position:fixed;
    bottom:25px;
    right:25px;
    background:#00ffff;
    color:black;
    padding:14px 22px;
    border-radius:50px;
    font-weight:bold;
    z-index:9999;
    }}

    .ai-panel {{
    position:fixed;
    top:0;
    right:0;
    width:380px;
    height:100%;
    background:#0f1116;
    border-left:2px solid #00ffff;
    padding:20px;
    overflow-y:auto;
    z-index:9998;
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

@st.cache_data(ttl=600)
def get_aqi(lat,lon):

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    return requests.get(url).json()

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

    name="HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name).to("cpu")

    return tokenizer,model

tokenizer,model=load_llm()

# ---------------- SEARCH ----------------

st.title("WEATHER_SYS_V2")

city=st.text_input("Search City","Bengaluru")

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

    lat=weather["coord"]["lat"]
    lon=weather["coord"]["lon"]

    if "clear" in desc:
        set_bg(WALLPAPERS["clear"])
    elif "cloud" in desc:
        set_bg(WALLPAPERS["cloud"])
    elif "rain" in desc:
        set_bg(WALLPAPERS["rain"])
    else:
        set_bg(WALLPAPERS["default"])

# cards
    c1,c2,c3,c4,c5=st.columns(5)

    with c1:
        st.markdown(f'<div class="weather-card">{icon_img(ICONS["clear"])}<br>Status<br>{desc}</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="weather-card">{icon_img(ICONS["temp"])}<br>{temp}°C</div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="weather-card">{icon_img(ICONS["humidity"])}<br>{humidity}%</div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="weather-card">{icon_img(ICONS["pressure"])}<br>{pressure}</div>',unsafe_allow_html=True)

    with c5:
        st.markdown(f'<div class="weather-card">{icon_img(ICONS["wind"])}<br>{wind} m/s</div>',unsafe_allow_html=True)

# gauges
    g1,g2,g3=st.columns(3)

    fig=go.Figure(go.Indicator(mode="gauge+number",value=wind,title={'text':'Wind Speed'},gauge={'axis':{'range':[0,20]}}))
    g1.plotly_chart(fig,width="stretch")

    fig=go.Figure(go.Indicator(mode="gauge+number",value=5,title={'text':'UV Index'},gauge={'axis':{'range':[0,12]}}))
    g2.plotly_chart(fig,width="stretch")

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
            icon=item["weather"][0]["description"]

            cols[i].markdown(
                f'<div class="weather-card">{icon_img(ICONS["clear"],40)}<br>{day}<br>{t}°C</div>',
                unsafe_allow_html=True
            )

            i+=1

            if len(days)==5:
                break

# hourly
    st.subheader("24 Hour Forecast")

    hours=[]
    temps=[]

    for item in forecast["list"][:8]:

        hour=datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M")
        temp=item["main"]["temp"]

        hours.append(hour)
        temps.append(temp)

    df=pd.DataFrame({"Hour":hours,"Temp":temps})

    st.plotly_chart(px.line(df,x="Hour",y="Temp",markers=True),width="stretch")

# precipitation
    rain=[item.get("pop",0)*100 for item in forecast["list"][:8]]

    df2=pd.DataFrame({"Hour":hours,"Rain%":rain})

    st.subheader("Rain Probability")

    st.plotly_chart(px.bar(df2,x="Hour",y="Rain%"),width="stretch")

# radar
    st.subheader("Radar Map")

    m=folium.Map(location=[lat,lon],zoom_start=6)

    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeather"
    ).add_to(m)

    st_folium(m,width=900,height=500)

# ---------------- FLOATING BUTTON ----------------

if "ai_open" not in st.session_state:
    st.session_state.ai_open=False

if st.button("🤖 Ask AI",key="ai_btn"):
    st.session_state.ai_open=not st.session_state.ai_open

# ---------------- CHAT PANEL ----------------

if st.session_state.ai_open:

    st.markdown('<div class="ai-panel">',unsafe_allow_html=True)

    st.markdown("## AI Weather Assistant")

    def context():

        weather=st.session_state.weather
        forecast=st.session_state.forecast

        desc=weather["weather"][0]["description"]
        temp=weather["main"]["temp"]
        humidity=weather["main"]["humidity"]
        wind=weather["wind"]["speed"]

        rain=forecast["list"][0].get("pop",0)*100

        return f"""
Temperature: {temp}°C
Condition: {desc}
Humidity: {humidity}%
Wind: {wind} m/s
Rain probability tomorrow: {rain}%
"""

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt:=st.chat_input("Ask weather questions..."):

        st.session_state.messages.append({"role":"user","content":prompt})

        prompt_text=f"""
Weather data:
{context()}

User question: {prompt}

Give practical advice.
"""

        inputs=tokenizer(prompt_text,return_tensors="pt")

        outputs=model.generate(**inputs,max_new_tokens=120)

        generated=outputs[0][inputs["input_ids"].shape[-1]:]

        response=tokenizer.decode(generated,skip_special_tokens=True)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role":"assistant","content":response})

    st.markdown("</div>",unsafe_allow_html=True)
