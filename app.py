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
import pydeck as pdk

# ---------------- CONFIG ----------------

API_KEY = "d0f4215f39312e5de368ee8edad554b8"

DEFAULT_BG = "wallpapers/bg1.png"

WALLPAPERS = {
    "clear":"wallpapers/bg2.png",
    "cloud":"wallpapers/bg3.png",
    "rain":"wallpapers/bg6.png",
    "snow":"wallpapers/bg4.png",
    "night":"wallpapers/bg5.png",
    "default":"wallpapers/bg1.png"
}

ICONS = {
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

def get_weather(city):

    weather_url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    forecast_url=f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={API_KEY}"

    weather=requests.get(weather_url).json()
    forecast=requests.get(forecast_url).json()

    return weather,forecast

def get_air_quality(lat,lon):

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    return requests.get(url).json()

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

    now=datetime.datetime.utcnow().timestamp()+timezone

    progress=(now-sunrise)/(sunset-sunrise)

    progress=max(0,min(progress,1))

    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=[0,0.5,1],
        y=[0,1,0],
        mode="lines"
    ))

    x=progress
    y=1-abs(progress-0.5)*2

    icon="☀" if sunrise<=now<=sunset else "🌙"

    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode="text",
        text=[icon],
        textfont=dict(size=40)
    ))

    fig.update_layout(height=260,xaxis_visible=False,yaxis_visible=False)

    return fig

# ---------------- ML FORECAST ----------------

def ml_temp_forecast(temps):

    X=np.arange(len(temps)).reshape(-1,1)
    y=np.array(temps)

    model=LinearRegression()
    model.fit(X,y)

    future=np.arange(len(temps),len(temps)+5).reshape(-1,1)

    return model.predict(future)

def rainfall_forecast(rain):

    X=np.arange(len(rain)).reshape(-1,1)
    y=np.array(rain)

    model=LinearRegression()
    model.fit(X,y)

    future=np.arange(len(rain),len(rain)+5).reshape(-1,1)

    return model.predict(future)

# ---------------- UI ----------------

st.markdown('<div class="title">WEATHER_SYS_V2</div>',unsafe_allow_html=True)

city=st.text_input("LOC","Bengaluru")

search=st.button("Search")

if search:

    weather,forecast=get_weather(city)

    if "weather" not in weather:
        st.error(weather.get("message"))
        st.stop()

    desc=weather["weather"][0]["description"]
    temp=weather["main"]["temp"]
    humidity=weather["main"]["humidity"]
    pressure=weather["main"]["pressure"]
    wind=weather["wind"]["speed"]
    sunrise=weather["sys"]["sunrise"]
    sunset=weather["sys"]["sunset"]
    timezone=weather["timezone"]
    wind_deg=weather["wind"].get("deg",0)

    lat=weather["coord"]["lat"]
    lon=weather["coord"]["lon"]

    bg=choose_background(desc)
    set_background(bg)

    icon_status=get_icon(desc)
    icon_temp=ICONS["high_temp"] if temp>18 else ICONS["low_temp"]

    st.subheader(f"// {city.upper()}")

# ---------------- WEATHER CARDS ----------------

    c1,c2,c3,c4,c5=st.columns(5)

    with c1:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(icon_status)}" width="70"><h4>Status</h4>{desc}</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(icon_temp)}" width="70"><h4>Temp</h4>{temp}°C</div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["humidity"])}" width="70"><h4>Humidity</h4>{humidity}%</div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["pressure"])}" width="70"><h4>Pressure</h4>{pressure} hPa</div>',unsafe_allow_html=True)

    with c5:
        st.markdown(f'<div class="card"><img src="data:image/png;base64,{encode_image(ICONS["wind"])}" width="70"><h4>Wind</h4>{wind} m/s</div>',unsafe_allow_html=True)

# ---------------- GAUGES ----------------

    g1,g2,g3=st.columns(3)

    g1.plotly_chart(wind_gauge(wind),use_container_width=True)
    g2.plotly_chart(uv_meter(),use_container_width=True)
    g3.plotly_chart(sun_moon_gauge(sunrise,sunset,timezone),use_container_width=True)

# ---------------- HOURLY TEMP ----------------

    hours=[]
    temps=[]
    rain=[]

    for item in forecast["list"][:12]:

        hours.append(datetime.datetime.fromtimestamp(item["dt"]).strftime("%H:%M"))
        temps.append(item["main"]["temp"])
        rain.append(item.get("pop",0)*100)

    df=pd.DataFrame({"Hour":hours,"Temp":temps})

    st.subheader("Hourly Temperature")

    fig=px.line(df,x="Hour",y="Temp",markers=True)

    st.plotly_chart(fig,use_container_width=True)

# ---------------- PRECIPITATION ----------------

    df2=pd.DataFrame({"Time":hours,"Rain%":rain})

    st.subheader("Precipitation Probability")

    fig2=px.bar(df2,x="Time",y="Rain%")

    st.plotly_chart(fig2,use_container_width=True)

# ---------------- ML FORECAST ----------------

    pred=ml_temp_forecast(temps)

    pred_df=pd.DataFrame({"Future":[f"+{i+1}h" for i in range(len(pred))],"Temp":pred})

    st.subheader("ML Temperature Forecast")

    st.line_chart(pred_df.set_index("Future"))

# ---------------- AI RAINFALL ----------------

    rain_pred=rainfall_forecast(rain)

    rain_df=pd.DataFrame({"Future":[f"+{i+1}h" for i in range(len(rain_pred))],"Rain%":rain_pred})

    st.subheader("AI Rainfall Forecast")

    st.line_chart(rain_df.set_index("Future"))

# ---------------- AQI ----------------

    air=get_air_quality(lat,lon)

    aqi=air["list"][0]["main"]["aqi"]

    st.metric("Air Quality Index",aqi)

# ---------------- WEATHER MAP ----------------

    st.subheader("Weather Map")

    st.map(pd.DataFrame({"lat":[lat],"lon":[lon]}))

# ---------------- RADAR MAP ----------------

    st.subheader("Storm Radar")

    m=folium.Map(location=[lat,lon],zoom_start=6)

    folium.TileLayer(
        tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={API_KEY}",
        attr="OpenWeatherMap"
    ).add_to(m)

    st_folium(m,width=900,height=500)

# ---------------- 3D WEATHER GLOBE ----------------

    st.subheader("3D Weather Globe")

    globe=pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-v9",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=1,
            pitch=45
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat":lat,"lon":lon}],
                get_position="[lon, lat]",
                get_radius=50000,
                get_color="[255,0,0]"
            )
        ]
    )

    st.pydeck_chart(globe)