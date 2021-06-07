#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 19:55:14 2021

@author: hrishikeshsrinivasan07
"""

import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App
Shown are the stock closing price and colume of Google!
""")

tickerSymbol = "GOOGL"

tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period="Id", start="2010-5-31", end="2021-5-31")

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)