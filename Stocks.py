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
# Basic Stock price data
""")

tickerSymbol1 = "GOOGL"
tickerSymbol2 = "AAPL"

tickerData1 = yf.Ticker(tickerSymbol1)
tickerDf1 = tickerData1.history(period="Id", start="2010-5-31", end="2021-5-31")

tickerData2 = yf.Ticker(tickerSymbol2)
tickerDf2 = tickerData2.history(period="Id", start="2010-5-31", end="2021-5-31")

st.write("""
##Google
""" )
st.bar_chart(tickerDf1.Close)
st.line_chart(tickerDf1.Volume)

st.write("""
##Apple
""" )
st.bar_chart(tickerDf2.Close)
st.line_chart(tickerDf2.Volume)