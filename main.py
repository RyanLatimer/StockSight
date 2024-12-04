import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#Create a function to plot the historical data to ease readability in if then statements
def history_data_plot(ticker, time_period):

    plt.figure(figsize=(10,6)) #Set the window size
    plt.plot(time_period.index, time_period['Close'], label=f'{ticker} Close Price', color='b')

    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"{ticker} Historical Stock Price For 1 Month")

    plt.xticks(rotation=45) # Rotate x-ticks to avoid overlap
    plt.tight_layout()
    plt.legend() # Create a legend
    plt.show()

#Create a function to gather the user input
def gather_user_input():

    #Gather The Ticker from the User
    user_ticker_input = input("Enter the desired Ticker to Track:  ")

    #Create a ticker object to pull the ticker form Yahoo Finance
    ticker = yf.Ticker(user_ticker_input)

    #Ask the user the Length of Time they would like to view data from
    user_data_period_length = input("Enter the desired amount of time to view (1m, 3m, 6m, 1y, 3y, 5y, max:  ")

    return [ticker, user_data_period_length]

#Define a function to determine what data to plot
def plot_historical_data_based_on_user_input(ticker, user_data_period_length):
    if user_data_period_length is not None:
        if user_data_period_length == "1m":
            #Gather data for entered time period
            one_month_history = ticker.history(period="1mo")

            #Plot the data
            history_data_plot(ticker, one_month_history)

        elif user_data_period_length == "3m":
            #Gather the data for the time period
            three_month_history = ticker.history(period="3mo")

            #Plot the data
            history_data_plot(ticker, three_month_history)

        elif user_data_period_length == "6m":
            #Gather the data for the time period
            six_month_history = ticker.history(period="6mo")

            #Plot the Data
            history_data_plot(ticker, six_month_history)

        elif user_data_period_length == "1y":
            #Gather the data for the time period
            one_year_history = ticker.history(period="1y")

            #Plot the data
            history_data_plot(ticker, one_year_history)


if __name__ == "__main__":
    #Gather user data and define variables gathered from the function:
    user_data = gather_user_input()

    #Assign sections of the array to specific variables
    ticker = user_data[0]
    print(ticker)
    user_data_period_length = user_data[1]

    #Call the function to plot the data
    plot_historical_data_based_on_user_input(ticker, user_data_period_length)