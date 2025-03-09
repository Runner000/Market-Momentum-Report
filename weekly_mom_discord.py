import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2_contingency
import warnings, datetime, os, pandas as pd, numpy as np, discord, datetime as dt
from binance import Client
from parameters import discord_token as TOKEN
client = discord.Client(intents = discord.Intents.default())

@client.event
async def on_ready():

    async def send_plot(plt_title):
        plotname = f"{plt_title}.png"
        file = discord.File(f"{path}/{plotname}", filename=plotname)
        embed = discord.Embed()
        embed.set_image(url=f"attachment://{plotname}")
        await channel.send(file=file, embed=embed)

    # Ignore specific matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

    async def calculate_and_plot_z_scores(data, start_year_to_display=None, end_year_to_display=None, data_frequency='daily'):
        # plt.style.use()
        # New Analysis Section
        await channel.send(f"Z-Score Analysis for {data_frequency.capitalize()} Data:")

        # Calculate average close price for each day
        all_close_prices = [df.set_index('date')['close'] for df in data.values()]
        avg_close = pd.concat(all_close_prices, axis=1).mean(axis=1)

        # Ensure that index is in datetime format
        avg_close.index = pd.to_datetime(avg_close.index)

        # Calculate SMAs
        avg_close_df = pd.DataFrame(avg_close, columns=['avg_close'])
        avg_close_df['20_sma'] = avg_close_df['avg_close'].rolling(window=20).mean()
        avg_close_df['50_sma'] = avg_close_df['avg_close'].rolling(window=50).mean()
        avg_close_df['200_sma'] = avg_close_df['avg_close'].rolling(window=200).mean()

        # Compute % deviation from the SMAs
        avg_close_df['dev_20'] = ((avg_close_df['avg_close'] - avg_close_df['20_sma']) / avg_close_df['20_sma']) * 100
        avg_close_df['dev_50'] = ((avg_close_df['avg_close'] - avg_close_df['50_sma']) / avg_close_df['50_sma']) * 100
        avg_close_df['dev_200'] = ((avg_close_df['avg_close'] - avg_close_df['200_sma']) / avg_close_df['200_sma']) * 100

        # Calculate Z-scores for these deviations
        avg_close_df['zscore_20'] = (avg_close_df['dev_20'] - avg_close_df['dev_20'].rolling(window=20).mean()) / avg_close_df['dev_20'].rolling(window=20).std()
        avg_close_df['zscore_50'] = (avg_close_df['dev_50'] - avg_close_df['dev_50'].rolling(window=20).mean()) / avg_close_df['dev_50'].rolling(window=20).std()
        avg_close_df['zscore_200'] = (avg_close_df['dev_200'] - avg_close_df['dev_200'].rolling(window=20).mean()) / avg_close_df['dev_200'].rolling(window=20).std()

        # Filter data if year_to_display is specified
        if start_year_to_display:
            start_date = f"{start_year_to_display}-01-01"
            end_date = f"{end_year_to_display}-12-31"
            avg_close_df = avg_close_df[(avg_close_df.index >= pd.to_datetime(start_date)) & (avg_close_df.index <= pd.to_datetime(end_date))]
        
        # 5. Plot the Z-scores on separate subplots with x-axis date formatting
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Format dates for x-axis
        date_format = mdates.DateFormatter('%Y-%m')
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])  # These months represent quarters
        
        # Print the chart titles
        chart_titles = [
            'Bitcoin and Altcoins - Rolling Normalized Deviation from 20-Moving Average',
            'Bitcoin and Altcoins - Rolling Normalized Deviation from 50-Moving Average',
            'Bitcoin and Altcoins - Rolling Normalized Deviation from 200-Moving Average'
        ]
        
        # Calculate the current and average Z-Scores
        current_zscores = avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].iloc[-1]
        z_score_averages = {days: avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].tail(days).mean() for days in [5, 15, 30]}
        
        # Print the chart titles along with their current Z-Scores and averages with two decimal places
        for i, ax in enumerate(axes):
            title = chart_titles[i]
            current_zscore = current_zscores.iloc[i]
            avg_5d = z_score_averages[5].iloc[i]
            avg_15d = z_score_averages[15].iloc[i]
            avg_30d = z_score_averages[30].iloc[i]
        
            # Set the title for the chart
            ax.set_title(title)
        
            # Print the title with the formatted current Z-score and averages
            await channel.send(f"{title}\n"
                               f"Current Z-Score: {current_zscore:.2f}\n"
                               f"Average Z-Score Last 5 Days: {avg_5d:.2f}\n"
                               f"Average Z-Score Last 15 Days: {avg_15d:.2f}\n"
                               f"Average Z-Score Last 30 Days: {avg_30d:.2f}\n"
                               )

        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(date_format)
        
        axes[0].plot(avg_close_df.index, avg_close_df['zscore_20'], color='blue')
        axes[0].axhline(0, color='black', linewidth=0.5)
        axes[0].set_title('Bitcoin and Altcoins - Rolling Normalized Deviation from 20-Moving Average')
        axes[0].grid(True)

        axes[1].plot(avg_close_df.index, avg_close_df['zscore_50'], color='green')
        axes[1].axhline(0, color='black', linewidth=0.5)
        axes[1].set_title('Bitcoin and Altcoins - Rolling Normalized Deviation from 50-Moving Average')
        axes[1].grid(True)
        
        axes[2].plot(avg_close_df.index, avg_close_df['zscore_200'], color='red')
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].set_title('Bitcoin and Altcoins - Rolling Normalized Deviation from 200-Moving Average')
        axes[2].grid(True)
        
        # Print Current Z-Scores
        current_zscores = avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].iloc[-1]

        # Adding current z-score values as horizontal lines in the plots
        axes[0].axhline(current_zscores['zscore_20'], color='purple', linestyle='--', label='Current Z-Score 20')
        axes[1].axhline(current_zscores['zscore_50'], color='purple', linestyle='--', label='Current Z-Score 50')
        axes[2].axhline(current_zscores['zscore_200'], color='purple', linestyle='--', label='Current Z-Score 200')
        
        fig.autofmt_xdate()  # Rotates the dates for better formatting
        plt.tight_layout()
        plt_title = 'Bitcoin and Altcoins - Rolling Normalized Deviation from (20,50,200)-Moving Average'
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # 6. Simulation of returns
        holding_days = [3, 5, 15]
        SMAs = ['20', '50', '200']
        
        # Separating out simulated returns based on SMA strategy
        simulated_returns = {sma: {} for sma in SMAs}
        
        for sma in SMAs:
            for hold in holding_days:
                condition = avg_close_df[f'zscore_{sma}'] < -2
                avg_close_df[f'return_{sma}_{hold}'] = np.where(
                    condition,
                    avg_close_df['avg_close'].shift(-hold) / avg_close_df['avg_close'] - 1,
                    0
                )
                simulated_returns[sma][hold] = (avg_close_df[f'return_{sma}_{hold}'] + 1).cumprod()
        
        # Plot Simulated Returns
        fig, axes = plt.subplots(len(SMAs), 1, figsize=(15, 16))
        
        date_format = mdates.DateFormatter('%Y-%m')
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        
        for i, sma in enumerate(SMAs):
            ax = axes[i]
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(date_format)
            
            for hold, ret in simulated_returns[sma].items():
                ax.plot(ret.index, ret, label=f'Holding {hold} days')
            
            ax.set_title(f'Simulated Returns for {sma} SMA Z-score > 2 strategy')
            ax.legend()
            ax.grid(True)
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt_title =  'Simulated Returns for (20,50,200) SMA Z-score more than 2 strategy'
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
            
        # Historical Context
        historical_extremes = avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].agg(['min', 'max'])
        
        # Statistical Summary
        zscore_summary = avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].describe()
        
        # Current Z-Score Values
        current_zscores = avg_close_df[['zscore_20', 'zscore_50', 'zscore_200']].iloc[-1]
        
        # Basic Interpretation with Plot Titles
        extreme_threshold = 2
        plot_titles = [
            'Rolling Normalized Deviation from 20-Moving Average',
            'Rolling Normalized Deviation from 50-Moving Average',
            'Rolling Normalized Deviation from 200-Moving Average'
        ]
        
        # Plot for the entire dataset
        plt.figure(figsize=(15, 6))
        plt.plot(avg_close_df.index, avg_close_df['avg_close'], color='black')
        plt_title = 'Average Close Price with Bear Deviations (Entire Dataset)'
        plt.title(plt_title)
        plt.grid(True)
        # Adding vertical lines for Z-score deviations
        for idx, row in avg_close_df.iterrows():
            if row['zscore_200'] > 2:
                plt.axvline(x=idx, color='red', linestyle='--', alpha=0.7)  # Red for zscore_200
        plt.xlabel('Date')
        plt.ylabel('Average Close Price')
        plt.legend()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Plot for the entire dataset
        plt.figure(figsize=(15, 6))
        plt.plot(avg_close_df.index, avg_close_df['avg_close'], color='black')
        plt_title = 'Average Close Price with Bull Deviations (Entire Dataset)'
        plt.title(plt_title)
        plt.grid(True)
        # Adding vertical lines for Z-score deviations
        for idx, row in avg_close_df.iterrows():
            if row['zscore_200'] < -2:
                plt.axvline(x=idx, color='green', linestyle='--', alpha=0.7)  # Red for zscore_200
        plt.xlabel('Date')
        plt.ylabel('Average Close Price')
        plt.legend()
        plt.savefig(f'{path}\{plt_title}')
        await send_plot(plt_title)

    async def momentum_strategy_returns(data, window_size=90):
        """
        Calculates the cumulative returns of a momentum strategy based on average z-scores.

        Parameters:
        - data: Dictionary where keys are asset names and values are DataFrames with price data.
        - window_size: Number of days for the rolling window.

        Returns:
        - DataFrame with dates, strategy returns, and buy/sell signals.
        """

        # 1. Calculate the daily return for each asset
        for asset, df in data.items():
            df['return'] = df['close'].pct_change()

        # 2. Compute the rolling z-score for each asset's returns
        for asset, df in data.items():
            rolling_mean = df['return'].rolling(window=window_size).mean()
            rolling_std = df['return'].rolling(window=window_size).std()
            df['z_score'] = (df['return'] - rolling_mean) / rolling_std

        # 3. Calculate the equally-weighted average z-score and return for each day
        combined_df = pd.concat(data.values(), axis=0).reset_index(drop=True)
        average_df = combined_df.groupby('date').agg({'z_score':'mean', 'return':'mean'}).reset_index()
        
        # Compute the 20-bar average of the z-scores
        average_df['z_score_moving_avg'] = average_df['z_score'].rolling(window=20).mean()

        # Remove returns in the top 5% (95th percentile)
        cutoff = average_df['return'].quantile(0.99)
        average_df.loc[average_df['return'] > cutoff, 'return'] = 0

        # 4. Generate buy/sell signals based on z-score
        average_df['long_position'] = np.where(average_df['z_score_moving_avg'] > 0, 1, 0)  # 1 for long
        average_df['short_position'] = np.where(average_df['z_score_moving_avg'] < 0, -1, 0)  # -1 for short

        # 5. Calculate the strategy's daily return
        average_df['long_strategy_return'] = average_df['long_position'] * average_df['return']
        average_df['short_strategy_return'] = - average_df['short_position'] * average_df['return']

        # 6. Compute the cumulative returns for visualization
        average_df['long_cumulative_return'] = (average_df['long_strategy_return'].fillna(0)).cumsum()
        average_df['short_cumulative_return'] = (average_df['short_strategy_return'].fillna(0)).cumsum()
        
        # Calculating the consecutive days z-score is above 0
        average_df['consecutive_days_above_0'] = (average_df['z_score_moving_avg'] > 0).astype(int)
        average_df['consecutive_days_above_0'] = average_df['consecutive_days_above_0'].groupby((average_df['z_score_moving_avg'] <= 0).cumsum()).cumsum()
            
        # Calculate the rolling average of consecutive days above 0 where value is at least 1
        average_df['rolling_avg_consecutive_days'] = average_df.loc[average_df['consecutive_days_above_0'] >= 1, 'consecutive_days_above_0'].rolling(window=20).mean()
        average_df['rolling_avg_consecutive_days'] = average_df['rolling_avg_consecutive_days'].ffill()

        # Plotting the cumulative returns using subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        ax1.plot(average_df['date'], average_df['long_cumulative_return'], label='Long Strategy Cumulative Return')
        ax2.plot(average_df['date'], average_df['short_cumulative_return'], label='Short Strategy Cumulative Return', color='red')
        
        # Format axes for cumulative returns
        ax1.set_title('Long Momentum Strategy Cumulative Returns (Z-Score > 0)')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, which="both", ls="--")
        ax1.legend()
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax2.set_title('Short Momentum Strategy Cumulative Returns (Z-Score < 0)')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        plt.tight_layout()
        plt_title = 'Long & Short Momentum Strategy Cumulative Returns'
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)

        # Separate plot for consecutive days with Z-Score above 0
        plt.figure(figsize=(15, 6))
        plt.plot(average_df['date'], average_df['consecutive_days_above_0'], label='Consecutive Days Z-Score > 0', color='green')
        plt_title = 'Consecutive Days with Z-Score Above 0'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Consecutive Days')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)

        # Identifying the peak values
        shifted_values = average_df['consecutive_days_above_0'].shift(-1).fillna(0)
        peaks_mask = (average_df['consecutive_days_above_0'] > shifted_values) & (average_df['consecutive_days_above_0'] > 0)

        # Extracting only the peak values
        peak_series = average_df.loc[peaks_mask, 'consecutive_days_above_0']

        # Calculating rolling average and median of peaks
        rolling_avg_peak = peak_series.rolling(window=20).mean().ffill()
        rolling_median_peak = peak_series.rolling(window=20).apply(lambda x: np.nanmedian(x), raw=True).ffill()

        # Merge the rolling metrics back into the average_df for plotting
        average_df = average_df.merge(rolling_avg_peak.rename('rolling_avg_peak'), left_index=True, right_index=True, how='left')
        average_df = average_df.merge(rolling_median_peak.rename('rolling_median_peak'), left_index=True, right_index=True, how='left')

        # Forward filling the NaN values
        average_df['rolling_avg_peak'] = average_df['rolling_avg_peak'].ffill()
        average_df['rolling_median_peak'] = average_df['rolling_median_peak'].ffill()

        # Plotting the rolling average and median of peaks
        plt.figure(figsize=(15, 6))
        plt.plot(average_df['date'], average_df['rolling_avg_peak'], label='Rolling Average Peak', color='blue')
        plt.plot(average_df['date'], average_df['rolling_median_peak'], label='Rolling Median Peak', color='orange')
        plt_title = 'Rolling Average and Median of Consecutive Days Peak with Z-Score Above 0'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Rolling Metric')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)

        # Bar chart with average and median peak
        avg_peak = peak_series.mean()
        median_peak = peak_series.median()
        plt.figure(figsize=(10, 6))
        plt.bar(['Average Peak', 'Median Peak'], [avg_peak, median_peak], color=['cyan', 'purple'])
        plt_title = 'Average and Median Peaks of Consecutive Days with Z-Score Above 0'
        plt.title(plt_title)
        plt.ylabel('Days')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Calculating the consecutive days z-score is below 0
        average_df['consecutive_days_below_0'] = (average_df['z_score_moving_avg'] < 0).astype(int)
        average_df['consecutive_days_below_0'] = average_df['consecutive_days_below_0'].groupby((average_df['z_score_moving_avg'] >= 0).cumsum()).cumsum()
        
        # Separate plot for consecutive days with Z-Score below 0
        plt.figure(figsize=(15, 6))
        plt.plot(average_df['date'], average_df['consecutive_days_below_0'], label='Consecutive Days Z-Score < 0', color='red')
        plt_title = 'Consecutive Days with Z-Score Below 0'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Consecutive Days')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Identifying the peak values for Z-Score below 0
        shifted_values_below = average_df['consecutive_days_below_0'].shift(-1).fillna(0)
        peaks_mask_below = (average_df['consecutive_days_below_0'] > shifted_values_below) & (average_df['consecutive_days_below_0'] > 0)
        
        # Extracting only the peak values for Z-Score below 0
        peak_series_below = average_df.loc[peaks_mask_below, 'consecutive_days_below_0']
        
        # Calculating rolling average and median of peaks for Z-Score below 0
        rolling_avg_peak_below = peak_series_below.rolling(window=20).mean().ffill()
        rolling_median_peak_below = peak_series_below.rolling(window=20).apply(lambda x: np.nanmedian(x), raw=True).ffill()
        
        # Merge the rolling metrics back into the average_df for plotting
        average_df = average_df.merge(rolling_avg_peak_below.rename('rolling_avg_peak_below'), left_index=True, right_index=True, how='left')
        average_df = average_df.merge(rolling_median_peak_below.rename('rolling_median_peak_below'), left_index=True, right_index=True, how='left')
        
        # Forward filling the NaN values for Z-Score below 0
        average_df['rolling_avg_peak_below'] = average_df['rolling_avg_peak_below'].ffill()
        average_df['rolling_median_peak_below'] = average_df['rolling_median_peak_below'].ffill()
        
        # Plotting the rolling average and median of peaks for Z-Score below 0
        plt.figure(figsize=(15, 6))
        plt.plot(average_df['date'], average_df['rolling_avg_peak_below'], label='Rolling Average Peak (Z < 0)', color='blue')
        plt.plot(average_df['date'], average_df['rolling_median_peak_below'], label='Rolling Median Peak (Z < 0)', color='orange')
        plt_title = 'Rolling Average and Median of Consecutive Days Peak with Z-Score Below 0'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Rolling Metric')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Bar chart with average and median peak for Z-Score below 0
        avg_peak_below = peak_series_below.mean()
        median_peak_below = peak_series_below.median()
        plt.figure(figsize=(10, 6))
        plt.bar(['Average Peak (Z < 0)', 'Median Peak (Z < 0)'], [avg_peak_below, median_peak_below], color=['cyan', 'purple'])
        plt_title = 'Average and Median Peaks of Consecutive Days with Z-Score Below 0'
        plt.title(plt_title)
        plt.ylabel('Days')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # 7. Create a single series for the current momentum period
        average_df['momentum_period'] = np.where(average_df['z_score_moving_avg'] > 0, average_df['consecutive_days_above_0'], -average_df['consecutive_days_below_0'])
        
        # Get the current mean and median values for both positive and negative momentum
        current_avg_peak = average_df['rolling_avg_peak'].iloc[-1]
        current_median_peak = average_df['rolling_median_peak'].iloc[-1]
        current_avg_peak_below = average_df['rolling_avg_peak_below'].iloc[-1]
        current_median_peak_below = average_df['rolling_median_peak_below'].iloc[-1]

        # Plotting the current momentum period with horizontal lines for current mean and median
        plt.figure(figsize=(15, 6))
        plt.plot(average_df['date'], average_df['momentum_period'], label='Current Momentum Period', color='blue')
        plt.axhline(y=current_avg_peak, color='green', linestyle='--', label=f'Current Avg Peak (Z-Score > 0) = {current_avg_peak:.2f}')
        plt.axhline(y=current_median_peak, color='orange', linestyle='--', label=f'Current Median Peak (Z-Score > 0) = {current_median_peak:.2f}')
        plt.axhline(y=-current_avg_peak_below, color='red', linestyle='-.', label=f'Current Avg Peak (Z-Score < 0) = {-current_avg_peak_below:.2f}')
        plt.axhline(y=-current_median_peak_below, color='purple', linestyle='-.', label=f'Current Median Peak (Z-Score < 0) = {-current_median_peak_below:.2f}')
        plt_title = 'Current Momentum Period with Latest Average and Median Values'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Days')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Filter the data for the last 365 bars
        last_365_days = average_df.iloc[-365:]
        
        # Plotting the current momentum period for the last 365 bars with horizontal lines for current mean and median
        plt.figure(figsize=(15, 6))
        plt.plot(last_365_days['date'], last_365_days['momentum_period'], label='Current Momentum Period', color='blue')
        plt.axhline(y=current_avg_peak, color='green', linestyle='--', label=f'Current Avg Peak (Z-Score > 0) = {current_avg_peak:.2f}')
        plt.axhline(y=current_median_peak, color='orange', linestyle='--', label=f'Current Median Peak (Z-Score > 0) = {current_median_peak:.2f}')
        plt.axhline(y=-current_avg_peak_below, color='red', linestyle='-.', label=f'Current Avg Peak (Z-Score < 0) = {-current_avg_peak_below:.2f}')
        plt.axhline(y=-current_median_peak_below, color='purple', linestyle='-.', label=f'Current Median Peak (Z-Score < 0) = {-current_median_peak_below:.2f}')
        plt_title = 'Current Momentum Period with Latest Average and Median Values (Last 365 Bars)'
        plt.title(plt_title)
        plt.xlabel('Date')
        plt.ylabel('Days')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)

        # Modify the peak masks to include the condition for durations greater than 20 days
        peaks_mask_above_20 = (peaks_mask) & (average_df['consecutive_days_above_0'] > 20)
        peaks_mask_below_20 = (peaks_mask_below) & (average_df['consecutive_days_below_0'] > 20)
        
        # Extracting the indices of the peaks above 0 that last more than 20 days
        peak_indices_above_20 = average_df.loc[peaks_mask_above_20, 'consecutive_days_above_0'].index
        
        # Similarly for peaks below 0
        peak_indices_below_20 = average_df.loc[peaks_mask_below_20, 'consecutive_days_below_0'].index
        
        # Calculating the distances between these peak indices above 0
        distances_between_peaks_above = np.diff(peak_indices_above_20)
        
        # Filtering for distances above 20 days
        distances_above_20_above = distances_between_peaks_above[distances_between_peaks_above > 20]
        
        # Calculating average and median distances
        avg_distance_above = distances_above_20_above.mean()
        median_distance_above = np.median(distances_above_20_above)
        
        # Calculating the distances between these peak indices below 0
        distances_between_peaks_below = np.diff(peak_indices_below_20)
        
        # Filtering for distances above 20 days
        distances_above_20_below = distances_between_peaks_below[distances_between_peaks_below > 20]
        
        # Calculating average and median distances for below 0
        avg_distance_below = distances_above_20_below.mean()
        median_distance_below = np.median(distances_above_20_below)
        
        # Bar chart with average and median distances for peaks above and below 0
        plt.figure(figsize=(12, 6))
        plt.bar(['Average Distance Positive Momentum', 'Median Distance (Days)', 'Avg Dist Below 0', 'Med Dist Below 0'],
                [avg_distance_above, median_distance_above, avg_distance_below, median_distance_below],
                color=['cyan', 'purple', 'blue', 'orange'])
        plt_title = 'Average and Median Distances Between Peaks (Above 20 Days)'
        plt.title(plt_title)
        plt.ylabel('Days')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Function to compute the length of consecutive momentum days
        def compute_streaks(s):
            streaks = (s.groupby(s.ne(s.shift()).cumsum()).cumcount(ascending=False) + 1) * s
            return streaks[s != 0]
        
        # Compute consecutive positive momentum streaks
        positive_streaks = compute_streaks((average_df['momentum_period'] > 0).astype(int))
        consecutive_positive_momentum = positive_streaks.value_counts().sort_index()
        
        # Compute consecutive negative momentum streaks
        negative_streaks = compute_streaks((average_df['momentum_period'] < 0).astype(int))
        consecutive_negative_momentum = negative_streaks.value_counts().sort_index()
        
        # Compute the probabilities for positive momentum continuing
        positive_momentum_probability = {}
        for n in consecutive_positive_momentum.index[:-1]: # excluding the max streak
            total_for_n = consecutive_positive_momentum[n]
            continued_to_n_plus_1 = consecutive_positive_momentum.get(n+5, 0)
            positive_momentum_probability[n] = continued_to_n_plus_1 / total_for_n
        
        # Compute the probabilities for negative momentum continuing
        negative_momentum_probability = {}
        for n in consecutive_negative_momentum.index[:-1]: # excluding the max streak
            total_for_n = consecutive_negative_momentum[n]
            continued_to_n_plus_1 = consecutive_negative_momentum.get(n+5, 0)
            negative_momentum_probability[n] = continued_to_n_plus_1 / total_for_n
        
        # Plotting the probabilities
        plt.figure(figsize=(15, 6))
        plt.plot(list(positive_momentum_probability.keys()), list(positive_momentum_probability.values()), '-o', label='Probability of Positive Momentum Continuing', color='green')
        plt.plot(list(negative_momentum_probability.keys()), list(negative_momentum_probability.values()), '-o', label='Probability of Negative Momentum Continuing', color='red')
        plt_title = 'Probability of Momentum Continuing for N+5 Days Given N Days of Consecutive Momentum'
        plt.title(plt_title)
        plt.xlabel('Consecutive Days')
        plt.ylabel('Probability')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Convert the dictionaries to pandas Series
        positive_prob_series = pd.Series(positive_momentum_probability)
        negative_prob_series = pd.Series(negative_momentum_probability)
        
        # Compute rolling average
        rolling_window = 5
        positive_roll_avg = positive_prob_series.rolling(window=rolling_window, min_periods=1).mean()
        negative_roll_avg = negative_prob_series.rolling(window=rolling_window, min_periods=1).mean()
        
        # Plotting the averaged probabilities
        plt.figure(figsize=(15, 6))
        plt.plot(positive_roll_avg.index, positive_roll_avg.values, '-o', label='Averaged Probability of Positive Momentum Continuing', color='green')
        plt.plot(negative_roll_avg.index, negative_roll_avg.values, '-o', label='Averaged Probability of Negative Momentum Continuing', color='red')
        plt_title = f'Rolling {rolling_window}-Day Average Probability of Momentum Continuing for N+5 Days Given N Days of Consecutive Momentum'
        plt.title(plt_title)
        plt.xlabel('Consecutive Days')
        plt.ylabel('Probability')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Convert the dictionaries to pandas Series
        positive_prob_series = pd.Series(positive_momentum_probability)
        negative_prob_series = pd.Series(negative_momentum_probability)
        
        # Get the weights (occurrences for each streak length)
        positive_weights = consecutive_positive_momentum.reindex(positive_prob_series.index, fill_value=1)
        negative_weights = consecutive_negative_momentum.reindex(negative_prob_series.index, fill_value=1)
        
        # Compute weighted rolling average
        rolling_window = 5
        positive_weighted_avg = (positive_prob_series * positive_weights).rolling(window=rolling_window).sum() / positive_weights.rolling(window=rolling_window).sum()
        negative_weighted_avg = (negative_prob_series * negative_weights).rolling(window=rolling_window).sum() / negative_weights.rolling(window=rolling_window).sum()
        
        # Handle potential NaN values after division
        positive_weighted_avg.fillna(0, inplace=True)
        negative_weighted_avg.fillna(0, inplace=True)
        
        # Plotting the weighted averaged probabilities
        plt.figure(figsize=(15, 6))
        plt.plot(positive_weighted_avg.index, positive_weighted_avg.values, '-o', label='Weighted Average Probability of Positive Momentum Continuing', color='green')
        plt.plot(negative_weighted_avg.index, negative_weighted_avg.values, '-o', label='Weighted Average Probability of Negative Momentum Continuing', color='red')
        plt_title = f'Weighted {rolling_window}-Day Average Probability of Momentum Continuing for N+5 Days Given N Days of Consecutive Momentum'
        plt.title(plt_title)
        plt.xlabel('Consecutive Days')
        plt.ylabel('Probability')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        # Get the latest value of the momentum period
        current_momentum_period = average_df['momentum_period'].iloc[-1]
        
        # Determine if the current momentum is positive or negative
        if current_momentum_period > 0:
            momentum_type = "Positive"
        elif current_momentum_period < 0:
            momentum_type = "Negative"
        else:
            momentum_type = "No"
        
        # Print the duration of the current momentum period
        await channel.send(f"Current {momentum_type} Momentum Period Duration: {abs(current_momentum_period)} days\n"
        
        # Average and Median Peaks when Z-Score is above 0
        f"Average Peak when Z-Score > 0: {current_avg_peak}\n"
        f"Median Peak when Z-Score > 0: {current_median_peak}\n"
        
        # Average and Median Peaks when Z-Score is below 0
        f"Average Peak when Z-Score < 0: {current_avg_peak_below}\n"
        f"Median Peak when Z-Score < 0: {current_median_peak_below}\n")  
        
        # Separate positive and negative momentum periods
        positive_momentum = average_df[average_df['momentum_period'] > 0]['momentum_period']
        negative_momentum = average_df[average_df['momentum_period'] < 0]['momentum_period']
        
        # Set up the figure and axes for the two histograms with reversed axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the histogram for positive momentum periods with reversed x-axis
        ax1.hist(positive_momentum, bins=20, color='green', edgecolor='black')
        ax1.set_title('Positive Momentum Periods')
        ax1.set_xlabel('Momentum Period')
        ax1.set_ylabel('Frequency')
        ax1.invert_xaxis()
        
        # Plot the histogram for negative momentum periods with reversed x-axis
        ax2.hist(negative_momentum, bins=20, color='red', edgecolor='black')
        ax2.set_title('Negative Momentum Periods')
        ax2.set_xlabel('Momentum Period')
        ax2.set_ylabel('Frequency')
        ax2.invert_xaxis()
        
        # Add the current momentum period as a vertical line and label on the appropriate histogram
        label_text = f"Current: {current_momentum_period} days"
        if current_momentum_period > 0:
            ax1.axvline(x=current_momentum_period, color='blue', linestyle='dashed', linewidth=2)
            ax1.text(current_momentum_period, max(ax1.get_ylim()) * 0.95, label_text, color='black',
                    ha='right', va='top', rotation=90)
        elif current_momentum_period < 0:
            ax2.axvline(x=current_momentum_period, color='blue', linestyle='dashed', linewidth=2)
            ax2.text(current_momentum_period, max(ax2.get_ylim()) * 0.95, label_text, color='black',
                    ha='right', va='top', rotation=90)
        
        # Display the plot
        plt.tight_layout()
        plt.savefig(f'{path}/Momentum Periods')
        await send_plot(plt_title)
        
        # Determine daily momentum as positive (1) or negative (-1)
        average_df['daily_momentum'] = np.sign(average_df['return'])
        
        # Create a lagged column for the previous day's momentum
        average_df['prev_day_momentum'] = average_df['daily_momentum'].shift(1)
        
        # Drop the first row since it doesn't have a previous day's momentum
        average_df = average_df.dropna(subset=['prev_day_momentum'])
        
        # Create a contingency table for the Chi-square test
        contingency_table = pd.crosstab(average_df['prev_day_momentum'], average_df['daily_momentum'])
        
        # Perform the Chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Print the p-value
        await channel.send(f"Chi-square test p-value: {p}\n")
        
        # Add a new column indicating if z_score_moving_avg is positive the next day and for the next 5, 15, 30 days
        for days in [5, 15, 30, 90]:
            average_df[f'z_score_moving_avg_positive_next_{days}_days'] = (average_df['z_score_moving_avg'].shift(-days) > 0).astype(int)

        # Determine the 20th and 80th percentiles for consecutive positive momentum days
        percentile_20 = average_df['consecutive_days_above_0'].quantile(0.02)
        percentile_80 = average_df['consecutive_days_above_0'].quantile(0.98)

        # Filter the data between the 20th and 80th percentiles
        filtered_df = average_df[(average_df['consecutive_days_above_0'] >= percentile_20) & (average_df['consecutive_days_above_0'] <= percentile_80)]

        # Initialize a DataFrame to hold probabilities for each time frame
        probability_dfs = {}

        # Calculate probabilities for each time frame
        for days in [5, 15, 30, 90]:
            column_name = f'z_score_moving_avg_positive_next_{days}_days'
            grouped = filtered_df.groupby('consecutive_days_above_0')[column_name]
            probability_df = grouped.mean().rolling(window=10, min_periods=1).mean()
            probability_dfs[days] = probability_df

        # Plotting all moving averages
        plt.figure(figsize=(12, 6))
        for days, probability_df in probability_dfs.items():
            plt.plot(probability_df.index, probability_df.values, label=f'Next {days} Days')
        
        # Add a vertical line for the current momentum period
        plt.axvline(x=current_momentum_period, color='darkred', linestyle='--')
        
        # Adjust the position of the label to add more space from the vertical line
        label_offset = 0.5  # Adjust the offset as needed
        plt.text(current_momentum_period + label_offset, 0.5, f'Current: {current_momentum_period} days', color='darkred', rotation=90)
        plt_title = 'Moving Average of Probability of Normalized Momentum Being Positive'
        plt.title(plt_title)
        plt.xlabel('Consecutive Positive Momentum Days')
        plt.ylabel('Moving Average of Probability')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{path}/{plt_title}')
        await send_plot(plt_title)
        
        return average_df[['date', 'long_strategy_return', 'short_strategy_return', 'long_cumulative_return', 'short_cumulative_return', 'long_position', 'short_position', 'rolling_avg_consecutive_days']]

    # Discord preperation
    guild = discord.utils.get(client.guilds, name='Trading')
    channel = discord.utils.get(guild.text_channels, name='🌐│momentum')
    await channel.send(str(datetime.datetime.today().date()))

    # Usage:
    await calculate_and_plot_z_scores(daily_data, start_year_to_display=2023, end_year_to_display=2024, data_frequency='weekly')
    strategy_results_df = await momentum_strategy_returns(daily_data)

# Connect to Binance API
def connect_to_binance():
    return Client()

# Create Folder for current scan plots (name = current date)
def data_folder():
    directory = str(datetime.datetime.today().date())
    parent_dir = 'D:/pyQuant/Momentum_Scans/Momentum_Plots'
    path = os.path.join(parent_dir, directory) 
    try:
        os.mkdir(path) 
        print("Directory '% s' created" % directory)
    except Exception as e:
            print("Error occured - {}".format(e))
    return path

def get_spot_data(client, timeframe, days_range, n_assets=None):
    spot_data = {}

    # Get the current date
    today_date = datetime.date.today()

    # Calculate the trade date for the specified number of days in the past
    trade_date_data = today_date - relativedelta(days=days_range)

    # Store the arguments for the get_historical_klines() method in variables
    timeframe_for_data = timeframe
    start_date_for_data = str(trade_date_data)

    # Get the list of spot symbols
    markets_binance = client.get_exchange_info()

    # Extract the symbols of spot contracts traded against USDT using a list comprehension
    assets_spot = [i["symbol"] for i in markets_binance["symbols"] if i["status"] == "TRADING" and i["symbol"].endswith("USDT")]

    if n_assets is not None:
        assets_spot = assets_spot[:n_assets]

    counter = 0
    # Loop through the symbols and get the daily data for each one
    for symbol in assets_spot:
        print(f"Extracting data for: {symbol}")
        # Initialize an empty list to store the data for the symbol
        symbol_data = []

        # Get the daily data for the symbol
        data = client.get_historical_klines(symbol, timeframe_for_data, start_date_for_data)

        # Loop through the daily data and extract the relevant information
        for daily_data in data:
            # Extract the timestamp, open price, high price, low price, close price, and volume
            timestamp, open_price, high_price, low_price, close_price, volume = daily_data[:6]

            # Convert the timestamp to a date
            date = datetime.datetime.fromtimestamp(int(timestamp) / 1000)

            # Add the cleaned data to the list
            symbol_data.append({
                'date': date,
                'asset': symbol,
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume)
            })

        # Convert the data for the symbol to a dataframe
        spot_data[symbol] = pd.DataFrame(symbol_data)

        counter += 1
        print(f"Finished Extracting data for: {symbol}")
        print(f"Task Percentage Done: {counter / len(assets_spot)}")

    return spot_data

if dt.datetime.now().hour in [16,17] and dt.datetime.weekday(dt.datetime.now()) == 6:
    path = data_folder()
    daily_data = get_spot_data(connect_to_binance(), "1d", 3000)
    client.run(TOKEN)
else:
    print("Skipping Momentum scan because it is not the wOpen.")