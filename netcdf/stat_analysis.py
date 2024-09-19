import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import numpy as np
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import pearsonr, spearmanr
import statistics

name_map = {'VHM0': 'Sea surface wave significant height',
            'VTM01_WW': 'Sea surface wind wave mean period',
            'VMDR_WW': 'Sea surface wind wave from direction',
            'wind_dir': 'Wind direction',
            'wind_speed': 'Wind speed'
            }


def print_stats(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):
    # For numeric columns, calculate the mean, std, median, 25th-percentile, and 75th-percentile of each column
    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:
            column_data = df[column][df[column] != exclude_value]

            min_value = column_data.min()
            max_value = column_data.max()

            mean_value = column_data.mean()
            std_value = column_data.std()

            text = f"Column '{name_map[column]} [{column}]'"
            text = text.ljust(60)

            print(f"{text} : Mean: {mean_value:.2f}, Std: {std_value:.2f}, Min: {min_value:.2f}, Max: {max_value:.2f}")
    print()


def relation_measures(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    grouped = df.groupby(['LAT', 'LON'])

    dfs = dict()
    # Iterate over each ('LAT', 'LON') group, sort by 'TIME' and store the DataFrame in the dictionary
    for i, (group_name, group_df) in enumerate(grouped):
        dfs[f'LAT_{group_name[0]}_LON_{group_name[1]}'] = group_df.sort_values(by='TIME')

    print(len(dfs))

    columns = df.select_dtypes(include='number').columns
    columns = [col for col in columns if col not in except_columns]
    pairs = list(itertools.combinations(columns, 2))

    few_datapoints_cntr = 0

    pairs_corr = dict()
    for pair in pairs:
        pairs_corr[pair] = {'pearson': [], 'spearman': []}

    for key, df_ll in dfs.items():

        for col1, col2 in pairs:
            df_selected = df_ll.loc[:, [col1, col2]]
            df_selected = df_selected[df_selected[col2] != exclude_value]
            df_selected = df_selected[df_selected[col1] != exclude_value]

            if df_selected.shape[0] > 5:
                pearson_corr, _ = pearsonr(df_selected[col1], df_selected[col2])
                spearman_corr, _ = spearmanr(df_selected[col1], df_selected[col2])

                text = f"{col1} vs {col2}"
                text = text.ljust(25)

                # (f'{text} - Pearson correlation : {pearson_corr:.3f} and Sperman correlation : {spearman_corr:.3f}')

                pairs_corr[(col1, col2)]['pearson'].append(pearson_corr)
                pairs_corr[(col1, col2)]['spearman'].append(spearman_corr)
            else:
                few_datapoints_cntr += 1

    for pair in pairs:
        pearson_corr = pairs_corr[pair]['pearson']
        spearman_corr = pairs_corr[pair]['spearman']

        pairs_corr[pair]['pearson_stats'] = [statistics.mean(pearson_corr),
                                             statistics.mean([abs(x) for x in pearson_corr]),
                                             statistics.stdev(pearson_corr)]
        pairs_corr[pair]['spearman_stats'] = [statistics.mean(spearman_corr),
                                              statistics.mean([abs(x) for x in spearman_corr]),
                                              statistics.stdev(spearman_corr)]

        col1, col2 = pair
        plt.hist(pearson_corr, edgecolor='black')
        plt.title(f'Histogram of Pearson Correlation ({col1} vs {col2})')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.savefig(f"../plots/histogram_pearson_{col1}_vs_{col2}.png", bbox_inches='tight', dpi=600)
        plt.clf()

    for col1, col2 in pairs:
        text = f"{col1} vs {col2}"
        text = text.ljust(25)

        print(f"{text} - Pearson correlation : {abs(pairs_corr[(col1, col2)]['pearson_stats'][1])} , {pairs_corr[(col1, col2)]['pearson_stats'][2]} "
              f"and Spearman correlation : {abs(pairs_corr[(col1, col2)]['spearman_stats'][1])} , {pairs_corr[(col1, col2)]['spearman_stats'][2]}")


def plot_wavelets(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    fs = 20000
    # Parameters for wavelet analysis
    wavelet = 'cmor'  # Complex Morlet wavelet

    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:

            column_data = df[column][df[column] != exclude_value]

            scales = np.arange(1, 128)
            signal = column_data

            # Compute the Continuous Wavelet Transform (CWT)
            coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)

            # Compute the power spectrum
            power_spectrum = np.abs(coefficients) ** 2

            # Average the power spectrum across time
            average_power_spectrum = np.mean(power_spectrum, axis=1)

            # Plot the results
            plt.figure(figsize=(12, 8))

            # Plot the average power spectrum
            plt.plot(frequencies, average_power_spectrum, color='blue')
            plt.title(f'Average Power Spectrum from Wavelet Transform - [{name_map[column]} ({column})]')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.grid(True)
            plt.savefig(f"../plots/power_{column}.png", bbox_inches='tight', dpi=600)


def plot_fft(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:

            column_data = df[column][df[column] != exclude_value]

            fs = 20000
            window_size = 100

            signal = column_data
            power_spectra = np.zeros((len(signal) // window_size + 1, window_size // 2))
            frequencies = None

            # Perform windowed FFT and phase analysis
            for i, start in enumerate(range(0, len(signal), window_size)):
                end = start + window_size
                windowed_signal = signal[start:end]

                yf = fft(windowed_signal)
                xf = fftfreq(window_size, 1 / fs)

                # Store frequency vector from the first window (assumed consistent across windows)
                if frequencies is None:
                    frequencies = xf[:window_size // 2]  # Keep only positive frequencies

                # Compute power spectrum
                power_spectrum = np.abs(yf) ** 2
                power_spectra[i, :] = power_spectrum[:window_size // 2]  # Keep only positive frequencies

            # Average the power spectra across all windows
            average_power_spectrum = np.mean(power_spectra, axis=0)
            f, t_spec, Sxx = spectrogram(signal, fs, nperseg=window_size)

            # Plot the results
            plt.figure(figsize=(12, 8))

            # Plot the power spectrum
            plt.subplot(2, 1, 1)
            plt.plot(frequencies, average_power_spectrum, color='blue')
            plt.title(f'Average Power Spectrum from Windowed FFT [{name_map[column]} ({column})]')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.grid(True)

            # Plot the spectrogram
            plt.subplot(2, 1, 2)
            plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
            plt.title('Spectrogram')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            plt.show()


def plot_spectogram(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:

            column_data = df[column][df[column] != exclude_value]

            fs = 20000
            signal = column_data

            f, t_spec, Sxx = spectrogram(signal, fs)

            plt.figure(figsize=(12, 8))
            plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
            plt.title(f'Spectrogram - {name_map[column]} ({column})')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            plt.savefig(f"../plots/spectogram_{column}.png", bbox_inches='tight', dpi=600)
            plt.clf()


def histogram(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):
    # For numeric columns, calculate the mean, std, median, 25th-percentile, and 75th-percentile of each column
    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:
            column_data = df[column][df[column] != exclude_value]
            column_data.plot(kind='hist', edgecolor='black', color='skyblue')

            # Set the title and labels
            plt.title(f'Histogram - {name_map[column]} ({column})')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.savefig(f"../plots/histogram_{column}.png", bbox_inches='tight', dpi=600)
            plt.clf()


def plot_values_time(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    time = 'TIME'

    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:

            df_selected = df.loc[:, [time, column]]
            df_selected = df_selected[df_selected[column] != exclude_value]

            plt.figure()
            plt.plot(df_selected[time], df_selected[column], label=column, color='skyblue')

            # Set title and labels
            plt.title(f'{time} vs {column}')
            plt.xlabel(time)
            plt.ylabel(column)

            plt.legend()
            plt.show()


def plot_values_idx(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    for column in df.select_dtypes(include='number').columns:
        if column not in except_columns:

            df_selected = df.loc[:, [column]]
            df_selected = df_selected[df_selected[column] != exclude_value]

            plt.figure()
            plt.plot(df_selected[column], label=column, color='skyblue')

            # Set title and labels
            plt.title(f'{column}')
            plt.xlabel('index')
            plt.ylabel(column)

            plt.legend()
            plt.show()


def plot_values(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    columns = df.select_dtypes(include='number').columns
    columns = [col for col in columns if col not in except_columns]

    pairs = list(itertools.combinations(columns, 2))

    for col1, col2 in pairs:

        df_selected = df.loc[:, [col1, col2]]
        df_selected = df_selected[df_selected[col2] != exclude_value]
        df_selected = df_selected[df_selected[col1] != exclude_value]

        plt.figure()
        plt.plot(df_selected[col1], df_selected[col2], color='skyblue')

        # Set title and labels
        plt.title(f'{col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)

        plt.show()


def plot_values_3d(df, except_columns=('LAT', 'LON', 'TIME'), exclude_value=-999):

    columns = df.select_dtypes(include='number').columns
    columns = [col for col in columns if col not in except_columns]

    pairs = list(itertools.combinations(columns, 2))

    for col1, col2 in pairs[:2]:

        df_selected = df.loc[:, [col1, col2, 'TIME']]
        df_selected = df_selected[df_selected[col2] != exclude_value]
        df_selected = df_selected[df_selected[col1] != exclude_value]

        x = df_selected[col1]
        y = df_selected[col2]
        z = df_selected['TIME']

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
        plt.colorbar(sc, ax=ax, label='Depth (Z)')

        # Setting labels for axes
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_zlabel('time')

        plt.show()


os.makedirs("../plots", exist_ok=True)

paths = ["data-wind_dims"]  # , "data-wind_dims", "data_noc"]

for path in paths:
    folder_path = f'../data/{path}'

    files_and_folders = os.listdir(folder_path)

    for item in files_and_folders:
        csv_pth = f'{folder_path}/{item}'
        df = pd.read_csv(csv_pth, index_col=0)

        print(csv_pth)
        print_stats(df)
        relation_measures(df)

        print("Creating histograms...")
        histogram(df)

        df['DATETIME'] = pd.to_datetime(df['TIME'], unit='s')
        df.to_csv(f'{folder_path}/data_datetime.csv', index=False)