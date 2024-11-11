import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from astropy.timeseries import LombScargle

st.title("RV Period Peak Improver")

RV_DIR = "RV"

@st.cache_data
def get_csv_files(rv_dir):
    csv_files = []
    for file in os.listdir(rv_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(rv_dir, file)
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                csv_files.append((file, row_count))
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")
        else:
            st.warning(f"Ignoring non-CSV file: {file}")
    sorted_files = sorted(csv_files, key=lambda x: x[1], reverse=True)
    return sorted_files

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

st.sidebar.header("Select RV Dataset")

csv_files = get_csv_files(RV_DIR)
file_names = [os.path.splitext(file)[0] for file, _ in csv_files]
if not file_names:
    st.error(f"No CSV files found in the '{RV_DIR}' directory.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Choose a dataset:",
    options=file_names,
    format_func=lambda x: x
)

selected_index = file_names.index(selected_file)
selected_row_count = csv_files[selected_index][1]

st.sidebar.write(f"**Number of Data Points:** {selected_row_count}")

csv_filename = f"{selected_file}.csv"
csv_path = os.path.join(RV_DIR, csv_filename)
data = load_data(csv_path)

if data is not None:
    required_columns = ['MJD', 'RV', 'RV_error']
    if not all(col in data.columns for col in required_columns):
        st.error(f"One or more required columns are missing in {csv_filename}.")
        st.stop()

    n = st.sidebar.number_input(
        "Number of data points to remove (n)",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    use_greedy = st.sidebar.checkbox("Use Greedy Algorithm", value=False)

    mjd = data['MJD'].values
    time_span = mjd.max() - mjd.min()

    period_slider_min = 0.5
    period_slider_max = time_span * 150

    min_period = st.sidebar.number_input(
        "Minimum Period for Improvement (days)",
        min_value=period_slider_min,
        max_value=period_slider_max,
        value=period_slider_min,
        step=0.1
    )

    max_period = st.sidebar.number_input(
        "Maximum Period for Improvement (days)",
        min_value=period_slider_min,
        max_value=period_slider_max,
        value=period_slider_max,
        step=0.1
    )

    period_range = (min_period, max_period)
    
    st.subheader("Radial Velocity (RV) vs Modified Julian Date (MJD)")

    fig_rv = go.Figure()

    fig_rv.add_trace(
        go.Scatter(
            x=data['MJD'],
            y=data['RV'],
            error_y=dict(
                type='data',
                array=data['RV_error'],
                visible=True
            ),
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.7
            ),
            name='RV Data'
        )
    )

    fig_rv.update_layout(
        xaxis_title="MJD",
        yaxis_title="Radial Velocity (RV)",
        title=f"RV vs MJD for {selected_file}",
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(tickformat=".0f")
    )

    st.plotly_chart(fig_rv, use_container_width=True)

    st.subheader("Lomb-Scargle Periodogram Before Removal")

    rv = data['RV'].values
    rv_error = data['RV_error'].values

    min_period, max_period = period_range

    minimum_frequency = 1 / max_period
    maximum_frequency = 1 / min_period

    oversampling = 500
    freq_grid = np.linspace(minimum_frequency, maximum_frequency, int(len(mjd) * oversampling))
    period = 1 / freq_grid

    ls = LombScargle(mjd, rv, rv_error)
    power = ls.power(freq_grid)

    fap_levels = [0.5, 0.01, 0.001]
    fap_labels = ['50% FAP', '1% FAP', '0.1% FAP']
    fap_colors = ['green', 'orange', 'red']

    fig_ls_before = go.Figure()

    fig_ls_before.add_trace(
        go.Scatter(
            x=period,
            y=power,
            mode='lines',
            line=dict(color='red'),
            name='Lomb-Scargle Power'
        )
    )

    for fap, label, color in zip(fap_levels, fap_labels, fap_colors):
        level = ls.false_alarm_level(fap)
        fig_ls_before.add_hline(
            y=level,
            line_dash='dash',
            line_color=color,
            annotation_text=label,
            annotation_position="top left",
            annotation=dict(font_size=12, showarrow=False),
        )

    fig_ls_before.update_layout(
        xaxis_title="Period (days)",
        yaxis_title="Power",
        title=f"Lomb-Scargle Periodogram for {selected_file} (Before Removal)",
        template="plotly_white",
        xaxis=dict(type='log'),
        yaxis=dict(range=[0, max(power) * 1.1])
    )

    st.plotly_chart(fig_ls_before, use_container_width=True)

    peak_power_before = power.max()
    peak_freq_before = freq_grid[power.argmax()]
    peak_period_before = 1 / peak_freq_before
    st.markdown(f"**Dominant Period Before Removal:** {peak_period_before:.2f} days with power {peak_power_before:.4f}")

    if use_greedy:
        @st.cache_data
        def greedy_removal(data, n, freq_grid):
            data_remaining = data.copy()
            removed_indices = []
            mjd = data['MJD'].values
            rv = data['RV'].values
            rv_error = data['RV_error'].values
            ls = LombScargle(mjd, rv, rv_error)
            power = ls.power(freq_grid)
            peak_power = power.max()

            for _ in range(n):
                max_increase = 0
                best_index = None

                for index in data_remaining.index:
                    temp_data = data_remaining.drop(index)
                    temp_mjd = temp_data['MJD'].values
                    temp_rv = temp_data['RV'].values
                    temp_rv_error = temp_data['RV_error'].values

                    temp_ls = LombScargle(temp_mjd, temp_rv, temp_rv_error)
                    temp_power = temp_ls.power(freq_grid)
                    temp_peak_power = temp_power.max()

                    increase = temp_peak_power - peak_power

                    if increase > max_increase:
                        max_increase = increase
                        best_index = index
                        best_peak_power = temp_peak_power

                if max_increase <= 0:
                    break
                else:
                    data_remaining = data_remaining.drop(best_index)
                    removed_indices.append(best_index)
                    peak_power = best_peak_power

            return data_remaining, removed_indices, peak_power

        with st.spinner("Analyzing data using Greedy Algorithm..."):
            data_remaining, removed_indices, peak_power_after = greedy_removal(data, n, freq_grid)
    else:
        @st.cache_data
        def compute_influence_scores(data, freq_grid, target_frequency):
            influence_scores = []
            mjd = data['MJD'].values
            rv = data['RV'].values
            rv_error = data['RV_error'].values

            ls = LombScargle(mjd, rv, rv_error)
            power = ls.power(freq_grid)
            initial_peak_power = power.max()

            idx_peak = np.abs(freq_grid - target_frequency).argmin()

            for index in data.index:
                temp_data = data.drop(index)
                temp_mjd = temp_data['MJD'].values
                temp_rv = temp_data['RV'].values
                temp_rv_error = temp_data['RV_error'].values

                temp_ls = LombScargle(temp_mjd, temp_rv, temp_rv_error)
                temp_power = temp_ls.power(freq_grid)
                temp_peak_power = temp_power.max()

                influence = temp_peak_power - initial_peak_power
                influence_scores.append((index, influence))

            return influence_scores

        with st.spinner("Analyzing data using Influence Score Algorithm..."):
            target_frequency = peak_freq_before
            influence_scores = compute_influence_scores(data, freq_grid, target_frequency)

        influence_scores.sort(key=lambda x: x[1], reverse=True)

        removed_indices = [idx for idx, influence in influence_scores[:n] if influence > 0]

        if len(removed_indices) == 0:
            st.warning(f"Removal of {n} points did not result in the peak getting more prominent.")
            data_remaining = data.copy()
            peak_power_after = peak_power_before
        else:
            st.success(f"Found {len(removed_indices)} data point(s) whose removal improves the dominant peak.")
            data_remaining = data.drop(removed_indices)
            temp_mjd = data_remaining['MJD'].values
            temp_rv = data_remaining['RV'].values
            temp_rv_error = data_remaining['RV_error'].values

            ls_after = LombScargle(temp_mjd, temp_rv, temp_rv_error)
            power_after = ls_after.power(freq_grid)
            peak_power_after = power_after.max()

    if len(removed_indices) == 0:
        st.warning(f"Removal of {n} points did not result in the peak getting more prominent.")
        data_remaining = data.copy()
        peak_power_after = peak_power_before
        power_after = power
    else:
        #st.success(f"Found {len(removed_indices)} data point(s) whose removal improves the dominant peak.")

        temp_mjd = data_remaining['MJD'].values
        temp_rv = data_remaining['RV'].values
        temp_rv_error = data_remaining['RV_error'].values

        ls_after = LombScargle(temp_mjd, temp_rv, temp_rv_error)
        power_after = ls_after.power(freq_grid)
        peak_power_after = power_after.max()

        st.subheader("Radial Velocity (RV) vs Modified Julian Date (MJD) After Removal")

        fig_rv_after = go.Figure()

        fig_rv_after.add_trace(
            go.Scatter(
                x=data_remaining['MJD'],
                y=data_remaining['RV'],
                error_y=dict(
                    type='data',
                    array=data_remaining['RV_error'],
                    visible=True
                ),
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.7
                ),
                name='Remaining RV Data'
            )
        )

        removed_data = data.loc[removed_indices]
        fig_rv_after.add_trace(
            go.Scatter(
                x=removed_data['MJD'],
                y=removed_data['RV'],
                error_y=dict(
                    type='data',
                    array=removed_data['RV_error'],
                    visible=True
                ),
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.9,
                    symbol='x'
                ),
                name='Removed Data Points'
            )
        )

        fig_rv_after.update_layout(
            xaxis_title="MJD",
            yaxis_title="Radial Velocity (RV)",
            title=f"RV vs MJD for {selected_file} (After Removal)",
            template="plotly_white",
            hovermode="closest",
            xaxis=dict(tickformat=".0f")
        )

        st.plotly_chart(fig_rv_after, use_container_width=True)

        st.subheader("Lomb-Scargle Periodogram After Removal")

        fig_ls_after = go.Figure()

        fig_ls_after.add_trace(
            go.Scatter(
                x=period,
                y=power_after,
                mode='lines',
                line=dict(color='blue'),
                name='Lomb-Scargle Power (After Removal)'
            )
        )

        for fap, label, color in zip(fap_levels, fap_labels, fap_colors):
            level = ls_after.false_alarm_level(fap)
            fig_ls_after.add_hline(
                y=level,
                line_dash='dash',
                line_color=color,
                annotation_text=label,
                annotation_position="top left",
                annotation=dict(font_size=12, showarrow=False),
            )

        fig_ls_after.update_layout(
            xaxis_title="Period (days)",
            yaxis_title="Power",
            title=f"Lomb-Scargle Periodogram for {selected_file} (After Removal)",
            template="plotly_white",
            xaxis=dict(type='log'),
            yaxis=dict(range=[0, max(power_after) * 1.1])
        )

        st.plotly_chart(fig_ls_after, use_container_width=True)

        peak_freq_after = freq_grid[power_after.argmax()]
        peak_period_after = 1 / peak_freq_after
        st.markdown(f"**Dominant Period After Removal:** {peak_period_after:.2f} days with power {peak_power_after:.4f}")

        st.markdown(f"**Improvement in Peak Power:** {peak_power_after - peak_power_before:.4f}")

        st.subheader("Removed Data Points")
        st.dataframe(removed_data)

else:
    st.error("No data to display.")
