import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv('data/covid-data-cleaned.csv', parse_dates=['date'])
    df['location'] = df['location'].str.strip()
    return df

df = load_data()

# Tabs
tab_labels = [
    "Trendy dla kraju",
    "Porównanie szczepień",
    "Zależność: szczepienia vs zgony",
    "Zależność: szczepienia vs przypadki",
    "Zależność: restrykcje vs pandemia",
    "Porównanie kontynentów",
    "Mapa świata"
]

selected_tab = st.radio("Wybierz widok:", tab_labels, horizontal=True)

# -------------------------------
# SIDEBAR
# -------------------------------
if selected_tab == "Trendy dla kraju":
    st.sidebar.header("Ustawienia – kraj")

    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    countries = df['location'].unique()
    country = st.sidebar.selectbox("Wybierz kraj:", sorted(countries), index=sorted(countries).index('Poland'))

    st.sidebar.markdown("Wybierz trendy:")
    show_vacc = st.sidebar.checkbox("Pełne szczepienie (% populacji)", value=True)
    show_cases = st.sidebar.checkbox("Nowe przypadki dziennie")
    show_deaths = st.sidebar.checkbox("Nowe zgony dziennie")
    show_lockdown = st.sidebar.checkbox("Surowość restrykcji (stringency_index)")

    def draw_stat_controls(label_prefix):
        st.sidebar.markdown(f"Statystyki – {label_prefix}")
        return {
            "mean_local": st.sidebar.checkbox(f"Średnia krajowa – {label_prefix}"),
            "mean_global": st.sidebar.checkbox(f"Średnia globalna – {label_prefix}"),
            "median_local": st.sidebar.checkbox(f"Mediana krajowa – {label_prefix}"),
            "median_global": st.sidebar.checkbox(f"Mediana globalna – {label_prefix}")
        }

    stats_vacc = draw_stat_controls("Szczepienia") if show_vacc else {}
    stats_cases = draw_stat_controls("Nowe przypadki") if show_cases else {}
    stats_deaths = draw_stat_controls("Zgony") if show_deaths else {}
    stats_lockdown = draw_stat_controls("Stringency index") if show_lockdown else {}

elif selected_tab == "Porównanie szczepień":
    st.sidebar.header("Ustawienia – porównanie krajów")

    latest_vacc_data = df.sort_values("date").dropna(subset=["people_fully_vaccinated_per_hundred"])
    latest_vacc = latest_vacc_data.groupby("location").tail(1)
    top4 = latest_vacc.sort_values("people_fully_vaccinated_per_hundred", ascending=False)["location"].head(4).tolist()

    if "Poland" not in top4:
        top4.append("Poland")

    selected_countries = st.sidebar.multiselect(
        "Wybierz kraje do porównania:",
        sorted(df['location'].unique()),
        default=top4
    )

# -------------------------------
# TAB 1 - Trendy dla kraju
# -------------------------------
if selected_tab == "Trendy dla kraju":
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    country_df = df[(df['location'] == country) & (df['date'] >= start_date) & (df['date'] <= end_date)]

    def draw_stat_lines(ax, stats, local_series, global_series, color, prefix):
        if stats.get("mean_local") and not local_series.empty:
            ax.axhline(local_series.mean(), color=color, linestyle='--', linewidth=1, label=f"Średnia krajowa ({prefix})")
        if stats.get("mean_global") and not global_series.empty:
            ax.axhline(global_series.mean(), color=color, linestyle='--', linewidth=1.5, label=f"Średnia globalna ({prefix})")
        if stats.get("median_local") and not local_series.empty:
            ax.axhline(local_series.median(), color=color, linestyle=':', linewidth=1, label=f"Mediana krajowa ({prefix})")
        if stats.get("median_global") and not global_series.empty:
            ax.axhline(global_series.median(), color=color, linestyle=':', linewidth=1.5, label=f"Mediana globalna ({prefix})")

    def format_quarter(x, pos):
        date = mdates.num2date(x)
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year} - Q{quarter}"

    has_vacc = show_vacc and not country_df['people_fully_vaccinated_per_hundred'].dropna().empty
    has_cases = show_cases and not country_df['new_cases'].dropna().empty
    has_deaths = show_deaths and not country_df['new_deaths'].dropna().empty
    has_lockdown = show_lockdown and not country_df['stringency_index'].dropna().empty

    if not any([has_vacc, has_cases, has_deaths, has_lockdown]):
        st.warning("Brak danych do wyświetlenia dla wybranego kraju, zakresu dat i trendów.")
    else:
        st.subheader(f"Trendy COVID-19 – {country}")
        fig, ax = plt.subplots(figsize=(18, 12))

        if has_vacc:
            col = 'people_fully_vaccinated_per_hundred'
            ax.plot(country_df['date'], country_df[col], label="Pełne szczepienia (%)", color='green', linewidth=2)
            draw_stat_lines(ax, stats_vacc, country_df[col].dropna(), df[col].dropna(), 'green', "szczepienia")

        if has_cases:
            col = 'new_cases'
            smoothed = country_df[col].rolling(window=7, min_periods=1).mean()
            ax.plot(country_df['date'], smoothed, label="Nowe przypadki (śr. 7-dniowa)", color='steelblue', linewidth=1)
            draw_stat_lines(ax, stats_cases, smoothed.dropna(), df[col].rolling(window=7, min_periods=1).mean().dropna(), 'steelblue', "przypadki")

        if has_deaths:
            col = 'new_deaths'
            smoothed = country_df[col].rolling(window=7, min_periods=1).mean()
            ax.plot(country_df['date'], smoothed, label="Nowe zgony (śr. 7-dniowa)", color='red', linewidth=1)
            draw_stat_lines(ax, stats_deaths, smoothed.dropna(), df[col].rolling(window=7, min_periods=1).mean().dropna(), 'red', "zgony")

        if has_lockdown:
            col = 'stringency_index'
            ax.plot(country_df['date'], country_df[col], label="Stringency index", color='orange', linewidth=2)
            draw_stat_lines(ax, stats_lockdown, country_df[col].dropna(), df[col].dropna(), 'orange', "lockdown")
        elif show_lockdown:
            st.info("Brak danych o surowości restrykcji w wybranym zakresie dla tego kraju.")

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(format_quarter))
        plt.xticks(rotation=45)
        ax.set_xlabel("Data (kwartały)")
        ax.set_ylabel("Wartość")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# -------------------------------
# TAB 2 - Porównanie szczepień
# -------------------------------
elif selected_tab == "Porównanie szczepień":
    st.subheader("Trend szczepień – porównanie krajów")
    metric = 'people_fully_vaccinated_per_hundred'

    fig2, ax2 = plt.subplots(figsize=(18, 10))
    for c in selected_countries:
        data = df[df['location'] == c]
        if not data[metric].dropna().empty:
            ax2.plot(data['date'], data[metric], label=c)

    ax2.set_title("Tempo pełnych szczepień (% populacji)")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Pełne szczepienia (%)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# -------------------------------
# TAB 3 - Zależność: szczepienia vs zgony
# -------------------------------
elif selected_tab == "Zależność: szczepienia vs zgony":
    st.sidebar.header("Ustawienia – zależność szczepień i zgonów")

    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Wybór kraju
    country_options = ["(Wszystkie kraje)"] + sorted(df["location"].unique())
    selected_country = st.sidebar.selectbox("Wybierz kraj:", country_options, index=0)

    # Filtr po zakresie dat
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    if selected_country == "(Wszystkie kraje)":
        # Globalna analiza
        latest_data = df_filtered.sort_values("date").groupby("location").tail(1)
        valid = latest_data.dropna(subset=["people_fully_vaccinated_per_hundred", "total_deaths_per_million"])

        st.subheader("Zależność między poziomem szczepień a liczbą zgonów")
        st.caption("Każdy punkt to jeden kraj – pokazuje zależność między poziomem zaszczepienia a liczbą zgonów na milion.")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=valid,
            x="people_fully_vaccinated_per_hundred",
            y="total_deaths_per_million",
            hue="continent",
            ax=ax
        )
        sns.regplot(
            data=valid,
            x="people_fully_vaccinated_per_hundred",
            y="total_deaths_per_million",
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linestyle": "--"}
        )

        ax.set_xlabel("Pełne szczepienia (% populacji)")
        ax.set_ylabel("Zgony na milion mieszkańców")
        ax.grid(True)
        ax.legend(title="Kontynent", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    else:
        # Analiza pojedynczego kraju
        country_df = df_filtered[df_filtered["location"] == selected_country].dropna(subset=[
            "people_fully_vaccinated_per_hundred", "new_deaths_per_million"
        ])

        if country_df.empty:
            st.warning("Brak wystarczających danych dla wybranego kraju.")
        else:
            st.subheader(f"{selected_country} – Trend szczepień i zgonów")
            st.caption("Porównanie w czasie: poziom szczepień i zgony")

            fig, ax1 = plt.subplots(figsize=(14, 8))

            # Oś lewa: szczepienia
            ax1.plot(
                country_df['date'],
                country_df['people_fully_vaccinated_per_hundred'],
                color='green',
                label='Pełne szczepienia (%)',
                linewidth=2
            )
            ax1.set_ylabel("Szczepienia (% populacji)", color='green')
            ax1.tick_params(axis='y', labelcolor='green')

            # Oś prawa: zgony
            ax2 = ax1.twinx()
            ax2.plot(
                country_df['date'],
                country_df['new_deaths_per_million'],
                color='red',
                label='Zgony na mln (śr. 7-dniowa)',
                linewidth=2
            )
            ax2.set_ylabel("Zgony na milion", color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax1.set_xlabel("Data")
            ax1.grid(True)

            fig.tight_layout()
            st.pyplot(fig)

# -------------------------------
# TAB 4 - Zależność: szczepienia vs przypadki
# -------------------------------
elif selected_tab == "Zależność: szczepienia vs przypadki":
    st.sidebar.header("Ustawienia – szczepienia vs przypadki")

    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Wybór kraju
    country_options = ["(Wszystkie kraje)"] + sorted(df["location"].unique())
    selected_country = st.sidebar.selectbox("Wybierz kraj:", country_options, index=0)

    # Filtrowanie danych
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    if selected_country == "(Wszystkie kraje)":
        #Globalny wykres punktowy
        latest_data = df_filtered.sort_values("date").groupby("location").tail(1)
        valid = latest_data.dropna(subset=["people_fully_vaccinated_per_hundred", "new_cases_per_million"])

        st.subheader("Zależność między szczepieniami a liczbą przypadków")
        st.caption("Każdy punkt to jeden kraj – poziom wyszczepienia vs nowe przypadki.")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=valid,
            x="people_fully_vaccinated_per_hundred",
            y="new_cases_per_million",
            hue="continent",
            ax=ax
        )
        sns.regplot(
            data=valid,
            x="people_fully_vaccinated_per_hundred",
            y="new_cases_per_million",
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linestyle": "--"}
        )

        ax.set_xlabel("Pełne szczepienia (% populacji)")
        ax.set_ylabel("Nowe przypadki na mln")
        ax.grid(True)
        ax.legend(title="Kontynent", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    else:
        #Krajowy wykres dwuosiowy
        country_df = df_filtered[df_filtered["location"] == selected_country].dropna(subset=[
            "people_fully_vaccinated_per_hundred", "new_cases_per_million"
        ])

        if country_df.empty:
            st.warning("Brak wystarczających danych dla wybranego kraju.")
        else:
            st.subheader(f"{selected_country} – Szczepienia vs przypadki w czasie")
            st.caption("Porównanie w czasie: poziom szczepień i nowe przypadki")

            fig, ax1 = plt.subplots(figsize=(14, 8))

            # Szczepienia (lewa oś)
            ax1.plot(
                country_df['date'],
                country_df['people_fully_vaccinated_per_hundred'],
                color='green',
                label='Pełne szczepienia (%)',
                linewidth=2
            )
            ax1.set_ylabel("Szczepienia (% populacji)", color='green')
            ax1.tick_params(axis='y', labelcolor='green')

            # Przypadki (prawa oś)
            ax2 = ax1.twinx()
            ax2.plot(
                country_df['date'],
                country_df['new_cases_per_million'],
                color='steelblue',
                label='Nowe przypadki na mln',
                linewidth=2
            )
            ax2.set_ylabel("Nowe przypadki na milion", color='steelblue')
            ax2.tick_params(axis='y', labelcolor='steelblue')

            ax1.set_xlabel("Data")
            ax1.grid(True)
            fig.tight_layout()
            st.pyplot(fig)


# -------------------------------
# TAB 5 - Zależność: restrykcje vs pandemia
# -------------------------------
elif selected_tab == "Zależność: restrykcje vs pandemia":
    st.sidebar.header("Ustawienia – restrykcje vs pandemia")

    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Wybór wskaźnika pandemicznego
    target_metric = st.sidebar.radio(
        "Co analizować w relacji do restrykcji?",
        ["Nowe przypadki", "Nowe zgony"]
    )

    # Wybór kraju
    country_options = ["(Wszystkie kraje)"] + sorted(df["location"].unique())
    selected_country = st.sidebar.selectbox("Wybierz kraj:", country_options, index=0)

    # Dane filtrowane
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Mapa wskaźników
    metric_col = {
        "Nowe przypadki": "new_cases_per_million",
        "Nowe zgony": "new_deaths_per_million"
    }
    y_metric = metric_col[target_metric]

    if selected_country == "(Wszystkie kraje)":
        # Globalny scatter plot
        latest = df_filtered.sort_values("date").groupby("location").tail(1)
        valid = latest.dropna(subset=["stringency_index", y_metric])

        st.subheader(f"Zależność: surowość restrykcji vs {target_metric.lower()}")
        st.caption("Każdy punkt to jeden kraj – ostatni dostępny dzień z zakresu.")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=valid,
            x="stringency_index",
            y=y_metric,
            hue="continent",
            ax=ax
        )
        sns.regplot(
            data=valid,
            x="stringency_index",
            y=y_metric,
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linestyle": "--"}
        )

        ax.set_xlabel("Stringency Index (0–100)")
        ax.set_ylabel(f"{target_metric} na milion (śr. 7-dniowa)")
        ax.grid(True)
        ax.legend(title="Kontynent", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    else:
        # Wykres liniowy dla kraju
        country_df = df_filtered[df_filtered["location"] == selected_country].dropna(subset=[
            "stringency_index", y_metric
        ])

        if country_df.empty:
            st.warning("Brak danych dla wybranego kraju.")
        else:
            st.subheader(f"{selected_country} – Restrakcje vs {target_metric.lower()}")
            st.caption("Porównanie w czasie: surowość restrykcji i przebieg pandemii.")

            fig, ax1 = plt.subplots(figsize=(14, 8))

            ax1.plot(
                country_df["date"],
                country_df["stringency_index"],
                color="orange",
                label="Stringency index",
                linewidth=2
            )
            ax1.set_ylabel("Stringency index (0–100)", color='orange')
            ax1.tick_params(axis='y', labelcolor='orange')

            ax2 = ax1.twinx()
            ax2.plot(
                country_df["date"],
                country_df[y_metric],
                color="purple",
                label=target_metric,
                linewidth=2
            )
            ax2.set_ylabel(f"{target_metric} na milion", color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

            ax1.set_xlabel("Data")
            ax1.grid(True)
            fig.tight_layout()
            st.pyplot(fig)


# -------------------------------
# TAB 6 - Porównanie kontynentów
# -------------------------------
elif selected_tab == "Porównanie kontynentów":
    st.sidebar.header("Ustawienia – porównanie kontynentów")

    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Wybór opcji
    show_bar = st.sidebar.checkbox("Średni poziom (barplot)", value=True)
    show_box = st.sidebar.checkbox("Rozkład (boxplot)", value=False)
    show_stats = st.sidebar.checkbox("Pokaż statystyki opisowe", value=True)

    # Filtrowanie danych
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df_filtered = df_filtered.dropna(subset=["continent", "people_fully_vaccinated_per_hundred"])

    # Ostatnie wartości na kraj
    latest_per_country = df_filtered.sort_values("date").groupby("location").tail(1)

    st.subheader("Porównanie kontynentów wg poziomu wyszczepienia")
    st.caption("Możesz analizować średnie, rozkłady i statystyki opisowe dla poszczególnych kontynentów.")

    if show_bar:
        st.markdown("#### Średni poziom pełnych szczepień (barplot)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mean_vacc = latest_per_country.groupby("continent")["people_fully_vaccinated_per_hundred"].mean().sort_values(ascending=False)
        sns.barplot(x=mean_vacc.index, y=mean_vacc.values, ax=ax1, palette="viridis")
        ax1.set_ylabel("Średni poziom szczepień (%)")
        ax1.set_xlabel("Kontynent")
        ax1.set_title("Średni poziom pełnych szczepień wg kontynentów")
        ax1.grid(True)
        st.pyplot(fig1)

    if show_box:
        st.markdown("#### Rozkład poziomu szczepień (boxplot)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=latest_per_country,
            x="continent",
            y="people_fully_vaccinated_per_hundred",
            palette="Set2",
            ax=ax2
        )
        ax2.set_ylabel("Pełne szczepienia (% populacji)")
        ax2.set_xlabel("Kontynent")
        ax2.set_title("Rozkład pełnych szczepień wg kontynentów")
        ax2.grid(True)
        st.pyplot(fig2)

    if show_stats:
        st.sidebar.markdown("### Statystyki opisowe")
        summary = latest_per_country.groupby("continent")["people_fully_vaccinated_per_hundred"].agg(
            Liczba_krajów="count",
            Średnia="mean",
            Mediana="median",
            Minimum="min",
            Maksimum="max"
        ).round(2)

        for continent, row in summary.iterrows():
            st.sidebar.markdown(f"**{continent}**")
            st.sidebar.markdown(
                f"- Średnia: {row['Średnia']}\n"
                f"- Mediana: {row['Mediana']}\n"
                f"- Min: {row['Minimum']}\n"
                f"- Max: {row['Maksimum']}\n"
                f"- Liczba krajów: {int(row['Liczba_krajów'])}"
            )


# -------------------------------
# TAB 7 - Mapa świata
# -------------------------------
elif selected_tab == "Mapa świata":
    st.sidebar.header("Ustawienia – mapa świata")

    # Zakres dat
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.slider(
        "Zakres dat:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Wybór metryki
    metric_options = {
        "Pełne szczepienia (% populacji)": "people_fully_vaccinated_per_hundred",
        "Zgony na milion": "total_deaths_per_million",
        "Nowe przypadki na milion": "new_cases_per_million",
        "Surowość restrykcji (stringency index)": "stringency_index"
    }

    selected_metric_label = st.sidebar.selectbox("Wybierz metrykę do wyświetlenia:", list(metric_options.keys()))
    selected_metric = metric_options[selected_metric_label]

    # Filtrowanie i agregacja
    df_map = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df[selected_metric].notna())
    ]

    latest = df_map.groupby("location", as_index=False)[selected_metric].mean()
    latest = latest.merge(df[['location', 'iso_code']].drop_duplicates(), on='location')
    latest = latest.dropna(subset=["iso_code", selected_metric])
    latest = latest[latest["iso_code"].str.len() == 3]

    st.subheader(f"Mapa świata – {selected_metric_label}")
    st.caption("Wartości przedstawiają ostatnie dostępne dane z zakresu dat dla każdego kraju.")

    fig = px.choropleth(
        latest,
        locations="iso_code",
        color=selected_metric,
        hover_name="location",
        color_continuous_scale="Viridis",
        labels={selected_metric: selected_metric_label},
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)


