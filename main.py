# %% imports
from collections import Counter
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# %% constants

# dates of interest
DOI_LABELS, DOI_DATES, DOI_COLOURS = zip(
    ("2011", datetime(2011, 1, 1), "black"),
    ("CEO 1", datetime(2011, 9, 1), "black"),
    ("2012", datetime(2012, 1, 1), "black"),
    ("2013", datetime(2013, 1, 1), "black"),
    ("2014", datetime(2014, 1, 1), "black"),
    ("2015", datetime(2015, 1, 1), "black"),
    ("2016", datetime(2016, 1, 1), "black"),
    ("Brexit vote", datetime(2016, 6, 23), "black"),
    ("2017", datetime(2017, 1, 1), "black"),
    ("CEO 2", datetime(2017, 9, 1), "black"),
    ("Big project", datetime(2017, 9, 26), "black"),
    ("Sue CEO 1", datetime(2017, 11, 16), "black"),
    ("2018", datetime(2018, 1, 1), "black"),
    ("Expand UK", datetime(2018, 8, 30), "green"),
    ("2019", datetime(2019, 1, 1), "black"),
    ("No UK HQ", datetime(2019, 1, 22), "black"),
    ("2020", datetime(2020, 1, 1), "black"),
    ("1st UK Covid", datetime(2020, 1, 31), "black"),
    ("Lockdown 1", datetime(2020, 3, 23), "black"),
    ("CEO 3", datetime(2020, 4, 1), "black"),
    ("Bad press 1", datetime(2020, 5, 21), "red"),
    ("Job cuts", datetime(2020, 7, 23), "red"),
    ("Lockdown 2", datetime(2020, 11, 5), "black"),
    ("Bad press 2", datetime(2020, 11, 11), "red"),
    ("2021", datetime(2021, 1, 1), "black")
)

def doi_date(label: str) -> datetime:
    try:
        return DOI_DATES[DOI_LABELS.index(label)]
    except IndexError:
        return None

def doi_label(date_: datetime) -> str:
    try:
        return DOI_LABELS[DOI_DATES.index(date_)]
    except IndexError:
        return None

START_DATE = doi_date("Brexit vote")
END_DATE = doi_date("2021")

README_TITLE = "Glassdoor Data (UK, Full Time)"

# variables and order to plot in the timeline
TIMELINE_KEYS = "stars recommends outlook ceo_opinion".split()

FILT_SHORT_DAYS = 14  # 2 weeks
FILT_LONG_DAYS = 365 / 4  # 3 months
FILT_ORDER = 2

Y_SCALE = 0.8

PLOT_TIMELINE_NAME = "plot_timeline.png"

# %% get raw data from csv file
raw = pd.read_excel(
    io="data.xlsx",
    names=(
        "date",
        "stars", 
        "employed", 
        "technical",
        "recommends", 
        "outlook", 
        "ceo_opinion",
        "years"
    ),
    parse_dates=["date"],
    date_parser=datetime.fromisoformat
)

# Excel file should be in descending date order
assert all(a >= b for a, b in zip(raw["date"][:-1], raw["date"][1:]))

# ensure all star ratings are present
assert not any(raw["stars"].isna())

# order dates in ascending order
raw.sort_values("date", inplace=True)

# %% remove out of range and nan data
in_date_range = (START_DATE <= raw["date"]) & (raw["date"] <= END_DATE)
clean = raw[in_date_range][["date", *TIMELINE_KEYS]]

# infer some nan values from the stars column
missing = clean["recommends"].isna()
clean["recommends"][missing & (clean["stars"] >= 4)] = 1

missing = clean["outlook"].isna()
clean["outlook"][missing & (clean["stars"] >= 4)] = 1
clean["outlook"][missing & (clean["stars"] <= 2)] = -1

# fill remaining nan values with assumptions
for key in "recommends outlook ceo_opinion".split():
    clean[key].fillna(value=0, inplace=True)

# %% interpolate to regular timebase of 1 day

# date indexes
start = clean["date"].min()
end = clean["date"].max()
clean_date = pd.DatetimeIndex(clean["date"])
interp_date = pd.date_range(start=start, end=end, freq="D")

# calculate numerical index to interpolate on
clean_delta_days = (clean_date - start).days
interp_delta_days = (interp_date - start).days

# function for performing interpolation
def do_interp(col: pd.Series) -> pd.Series:    
    return pd.Series(np.interp(interp_delta_days, clean_delta_days, col))

# remove date column, apply interpolation, set interpolated date index
interp = clean.drop(columns="date").apply(do_interp).set_index(interp_date)

# %% low pass filter

# filter coefficients
b_short, a_short = butter(FILT_ORDER, 1 / FILT_SHORT_DAYS)
b_long, a_long = butter(FILT_ORDER, 1 / FILT_LONG_DAYS)

# functions for performing filtering
def filt_short(col: pd.Series) -> pd.Series:
    return pd.Series(filtfilt(b_short, a_short, col))

def filt_long(col: pd.Series) -> pd.Series:
    return pd.Series(filtfilt(b_long, a_long, col))

# apply filtering, set same interpolated date index as interp
lp_short = interp.apply(filt_short).set_index(interp.index)
lp_long = interp.apply(filt_long).set_index(interp.index)

# %% create markdown report

# calculate some statistics
ceo_2_start = doi_date("CEO 2")
ceo_2_end = doi_date("CEO 3")
ceo_2_opinion = raw["ceo_opinion"].where(
    (ceo_2_start < raw["date"]) & (raw["date"] <= ceo_2_end)
)

ceo_3_start = ceo_2_end
ceo_3_end = END_DATE 
ceo_3_opinion = raw["ceo_opinion"].where(
    (ceo_3_start < raw["date"]) & (raw["date"] <= ceo_3_end)
)

# function for generating table in Markdown
def make_table() -> str:
    def make_row(bools: np.ndarray, label: str) -> str:
        def perc(data: np.ndarray) -> str:
            return f"{100 * data.mean():.0f}%"

        return "|".join([
            label,
            perc(bools), 
            perc(bools.where(raw["technical"] == 1)), 
            perc(bools.where(raw["technical"] == 0)), 
            perc(bools.where(raw["employed"] == 1)), 
            perc(bools.where(raw["employed"] == 0)),
        ])

    return "\n".join([
        "Statistic|Total|Technical|Non-technical|Employed|Ex-employee",
        "-|-|-|-|-|-",
        make_row(raw["stars"] == 5, "5 Stars"),
        make_row(raw["stars"] == 1, "1 Star"),
        make_row(raw["recommends"], "Recommend"),
        make_row(raw["outlook"] == 1, "Positive Outlook"),
        make_row(raw["outlook"] == -1, "Negative Outlook"),
        make_row(ceo_2_opinion == 1, "Approve CEO 2"),
        make_row(ceo_2_opinion == -1, "Disapprove CEO 2"),
        make_row(ceo_3_opinion == 1, "Approve CEO 3"),
        make_row(ceo_3_opinion == -1, "Disapprove CEO 3"),
    ])

# create file and write report
with open("README.md", "w") as readme_f:
    readme_f.write(f"# {README_TITLE}\n\n")
    readme_f.write(f"## {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d}\n\n")
    readme_f.write(make_table() + "\n\n")
    readme_f.write(f"![Timeline]({PLOT_TIMELINE_NAME})\n\n")

# %% PLOT

# set up figure and axes
fig, axes = plt.subplots(5, 1, sharex=True)
stars_ax, recommends_ax, outlook_ax, ceo_opinion_ax, freq_ax = axes
fig.subplots_adjust(left=0.08, bottom=0.11, right=0.92, top=0.9, hspace=0)
fig.suptitle(
    f"{README_TITLE} Timeline\n"
    f"total reviews: {len(clean['date'])}, "
    f"date range [weeks]: {(END_DATE - START_DATE).days / 7:.1f}, "
    f"short filter [weeks]: {FILT_SHORT_DAYS / 7:.1f}, "
    f"long filter [weeks]: {FILT_LONG_DAYS / 7:.1f}"
)
fig.set_size_inches(20, 12)
fig.set_facecolor("white")

# use date of interest label else ISO date by default
filt_dates = (d for d in DOI_DATES if START_DATE < d < END_DATE)
xticks = (START_DATE, *filt_dates, END_DATE)
xticklabels = tuple(doi_label(d) or d.isoformat() for d in xticks)
xlim = [min(xticks) - timedelta(days=1), max(xticks) + timedelta(days=1)]

# create a generator with all axis dependant data
plot_data = zip(
    axes,  # handle review frequency axis separately
    TIMELINE_KEYS,
    "Stars|Recommends|Outlook|CEO Opinion".split("|"),  # y labels
    ((1, 5), (0, 1), (-1, 1), (-1, 1)),  # y limits
    (  # y tick labels
        "1 2 3 4 5".split(),
        "No Yes".split(),
        "Negative Neutral Positive".split(),
        "Disapproves Neutral Approves".split()
    )
)

# loop to populate and style all data plots
for ax, key, title, ylim, ylabels in plot_data:
    y_mid = np.mean(ylim)
    raw_ = Y_SCALE * (raw[key] - y_mid) + y_mid
    clean_ = Y_SCALE * (clean[key] - y_mid) + y_mid
    lp_short_ = Y_SCALE * (lp_short[key] - y_mid) + y_mid
    lp_long_ = Y_SCALE * (lp_long[key] - y_mid) + y_mid
    yticks = Y_SCALE * (np.arange(ylim[0], ylim[1] + 1) - y_mid) + y_mid

    # plot all series
    ax.scatter(  # raw employed
        raw["date"], raw_.where(raw["employed"] == 1), 
        marker="x", linewidth=0.5, label="Employed (raw)"
    )
    ax.scatter(  # raw ex-employee
        raw["date"], raw_.where(raw["employed"] == 0), 
        marker="x", linewidth=0.5, label="Ex-employee (raw)"
    )
    ax.plot(  # filtered data
        interp.index, lp_long_, "grey",
        interp.index, lp_short_, "red",
        interp.index, np.where(lp_short_ > y_mid, lp_short_, np.nan), "green",
        linewidth=1
    )
        
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_xlim(xlim)
    ax.grid()

    # add axis ticks and labels
    ax2 = ax.twinx()
    for a in (ax, ax2):
        a.set_yticks(yticks)
        a.set_yticklabels(ylabels)
        a.set_ylabel(title)
        a.set_frame_on(False)
        a.set_ylim(ylim)
    
    # add vertical lines at the dates of interest
    for d, c in zip(DOI_DATES, DOI_COLOURS):
        ax.vlines(d, *ylim, color=c, linewidth=0.5)

# add legend to first plot
axes[0].legend(loc=(0, 1.05))

# plot review frequency
count = Counter(raw["date"])
n_days = (END_DATE - START_DATE).days + 1
n_weeks = n_days // 7
n_4_weeks = n_weeks // 4

freq_ax.hist(count, bins=n_4_weeks, label="per 4 weeks")
freq_ax.hist(count, bins=n_weeks, label="per week")

freq_ax.grid()
freq_ax.legend(loc="upper left")
freq_ax.set_xticks(xticks)
freq_ax.set_xticklabels(xticklabels, rotation=90)
freq_ax.set_xlim(xlim)

ymax = round(freq_ax.get_ylim()[1])
ylim = [0, ymax]
tick_step = ymax // 10  # 10 ticks no matter the range
for ax in (freq_ax, freq_ax.twinx()):
    ax.set_ylim(ylim)
    ax.set_yticks(range(0, ymax, tick_step))
    ax.set_frame_on(False)
    ax.set_ylabel("Review Frequency")

for d, c in zip(DOI_DATES, DOI_COLOURS):
    freq_ax.vlines(d, *ylim, color=c, linewidth=0.5)

# save and show the figure
fig.savefig(PLOT_TIMELINE_NAME)

plt.show()
