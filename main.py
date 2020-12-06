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
    except (ValueError, IndexError):
        return None


def doi_label(date: datetime) -> str:
    try:
        return DOI_LABELS[DOI_DATES.index(date)]
    except (ValueError, IndexError):
        return None


START_DATE = doi_date("CEO 2")
END_DATE = datetime(2020, 12, 6)

N_DAYS = (END_DATE - START_DATE).days + 1

README_TITLE = "Glassdoor Data (UK, Full Time)"

PLOT_TIMELINE_NAME = "plot_timeline.png"

# variables and order to plot in the timeline
TIMELINE_KEYS = "stars recommends outlook ceo_opinion".split()

FILT_SHORT_HOURS = 24 * 365 / 52 * 2  # 2 weeks
FILT_LONG_HOURS = 24 * 365 / 12 * 2  # 2 months
FILT_ORDER = 2

Y_SCALE = 0.8

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

# discard values outside date range of interest
raw = raw[(START_DATE <= raw["date"]) & (raw["date"] <= END_DATE)]

# %% create markdown report

# calculate some statistics on raw data
ceo_2 = (doi_date("CEO 2") <= raw["date"]) & (raw["date"] < doi_date("CEO 3"))
ceo_3 = (doi_date("CEO 3") <= raw["date"])

ceo_2_opinion = raw["ceo_opinion"][ceo_2]
ceo_3_opinion = raw["ceo_opinion"][ceo_3]

# calulate stat bool arrays, labels, and positivity
stats = (
    (raw["stars"] == 5, "5 Stars", True),
    (raw["stars"] == 1, "1 Star", False),
    (raw["recommends"], "Recommend", True),
    (raw["outlook"] == 1, "Positive Outlook", True),
    (raw["outlook"] == -1, "Negative Outlook", False),
    (ceo_2_opinion == 1, "Approve CEO 2", True),
    (ceo_2_opinion == -1, "Disapprove CEO 2", False),
    (ceo_3_opinion == 1, "Approve CEO 3", True),
    (ceo_3_opinion == -1, "Disapprove CEO 3", False),
)

# references for colour indicators
good, ok, bad = "good ok bad".split()

# nicely formatted dates for report
start = format(START_DATE, "%Y-%m-%d")
start += f" ({label})" if (label := doi_label(START_DATE)) else ""

end = format(END_DATE, "%Y-%m-%d")
end += f" ({label})" if (label := doi_label(END_DATE)) else ""


def make_row(bools: np.ndarray, label: str, is_positive=True) -> str:
    def indicator(frac: float) -> str:
        if is_positive:
            colour = good if frac >= 2 / 3 else ok if frac >= 1 / 3 else bad
        else:
            colour = bad if frac >= 2 / 3 else ok if frac >= 1 / 3 else good

        # create Markdown image with link to colour indicator
        return f"![{colour}]"

    # calculate all statistics for row
    fracs = (
        bools.mean(), 
        bools.where(raw["technical"] == 1).mean(), 
        bools.where(raw["technical"] == 0).mean(), 
        bools.where(raw["employed"] == 1).mean(), 
        bools.where(raw["employed"] == 0).mean(),
    )

    # format results
    results = (f"{indicator(frac)} {100 * frac:.0f}%" for frac in fracs)

    # create row data separated by pipes
    return "|".join([label, str(len(bools)), *results])


# function for generating table in Markdown
def make_table() -> str:
    # create 10x10 colour indicators
    indicator_refs = (
        f"[{good}]: https://via.placeholder.com/10/0f0?text=+\n\n"
        f"[{ok}]: https://via.placeholder.com/10/ff0?text=+\n\n"
        f"[{bad}]: https://via.placeholder.com/10/f00?text=+\n"
    )

    # create table header
    header = (
        "Statistic|N|Overall|Technical|Non-technical|Employed|Ex-employee\n"
        "-|-|-|-|-|-|-"
    )

    # create generator for all rows
    rows = (make_row(bools, label, is_pos) for bools, label, is_pos in stats)

    # join everything with line feeds
    return "\n".join([indicator_refs, header, *rows])


# create file and write report
with open("README.md", "w") as readme_f:
    def write_section(section: str):
        readme_f.write(section + "\n\n")

    write_section(f"# {README_TITLE}")
    write_section(f"## {start} to {end}")
    write_section(make_table())
    write_section(f"![Timeline]({PLOT_TIMELINE_NAME})")

# %% remove out of range and nan data
in_date_range = (START_DATE <= raw["date"]) & (raw["date"] <= END_DATE)
clean_cols = ["date", *TIMELINE_KEYS]

clean = raw[in_date_range][clean_cols]

# separate duplicate dates by an hour (max 24 reviews per day)
dup_date_data = clean[clean["date"].duplicated(keep=False)]

for date, group in dup_date_data.groupby("date"):
    hourly_range = pd.date_range(date, periods=len(group), freq="H")

    clean["date"][group.index] = hourly_range

# ensure all dates are unique
assert not any(clean["date"].duplicated())

# infer some nan values from the stars column
missing = clean["recommends"].isna()
clean["recommends"][missing & (clean["stars"] >= 4)] = 1

missing = clean["outlook"].isna()
clean["outlook"][missing & (clean["stars"] >= 4)] = 1
clean["outlook"][missing & (clean["stars"] <= 2)] = -1

# fill remaining nan values with assumptions
clean["recommends"].fillna(value=0.5, inplace=True)
clean["outlook"].fillna(value=0, inplace=True)
clean["ceo_opinion"].fillna(value=0, inplace=True)

# %% interpolate to regular timebase of 1 day

# date indexes
start = clean["date"].min()
end = clean["date"].max()

clean_date = pd.DatetimeIndex(clean["date"])
interp_date = pd.date_range(start=start, end=end, freq="H")

# calculate numerical index to interpolate on
clean_delta_hours = (clean_date - start).total_seconds() / 3600
interp_delta_hours = (interp_date - start).total_seconds() / 3600


# function for performing interpolation
def do_interp(col: pd.Series) -> pd.Series:    
    return pd.Series(np.interp(interp_delta_hours, clean_delta_hours, col))


# remove date column, apply interpolation, set interpolated date index
interp = (
    clean
    .drop(columns="date")
    .apply(do_interp)
    .set_index(interp_date)
)

# %% low pass filter

# filter coefficients
b_short, a_short = butter(FILT_ORDER, 1 / FILT_SHORT_HOURS)
b_long, a_long = butter(FILT_ORDER, 1 / FILT_LONG_HOURS)


# functions for performing filtering
def filt_short(col: pd.Series) -> pd.Series:
    return pd.Series(filtfilt(b_short, a_short, col))


def filt_long(col: pd.Series) -> pd.Series:
    return pd.Series(filtfilt(b_long, a_long, col))


# apply filtering, set same interpolated date index as interp
lp_short = interp.apply(filt_short).set_index(interp.index)
lp_long = interp.apply(filt_long).set_index(interp.index)

# %% PLOT

# set up figure and axes
fig, axes = plt.subplots(5, 1, sharex=True)
stars_ax, recommends_ax, outlook_ax, ceo_opinion_ax, freq_ax = axes

fig.subplots_adjust(left=0.08, bottom=0.11, right=0.92, top=0.9, hspace=0)
fig.suptitle(
    f"{README_TITLE} Timeline\n"
    f"total reviews: {len(clean['date'])}, "
    f"date range [years]: {N_DAYS / 365:.1f}, "
    f"long filter [months]: {FILT_LONG_HOURS / 24 / 365 * 12:.1f}, "
    f"short filter [weeks]: {FILT_SHORT_HOURS / 24 / 7:.1f}"
)
fig.set_size_inches(20, 12)
fig.set_facecolor("white")

# use date of interest label else ISO date by default
filt_dates = (d for d in DOI_DATES if START_DATE < d < END_DATE)
xticks = (START_DATE, *filt_dates, END_DATE)
xticklabels = tuple(doi_label(d) or format(d, "%Y-%m-%d") for d in xticks)
xlim = [START_DATE - timedelta(days=1), END_DATE + timedelta(days=1)]

# create a generator with all axis dependant data, apart from review freq
plot_data = zip(
    axes,
    TIMELINE_KEYS,
    "Stars|Recommends|Outlook|CEO Opinion".split("|"),  # y labels
    ((1, 5), (0, 1), (-1, 1), (-1, 1)),  # y limits
    (  # y tick labels
        "1 2 3 4 5".split(),
        "No Yes".split(),
        "Negative Neutral Positive".split(),
        "Disapprove Neutral Approve".split()
    )
)

# loop to populate and style all data plots
for ax, key, title, ylim, ylabels in plot_data:
    y_mid = np.mean(ylim)

    def scale(data):
        return Y_SCALE * (data - y_mid) + y_mid

    raw_ = scale(raw[key])
    lp_short_ = scale(lp_short[key])
    lp_short_positive = np.where(lp_short_ > y_mid, lp_short_, np.nan)
    lp_long_ = scale(lp_long[key])
    yticks = scale(np.arange(ylim[0], ylim[1] + 1))

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
        lp_long.index, lp_long_, "grey",
        lp_short.index, lp_short_, "red",
        lp_short.index, lp_short_positive, "green",
        linewidth=1
    )
    
    # set axis properties
    ax.set(
        xticks=xticks,
        xlim=xlim,
    )
    
    # can't be set with ax.set()
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.grid()

    # create y axis on right hand side and add ticks and labels
    plt.setp(
        (ax, ax.twinx()),
        yticks=yticks,
        yticklabels=ylabels,
        ylabel=title,
        frame_on=False,
        ylim=ylim,
    )
    
    # add vertical lines at the dates of interest
    for date_, colour in zip(DOI_DATES, DOI_COLOURS):
        if START_DATE <= date_ <= END_DATE:
            ax.vlines(date_, *ylim, color=colour, linewidth=0.5)

# add legend to first plot
axes[0].legend(loc=(0, 1.05))

# plot review frequency
count = Counter(raw["date"])

n_weeks = N_DAYS // 7
n_4_weeks = n_weeks // 4

freq_ax.hist(count, bins=n_4_weeks, label="per 4 weeks")
freq_ax.hist(count, bins=n_weeks, label="per week")

freq_ax.set(
    xticks=xticks,
    xlim=xlim,
)

freq_ax.set_xticklabels(xticklabels, rotation=90)
freq_ax.grid()
freq_ax.legend(loc="upper left")

ymax = round(freq_ax.get_ylim()[1])
ylim = [0, ymax]
tick_step = ymax // 10  # 10 ticks no matter the range
plt.setp(
    (freq_ax, freq_ax.twinx()),
    ylim=ylim,
    yticks=range(0, ymax, tick_step),
    frame_on=False,
    ylabel="Review Frequency",
)

for date_, colour in zip(DOI_DATES, DOI_COLOURS):
    if START_DATE <= date_ <= END_DATE:
        freq_ax.vlines(date_, *ylim, color=colour, linewidth=0.5)

# save and show the figure
fig.savefig(PLOT_TIMELINE_NAME)

plt.show()
