# %% imports
from collections import namedtuple
from copy import deepcopy
from datetime import date, timedelta
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# %% constants

# dates of interest
DOI_LABELS, DOI_DATES, DOI_COLOURS = zip(
    ("2011", date(2011, 1, 1), "black"),
    ("CEO 1", date(2011, 9, 1), "black"),
    ("2012", date(2012, 1, 1), "black"),
    ("2013", date(2013, 1, 1), "black"),
    ("2014", date(2014, 1, 1), "black"),
    ("2015", date(2015, 1, 1), "black"),
    ("2016", date(2016, 1, 1), "black"),
    ("Brexit vote", date(2016, 6, 23), "black"),
    ("2017", date(2017, 1, 1), "black"),
    ("CEO 2", date(2017, 9, 1), "black"),
    ("Big project", date(2017, 9, 26), "black"),
    ("Sue CEO 1", date(2017, 11, 16), "black"),
    ("2018", date(2018, 1, 1), "black"),
    ("2019", date(2019, 1, 1), "black"),
    ("No UK HQ", date(2019, 1, 22), "black"),
    ("2020", date(2020, 1, 1), "black"),
    ("1st UK Covid", date(2020, 1, 31), "black"),
    ("Lockdown 1", date(2020, 3, 23), "black"),
    ("CEO 3", date(2020, 4, 1), "black"),
    ("Bad press 1", date(2020, 5, 21), "red"),
    ("Job cuts", date(2020, 7, 23), "red"),
    ("Lockdown 2", date(2020, 11, 5), "black"),
    ("Bad press 2", date(2020, 11, 11), "red"),
    ("2021", date(2021, 1, 1), "black")
)

def doi_date(label: str) -> date:
    return DOI_DATES[DOI_LABELS.index(label)]

def doi_label(date_: date) -> str:
    return DOI_LABELS[DOI_DATES.index(date_)]

START_DATE = doi_date("CEO 2")
END_DATE = doi_date("2021")

README_TITLE = "Glassdoor Data (UK, Full Time)"

# variables and order to plot in the timeline
TIMELINE_KEYS = "stars recommends outlook ceo_opinion".split()

FILT_SHORT_DAYS = 14  # 2 weeks
FILT_LONG_DAYS = 365 / 4  # 3 months
FILT_ORDER = 2

Y_SCALE = 0.8

PLOT_TIMELINE_NAME = "plot_timeline.png"
PLOT_REVIEW_FREQ_NAME = "plot_review_freq.png"

# %% get raw data from csv file
df = pd.read_excel("data.xlsx").iloc[::-1]  # reverse to be ascending

# %% ensure ordered
iso = df["date"].apply(date.fromisoformat)
assert all(prev <= curr for prev, curr in zip(iso[:-1], iso[1:]))

# %% remove out of bounds rows
df = df[(START_DATE <= iso) & (iso <= END_DATE)]
raw = {
    "date": np.array([date.fromisoformat(x) for x in df["date"]]),
    "technical": df["is technical"],
    "employed": df["is employed"],
    "stars": df["stars"],
    "recommends": df["recommends"],
    "outlook": df["outlook"],
    "ceo_opinion": df["ceo opinion"],
    "years": df["years employed"]
}

# %% copy into variables to fill nan values
clean = deepcopy(raw)

# infer some nan values from other columns
clean["recommends"][clean["recommends"].isna() & (clean["outlook"] <= 0)] = 0
clean["recommends"][clean["recommends"].isna() & clean["outlook"].notna()] = 1
clean["recommends"][clean["recommends"].isna() & (clean["stars"] <= 3)] = 0
clean["recommends"][clean["recommends"].isna() & clean["stars"].notna()] = 1

clean["outlook"][clean["outlook"].isna() & clean["recommends"]] = 1
clean["outlook"][clean["outlook"].isna() & clean["recommends"].notna()] = 0
clean["outlook"][clean["outlook"].isna() & (clean["stars"] <= 2)] = -1
clean["outlook"][clean["outlook"].isna() & (clean["stars"] >= 4)] = 1
clean["outlook"][clean["outlook"].isna() & clean["stars"].notna()] = 0

# fill remaining nan values with assumptions
clean["technical"] = clean["technical"].fillna(value=0).astype(bool)
clean["employed"] = clean["employed"].fillna(value=0).astype(bool)
clean["stars"] = clean["stars"].fillna(value=3)
clean["recommends"] = clean["recommends"].fillna(value=0).astype(bool)
clean["outlook"] = clean["outlook"].fillna(value=0)
clean["ceo_opinion"] = clean["ceo_opinion"].fillna(value=0)
clean["years"] = clean["years"].fillna(value=0)

# %% create markdown report
ceo_2_start = doi_date("CEO 2")
ceo_2_end = doi_date("CEO 3")
ceo_2_opinion = clean["ceo_opinion"].where(
    (ceo_2_start < clean["date"]) & (clean["date"] <= ceo_2_end)
)

ceo_3_start = ceo_2_end
ceo_3_end = END_DATE 
ceo_3_opinion = clean["ceo_opinion"].where(
    (ceo_3_start < clean["date"]) & (clean["date"] <= ceo_3_end)
)

def make_table() -> str:
    def make_row(bools: np.ndarray, label: str) -> str:
        def perc(data: np.ndarray) -> str:
            return f"{100 * data.mean():.0f}%"

        return "|".join([
            label,
            perc(bools), 
            perc(bools[clean['technical']]), 
            perc(bools[~clean['technical']]), 
            perc(bools[clean['employed']]), 
            perc(bools[~clean['employed']]),
        ])

    return "\n".join([
        "Statistic|Total|Technical|Non-technical|Employed|Ex-employee",
        "-|-|-|-|-|-",
        make_row(clean["stars"] == 5, "5 Stars"),
        make_row(clean["stars"] == 1, "1 Star"),
        make_row(clean["recommends"], "Recommend"),
        make_row(clean["outlook"] == 1, "Positive Outlook"),
        make_row(clean["outlook"] == -1, "Negative Outlook"),
        make_row(ceo_2_opinion == 1, "Approve CEO 2"),
        make_row(ceo_2_opinion == -1, "Disapprove CEO 2"),
        make_row(ceo_3_opinion == 1, "Approve CEO 3"),
        make_row(ceo_3_opinion == -1, "Disapprove CEO 3"),
    ])

with open("README.md", "w") as readme_f:
    readme_f.write(f"# {README_TITLE}\n\n")
    readme_f.write(f"## {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d}\n\n")
    readme_f.write(make_table() + "\n\n")
    readme_f.write(f"![Timeline]({PLOT_TIMELINE_NAME})\n\n")
    readme_f.write(f"![Review frequency]({PLOT_REVIEW_FREQ_NAME})\n\n")

# %% interpolate to regular timebase of 1 day
interp = {"date": [clean["date"][0]]}
while interp["date"][-1] < clean["date"][-1]:
    interp["date"].append(interp["date"][-1] + timedelta(days=1))

delta_s = [(x - clean["date"][0]).days for x in clean["date"]]
interp_delta_s = [(x - interp["date"][0]).days for x in interp["date"]]

interp.update({
    key: np.interp(interp_delta_s, delta_s, clean[key])
    for key in TIMELINE_KEYS
})

# %% low pass filter
b_short, a_short = butter(FILT_ORDER, 1 / FILT_SHORT_DAYS)
b_long, a_long = butter(FILT_ORDER, 1 / FILT_LONG_DAYS)

lp = {"short": {}, "long": {}}
for key in TIMELINE_KEYS:
    lp["short"][key] = filtfilt(b_short, a_short, interp[key])
    lp["long"][key] = filtfilt(b_long, a_long, interp[key])

# %% set up figure and plot data
fig, axes = plt.subplots(4, 1, sharex=True)
stars_ax, recommends_ax, outlook_ax, ceo_opinion_ax = axes
fig.subplots_adjust(left=0.08, bottom=0.11, right=0.92, top=0.9, hspace=0)
fig.suptitle(
    f"{README_TITLE} Timeline\n"
    f"total reviews: {len(raw['date'])}, "
    f"date range [days]: {(END_DATE - START_DATE).days}, "
    f"short filter [weeks]: {FILT_SHORT_DAYS / 7:.1f}, "
    f"long filter [months]: {FILT_LONG_DAYS * 12 / 365:.1f}"
)
fig.set_size_inches(20, 10)
fig.set_facecolor("white")

filt_dates = (
    d for d in DOI_DATES
    if START_DATE < d < END_DATE
)
xticks = (START_DATE, *filt_dates, END_DATE)
# use date of interest label if exists else use ISO date by default
xticklabels = tuple(
    doi_label(d) if d in DOI_DATES else d.isoformat()
    for d in xticks
)

# create a generator with all axis dependant data
plot_data = zip(
    axes,
    (raw[k] for k in TIMELINE_KEYS),
    (clean[k] for k in TIMELINE_KEYS),
    (lp["short"][k] for k in TIMELINE_KEYS),
    (lp["long"][k] for k in TIMELINE_KEYS),
    "Stars Recommends Outlook CEO".split(),  # y labels
    ((1, 5), (0, 1), (-1, 1), (-1, 1)),  # y limits
    (  # y tick labels
        "1 2 3 4 5".split(),
        "No Yes".split(),
        "Negative Neutral Positive".split(),
        "Disapproves Neutral Approves".split()
    )
)

# loop to populate and style all plots
for ax, raw_, clean_, lp_short, lp_long, title, ylim, ylabels in plot_data:
    y_mid = np.mean(ylim)
    raw_ = Y_SCALE * (raw_ - y_mid) + y_mid
    clean_ = Y_SCALE * (clean_ - y_mid) + y_mid
    lp_short = Y_SCALE * (lp_short - y_mid) + y_mid
    lp_long = Y_SCALE * (lp_long - y_mid) + y_mid
    yticks = Y_SCALE * (np.arange(ylim[0], ylim[1] + 1) - y_mid) + y_mid

    # plot all series
    ax.scatter(  # raw employed
        raw["date"], raw_.where(clean["employed"]), 
        marker="x", linewidth=0.5, label="Employed (raw)"
    )
    ax.scatter(  # raw ex-employee
        raw["date"], raw_.where(~clean["employed"]), 
        marker="x", linewidth=0.5, label="Ex-employee (raw)"
    )
    ax.plot(  # filtered data
        interp["date"], lp_long, "grey",
        interp["date"], lp_short, "red",
        interp["date"], np.where(lp_short > y_mid, lp_short, np.nan), "green",
        linewidth=1
    )
    
    # add vertical lines at the dates of interest
    for d, c in zip(DOI_DATES, DOI_COLOURS):
        ax.vlines(d, *ylim, color=c, linewidth=0.5)

    # add axis ticks and labels
    ax2 = ax.twinx()
    for a in (ax, ax2):
        a.set_yticks(yticks)
        a.set_yticklabels(ylabels)
        a.set_ylabel(title)
        a.set_frame_on(False)
        a.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_xlim(
        [min(xticks) - timedelta(days=1), max(xticks) + timedelta(days=1)]
    )
    ax.grid()
else:
    # add legend to first plot
    axes[0].legend(loc=(0, 1.05))

    # save the figure
    fig.savefig(PLOT_TIMELINE_NAME)

    plt.show()

# %% plot review frequency

from collections import Counter
from math import ceil

count = Counter(raw["date"])

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
fig.set_facecolor("white")
fig.suptitle("Review frequency")
fig.subplots_adjust(left=0.02, bottom=0.11, right=0.98, top=0.9)

days = (END_DATE - START_DATE).days
ax.hist(count, bins=int(days * 12 / 365), label="Per month")
ax.hist(count, bins=int(days * 52 / 365), label="Per week")
ax.hist(count, bins=int(days), label="Per day")

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_xlim(
    [min(xticks) - timedelta(days=1), max(xticks) + timedelta(days=1)]
)
ymax = ceil(ax.get_ylim()[1])
ax.set_ylim([0, ymax])
tick_step = 2
ax.set_yticks(range(0, ymax + tick_step, tick_step))
ax.grid()
ax.legend()

fig.savefig(PLOT_REVIEW_FREQ_NAME)

plt.show()
