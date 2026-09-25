""""
Job Scraper using JobSpy
=========================
Scrapes job postings from LinkedIn and Indeed using the jobspy library.
Tracks already-seen postings to avoid duplicates across runs.

Usage:
    python job_scraper.py

Requirements:
    pip install python-jobspy
"""

import json
import os
import hashlib
from datetime import datetime

import pandas as pd
from jobspy import scrape_jobs

# ─────────────────────────────────────────────
#  CONFIG — Edit these before running
# ─────────────────────────────────────────────

KEYWORDS = [
    "python developer",
    "gis",
    "data scientist",
    "data analyst",
    "water",
    "cfm"
]

LOCATION        = "Chicago, IL"   # City, State — or "remote"
RESULTS_WANTED  = 100               # Max results per keyword per site
HOURS_OLD       = 88             # Only fetch jobs posted within this many hours
COUNTRY         = "USA"            # Country for Indeed (USA, UK, CA, etc.)
OUTPUT_CSV      = "jobs.csv"       # Output file
SEEN_IDS_FILE   = "seen_job_ids.json"

# ─────────────────────────────────────────────
#  DEDUPLICATION
# ─────────────────────────────────────────────

def load_seen_ids():
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_ids(seen_ids):
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(list(seen_ids), f)

def make_id(row):
    raw = f"{row.get('title','')}|{row.get('company','')}|{row.get('location','')}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()

# ─────────────────────────────────────────────
#  SCRAPER
# ─────────────────────────────────────────────

def scrape_keyword(keyword):
    print(f"\n🔍 Searching: '{keyword}'")
    try:
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin"],
            search_term=keyword,
            location=LOCATION,
            results_wanted=RESULTS_WANTED,
            hours_old=HOURS_OLD,
            country_indeed=COUNTRY,
            linkedin_fetch_description=False,  # Set True for full descriptions (slower)
            verbose=0,
        )
        print(f"   Found {len(jobs)} total listings")
        return jobs
    except Exception as e:
        print(f"   Error scraping '{keyword}': {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Job Scraper — LinkedIn & Indeed (via JobSpy)")
    print(f"  Keywords : {', '.join(KEYWORDS)}")
    print(f"  Location : {LOCATION}")
    print(f"  Hours old: {HOURS_OLD}h  |  Max per search: {RESULTS_WANTED}")
    print(f"  Output   : {OUTPUT_CSV}")
    print("=" * 55)

    seen_ids = load_seen_ids()
    all_new_jobs = []

    for keyword in KEYWORDS:
        df = scrape_keyword(keyword)

        if df.empty:
            continue

        # Tag with keyword and scrape time
        df["keyword"]    = keyword
        df["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Deduplicate against seen jobs
        df["_id"] = df.apply(make_id, axis=1)
        new_df = df[~df["_id"].isin(seen_ids)]
        dupes  = len(df) - len(new_df)

        seen_ids.update(new_df["_id"].tolist())
        all_new_jobs.append(new_df.drop(columns=["_id"]))

        print(f"   -> {len(new_df)} new  |  {dupes} skipped (already seen)")

    # Save results
    if all_new_jobs:
        combined = pd.concat(all_new_jobs, ignore_index=True)

        # Reorder columns — put the most useful ones first
        priority_cols = ["title", "company", "location", "date_posted", "job_type",
                         "salary_source", "min_amount", "max_amount", "currency",
                         "keyword", "site", "job_url", "scraped_at", "description"]
        existing_priority = [c for c in priority_cols if c in combined.columns]
        remaining = [c for c in combined.columns if c not in existing_priority]
        combined = combined[existing_priority + remaining]

        file_exists = os.path.exists(OUTPUT_CSV)
        combined.to_csv(OUTPUT_CSV, mode="a", header=not file_exists, index=False, encoding="utf-8")
        save_seen_ids(seen_ids)

        print(f"\n Saved {len(combined)} new job(s) to '{OUTPUT_CSV}'")
        print(f"   Total unique jobs tracked: {len(seen_ids)}")
    else:
        print("\n No new jobs found since last run.")

    print("\nTip: Schedule this script with cron (Mac/Linux) or Task Scheduler (Windows)")
    print("     to automatically check for new postings daily.")
    print("\n     Example cron (runs every morning at 8am):")
    print("     0 8 * * * /usr/bin/python3 /path/to/job_scraper.py")

if __name__ == "__main__":
    main()
