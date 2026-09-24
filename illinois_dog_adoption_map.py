from pathlib import Path
import json
import re
import sys

import pandas as pd
import requests
import plotly.express as px

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Illinois Department of Agriculture's current licensed animal shelter / animal
# control annual-report CSV. The Department says the report contains intake and
# outcome statistics for each licensed facility.
ILDOA_CSV = (
    "https://agr.illinois.gov/content/dam/soi/en/web/agr/animals/animalhealth/"
    "documents/animal_shelter_and_animal_control_facility.csv"
)

# U.S. Census TIGERweb county layer, filtered to Illinois (state FIPS 17).
CENSUS_GEOJSON = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "State_County/MapServer/3/query"
    "?where=STATE%3D%2717%27"
    "&outFields=GEOID,BASENAME,NAME"
    "&returnGeometry=true&outSR=4326&f=geojson"
)

COUNTY_ALIASES = {
    "county", "countyname", "county_name", "county name", "reportingcounty",
    "reporting county", "countyofoperation", "county of operation"
}

DOG_ADOPTION_ALIASES = {
    "dogadoptions", "dog_adoptions", "dogs adopted", "dogs_adopted",
    "dog adoption", "dog adoption(s)", "adoptions dogs", "adoptions - dogs",
    "dogs - adoptions", "dogs adopted out"
}

DOG_INTAKE_ALIASES = {
    "dogintake", "dog_intake", "dog intakes", "dog_intakes",
    "dog intake(s)", "intake dogs", "intakes - dogs", "dogs - intake",
    "dogs received", "dog intake total"
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_column(df, aliases, required=True):
    normalized = {norm(c): c for c in df.columns}
    for alias in aliases:
        key = norm(alias)
        if key in normalized:
            return normalized[key]
    # Fuzzy keyword fallback.
    candidates = []
    for c in df.columns:
        n = norm(c)
        if "dog" in n and any(k in n for k in ["adopt", "intake", "received"]):
            candidates.append(c)
    if aliases is DOG_ADOPTION_ALIASES:
        for c in candidates:
            if "adopt" in norm(c):
                return c
    if aliases is DOG_INTAKE_ALIASES:
        for c in candidates:
            if "intake" in norm(c) or "received" in norm(c):
                return c
    if required:
        raise KeyError(
            "Could not identify a required column. Available columns:\n"
            + "\n".join(map(str, df.columns))
        )
    return None


def clean_count(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
        errors="coerce"
    ).fillna(0)


def county_key(value):
    value = str(value).strip()
    value = re.sub(r"\s+county$", "", value, flags=re.I)
    return norm(value)


def download_sources():
    csv_path = DATA_DIR / "illinois_shelter_annual_report.csv"
    geo_path = DATA_DIR / "illinois_counties.geojson"

    if not csv_path.exists():
        print("Downloading Illinois Department of Agriculture annual report...")
        r = requests.get(ILDOA_CSV, timeout=60)
        r.raise_for_status()
        csv_path.write_bytes(r.content)

    if not geo_path.exists():
        print("Downloading Illinois county boundaries from U.S. Census TIGERweb...")
        r = requests.get(CENSUS_GEOJSON, timeout=60)
        r.raise_for_status()
        geo_path.write_bytes(r.content)

    return csv_path, geo_path


def load_and_aggregate(csv_path):
    # Keep everything as strings first because state annual-report files can
    # contain mixed numeric/text cells.
    df = pd.read_csv(csv_path, dtype=str, encoding_errors="replace")

    county_col = find_column(df, COUNTY_ALIASES)
    adoption_col = find_column(df, DOG_ADOPTION_ALIASES)
    intake_col = find_column(df, DOG_INTAKE_ALIASES)

    print(f"County column: {county_col}")
    print(f"Dog adoption column: {adoption_col}")
    print(f"Dog intake column: {intake_col}")

    work = pd.DataFrame({
        "county": df[county_col].astype(str).str.strip(),
        "dog_adoptions": clean_count(df[adoption_col]),
        "dog_intake": clean_count(df[intake_col]),
    })

    work = work[~work["county"].isin(["", "nan", "None"])]
    work["county_key"] = work["county"].map(county_key)

    agg = (
        work.groupby("county_key", as_index=False)
        .agg(
            dog_adoptions=("dog_adoptions", "sum"),
            dog_intake=("dog_intake", "sum"),
            reporting_facilities=("county", "size"),
        )
    )
    agg["dog_adoption_rate_pct"] = agg.apply(
        lambda r: (r.dog_adoptions / r.dog_intake * 100) if r.dog_intake else None,
        axis=1,
    )
    return agg


def build_map(agg, geo_path):
    geo = json.loads(geo_path.read_text(encoding="utf-8"))

    # Add normalized county keys to Census features for a clean join.
    for feature in geo["features"]:
        props = feature.get("properties", {})
        name = props.get("BASENAME") or props.get("NAME") or ""
        props["county_key"] = county_key(name)
        props["county_name"] = name

    geo_lookup = {
        f["properties"]["county_key"]: f["properties"].get("county_name", "")
        for f in geo["features"]
    }

    merged = pd.DataFrame({"county_key": list(geo_lookup.keys())})
    merged["county"] = merged["county_key"].map(geo_lookup)
    merged = merged.merge(agg, on="county_key", how="left")
    merged["dog_adoptions"] = merged["dog_adoptions"].fillna(0)
    merged["dog_intake"] = merged["dog_intake"].fillna(0)
    merged["reporting_facilities"] = merged["reporting_facilities"].fillna(0).astype(int)
    merged["dog_adoption_rate_pct"] = merged["dog_adoption_rate_pct"]

    merged.to_csv(OUTPUT_DIR / "illinois_county_dog_adoption.csv", index=False)

    fig = px.choropleth(
        merged,
        geojson=geo,
        locations="county_key",
        featureidkey="properties.county_key",
        color="dog_adoption_rate_pct",
        color_continuous_scale="YlGn",
        scope="usa",
        hover_name="county",
        custom_data=[
            "county",
            "dog_adoptions",
            "dog_intake",
            "reporting_facilities",
            "dog_adoption_rate_pct",
        ],
        labels={"dog_adoption_rate_pct": "Dog adoption rate (%)"},
        title="Illinois Dog Adoption by County",
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.6,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Dog adoption rate: %{customdata[4]:.1f}%<br>"
            "Dog adoptions: %{customdata[1]:,.0f}<br>"
            "Dog intake: %{customdata[2]:,.0f}<br>"
            "Reporting facilities: %{customdata[3]:,.0f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar_title="Adoption rate (%)",
    )

    html_path = OUTPUT_DIR / "illinois_dog_adoption_county_map.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True, config={"displaylogo": False})

    # Inject a clickable details panel into the exported HTML. Plotly's
    # plotly_click event exposes the same customdata used in the hover tooltip.
    html = html_path.read_text(encoding="utf-8")
    panel = """
<div id="county-details" style="font-family:Arial,sans-serif;max-width:900px;margin:12px auto;padding:14px 16px;border:1px solid #ddd;border-radius:10px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)">
  <strong>Click a county</strong> to see its dog adoption details.
</div>
<script>
(function(){
  var panel = document.getElementById('county-details');
  var gd = document.querySelector('.plotly-graph-div');
  if(!gd || !panel) return;
  gd.on('plotly_click', function(evt){
    var d = evt.points && evt.points[0] && evt.points[0].customdata;
    if(!d) return;
    var county = d[0];
    var adoptions = Number(d[1] || 0);
    var intake = Number(d[2] || 0);
    var facilities = Number(d[3] || 0);
    var rate = d[4] == null ? null : Number(d[4]);
    panel.innerHTML = '<strong>' + county + '</strong>' +
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:10px">' +
      '<div><small>Dog adoption rate</small><br><b>' + (rate == null ? 'N/A' : rate.toFixed(1) + '%') + '</b></div>' +
      '<div><small>Dog adoptions</small><br><b>' + adoptions.toLocaleString() + '</b></div>' +
      '<div><small>Dog intake</small><br><b>' + intake.toLocaleString() + '</b></div>' +
      '<div><small>Reporting facilities</small><br><b>' + facilities.toLocaleString() + '</b></div>' +
      '</div>';
  });
})();
</script>
"""
    html = html.replace("</body>", panel + "</body>")
    html_path.write_text(html, encoding="utf-8")

    return html_path, merged


def main():
    try:
        csv_path, geo_path = download_sources()
        agg = load_and_aggregate(csv_path)
        html_path, merged = build_map(agg, geo_path)
        print(f"\nCreated: {html_path}")
        print(f"Created: {OUTPUT_DIR / 'illinois_county_dog_adoption.csv'}")
        print(f"Counties in map: {len(merged)}")
    except Exception as exc:
        print("\nERROR:", exc)
        print("\nIf the Illinois source changed its column names, run this script once and use the printed column list to update the alias sets near the top of the file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
