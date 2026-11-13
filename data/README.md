# Data folder

The raw Kaggle file is **not committed** to this repository (see `.gitignore`).

## Option A — automatic (KaggleHub)

From the project root:

```powershell
python src/load_data.py
```

This downloads the public dataset once, caches it outside the repository,
locates `GlobalWeatherRepository.csv`, and copies it to
`data/GlobalWeatherRepository.csv`. It also prints a SHA-256 fingerprint.

If Kaggle asks for authentication, run this once:

```powershell
python -c "import kagglehub; kagglehub.login()"
```

Credentials are stored outside this repository. Never commit `kaggle.json`,
`.env`, or any API token.

> **If the download produces an empty 22-byte zip:** Kaggle currently marks
> this dataset as requiring a subscription/login. Anonymous downloads return
> an empty archive. Sign in to Kaggle and download the CSV from the dataset
> page, or authenticate KaggleHub with `kagglehub.login()`, then run
> `python src/load_data.py --force`.

## Option B — manual download

1. Open the Kaggle dataset page:
   <https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository>
2. Sign in to Kaggle and click **Download**.
3. Unzip the archive if needed.
4. Place the CSV here with exactly this name:

   ```
   data/GlobalWeatherRepository.csv
   ```

5. Do **not** download or commit `kaggle.json` or any API credentials.

## After downloading

Run the profiling script from the project root:

```powershell
python src/inspect_data.py
```

It does not modify the raw file. It prints a schema/quality profile and saves
a copy to `reports/data_profile.txt`.

Then create a small sample that can safely be committed:

```powershell
python src/make_sample.py
```

This writes `data/sample.csv` (a few hundred rows) which is allowed by
`.gitignore` and can be shared with reviewers.

## Processed data (generated, git-ignored)

Phase B creates:

- `data/processed/weather_clean_rows.csv.gz`
- `data/processed/weather_daily_city.csv.gz`
- `data/processed/weather_daily_global_core.csv`

Regenerate them with `python src/clean_data.py`. The raw CSV is never
overwritten; see `reports/cleaning_summary.txt` for every rule and count.

## Notes

- `last_updated` is the timestamp column we will use as the time index.
- The raw file must stay unchanged; all cleaning happens in code and is
  written to a separate output file.
- Check the Kaggle page for the dataset's current license and terms before
  redistributing anything beyond the small sample.
