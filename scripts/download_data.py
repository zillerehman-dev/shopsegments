"""Download the Online Retail dataset (UCI #352: 2010-12-01 -> 2011-12-09) as CSV.

Primary source: UCI ML Repository (xlsx). Fallback: a public CSV mirror of the same file.
Note: "Online Retail II" (UCI #502) contains this same 2010-2011 period as its second sheet.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_retail.csv"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
MIRROR_CSV = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
              "master/data/retail-data/all/online-retail-dataset.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"already exists: {OUT}")
        return
    try:
        import pandas as pd

        print("downloading from UCI ...")
        raw = urllib.request.urlopen(UCI_ZIP, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.endswith(".xlsx"))
            df = pd.read_excel(z.open(name), dtype={"InvoiceNo": str, "StockCode": str})
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%-m/%-d/%Y %-H:%M")
        df.to_csv(OUT, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"UCI download failed ({e}); using CSV mirror ...", file=sys.stderr)
        urllib.request.urlretrieve(MIRROR_CSV, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
