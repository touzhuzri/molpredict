"""从 PPDB 公开页面抓取带分类标签的农药结构。

标签来源：PPDB 的 atoz_fung / atoz_herb / atoz_insect 分类列表。
阳性 = 仅出现在 fungicide 列表；阴性 = 仅出现在 herbicide/insecticide 列表；
同时出现在杀菌剂和其他类别中的化合物视为 mixed，不进入训练集。

用法:
    python scripts/fetch_ppdb.py            # 全量（带网页缓存）
    python scripts/fetch_ppdb.py --limit 20 # 小规模检查
"""

import argparse
import csv
import html
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = "https://sitem.herts.ac.uk/aeru/ppdb/en"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external" / "ppdb"
PAGES = CACHE / "pages"
OUT = ROOT / "data" / "training" / "ppdb_labeled.csv"
CATEGORIES = {"fung": 1, "herb": 0, "insect": 0}
PROXY = os.environ.get("PPDB_PROXY", "")

handlers = [urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})] if PROXY else []
opener = urllib.request.build_opener(*handlers)
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    last = None
    for attempt in range(3):
        try:
            with opener.open(url, timeout=60) as r:
                data = r.read()
            if len(data) > 1000:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def parse_list(html_text: str) -> dict[str, str]:
    found = {}
    for m in re.finditer(
        r'href="Reports/(\d+)\.htm"[^>]*>(.*?)</a>', html_text, re.S | re.I
    ):
        name = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        if name:
            found[m.group(1)] = name
    return found


def cell_value(html_text: str, label: str) -> str:
    pattern = (
        r'<td class="row_header">' + re.escape(label) + r"</td>.*?"
        r'<td class="data1">(.*?)</td>'
    )
    m = re.search(pattern, html_text, re.S | re.I)
    if not m:
        return ""
    value = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return "" if value.lower() in ("no data", "-", "n/a") else value


def parse_page(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    name = html.unescape(title.group(1)).strip() if title else ""
    smiles = cell_value(text, "Isomeric SMILES") or cell_value(text, "Canonical SMILES")
    inchikey = cell_value(text, "International Chemical Identifier key (InChIKey)")
    return name, smiles, inchikey


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="每个类别最多处理多少条")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)

    lists = {}
    for category in CATEGORIES:
        list_path = CACHE / f"atoz_{category}.html"
        fetch(f"{BASE}/atoz_{category}.htm", list_path)
        lists[category] = parse_list(list_path.read_text(encoding="utf-8", errors="replace"))
        print(f"{category}: {len(lists[category])} 条")

    # 同一 report id 可出现在多个分类列表，记录其全部类别
    id_categories: dict[str, set[str]] = {}
    id_names: dict[str, str] = {}
    for category, entries in lists.items():
        for report_id, name in entries.items():
            id_categories.setdefault(report_id, set()).add(category)
            id_names.setdefault(report_id, name)

    ids = sorted(id_categories, key=int)
    if args.limit:
        kept = []
        for category in CATEGORIES:
            kept += [i for i in ids if category in id_categories[i]][: args.limit]
        ids = sorted(set(kept), key=int)

    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for report_id in ids:
            jobs[pool.submit(fetch, f"{BASE}/Reports/{report_id}.htm",
                             PAGES / f"{report_id}.html")] = report_id
        failed = 0
        for i, fut in enumerate(as_completed(jobs), 1):
            try:
                fut.result()
            except Exception as exc:
                failed += 1
                print(exc, file=sys.stderr)
            if i % 100 == 0:
                print(f"已下载 {i}/{len(ids)}")

    records, seen = [], {}
    skipped = {"no_smiles": 0, "bad_smiles": 0, "mixed": 0, "duplicate": 0}
    for report_id in ids:
        cats = id_categories[report_id]
        if "fung" in cats and len(cats) > 1:
            skipped["mixed"] += 1
            continue
        label = 1 if cats == {"fung"} else 0
        page = PAGES / f"{report_id}.html"
        if not page.exists():
            continue
        name, smiles, inchikey = parse_page(page)
        name = name or id_names[report_id]
        name = re.sub(r"\s*\(Ref:.*?\)\s*$", "", name).strip()
        if not smiles:
            skipped["no_smiles"] += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped["bad_smiles"] += 1
            continue
        computed = Chem.MolToInchiKey(mol)
        if not computed:
            skipped["bad_smiles"] += 1
            continue
        if computed in seen:
            skipped["duplicate"] += 1
            continue
        seen[computed] = True
        records.append({
            "name": name,
            "smiles": Chem.MolToSmiles(mol),
            "label": label,
            "category": "+".join(sorted(cats)),
            "source": "PPDB",
            "ppdb_id": report_id,
            "inchikey": computed,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "smiles", "label", "category", "source", "ppdb_id", "inchikey"
        ])
        writer.writeheader()
        writer.writerows(records)

    pos = sum(r["label"] for r in records)
    print(f"\n写入 {OUT}")
    print(f"总数 {len(records)}，阳性 {pos}，阴性 {len(records) - pos}")
    print(f"跳过: {skipped}，下载失败 {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
