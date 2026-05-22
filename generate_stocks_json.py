import os
import json
import twstock

def build_stock_metadata(stock_ids):

    results = []

    for sid in stock_ids:
        if sid not in twstock.codes:
            print(f"⚠️ skip {sid} (not found)")
            continue

        info = twstock.codes[sid]

        stock_data = {
            "id": sid,
            "name": info.name,
            "market": info.market,
            "group": info.group,
            "valid": None
        }

        results.append(stock_data)

    return {"stocks": results}

if __name__ == "__main__":
    STOCK_IDS = ["6770", "3481", "2344", "2485", "2367", "2337",
                 "2409", "4989", "2408", "2317", "3189", "2303",
                 "2313", "1785", "3006", "6182", "3576", "3049",
                 "3105", "2369", "1605", "2399", "2329", "6147",
                 "8096", "2353", "8021", "2495", "6443", "4958",
                 "4967", "2338", "0052", "2406", "5498", "3231",
                 "3702", "6282", "2455", "2327", "6274", "3714",
                 "5483", "2489", "4906", "4979", "6213", "2330",
                 "8112", "4919", "3701", "0050", "2324", "2301",
                 "3036", "3702", "3008", "2345", "2357", "2454",
                 "1303", "1519", "2467", "8064", "5536", "2308",
                 "6187", "2376", "6150", "2377", "3219", "2425",
                 "3661", "3515", "3540", "5386", "2481", "3491",
                 "2360", "2347", "3048"]

    data = build_stock_metadata(STOCK_IDS)

    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/stocks.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("✅ stocks.json generated!")