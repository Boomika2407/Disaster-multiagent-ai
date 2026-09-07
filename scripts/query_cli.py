import sys
import os
import json
import yaml
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_latest_reports(reports_dir: Path):
    """Finds all JSON assessments in reports directory."""
    if not reports_dir.exists():
        return {}
    reports = {}
    for json_file in reports_dir.glob("*_assessment.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                zone_id = data.get("zone_id", json_file.stem.replace("_assessment", ""))
                reports[zone_id] = data
        except Exception:
            continue
    return reports


def answer_query(query: str, reports: dict) -> str:
    """Answers operational questions across processed disaster zones."""
    query_lower = query.lower()

    if not reports:
        return "No processed zone reports found in outputs/reports/. Run scripts/run_coordinator.py first."

    # Status / overview
    if any(k in query_lower for k in ["status", "overview", "summary", "zones"]):
        lines = [f"### 📋 Multi-Zone Situational Overview ({len(reports)} Zones Processed)"]
        for zid, r in reports.items():
            fl = r.get("flood_assessment", {})
            dmg = r.get("building_damage_assessment", {})
            geo = r.get("geospatial_logistics", {})
            lines.append(
                f"- **Zone `{zid}`**: Flood: `{fl.get('flood_severity', 'N/A')}` | "
                f"Road Access: `{geo.get('access_status', 'N/A')}` | "
                f"Damage Score: `{dmg.get('damage_severity_score', 'N/A')}/100`"
            )
        return "\n".join(lines)

    # Road access / transit
    if any(k in query_lower for k in ["road", "access", "transit", "vehicle", "boat", "impassable"]):
        lines = ["### 🛣️ Road Access & Logistics Status:"]
        for zid, r in reports.items():
            geo = r.get("geospatial_logistics", {})
            lines.append(f"\n**Zone `{zid}`**:")
            lines.append(f"  - Status: `{geo.get('access_status', 'N/A')}` (Score: {geo.get('accessibility_score', 'N/A')}/100)")
            lines.append(f"  - Flooded Roads: {geo.get('road_metrics', {}).get('flooded_road_percentage', 'N/A')}%")
            lines.append(f"  - Recommended Transit: {', '.join(geo.get('recommended_transit', []))}")
            lines.append(f"  - Landing Zone Viable: {geo.get('logistics', {}).get('landing_zone_viable', False)}")
        return "\n".join(lines)

    # Damage query
    if any(k in query_lower for k in ["damage", "destroyed", "building", "structural"]):
        lines = ["### 🏗️ Structural Building Damage Breakdown:"]
        for zid, r in reports.items():
            dmg = r.get("building_damage_assessment", {})
            dist = dmg.get("building_damage_distribution", {})
            lines.append(f"\n**Zone `{zid}`**:")
            lines.append(f"  - Dominant Level: `{dmg.get('dominant_damage_level', 'N/A')}`")
            lines.append(f"  - Severity Score: `{dmg.get('damage_severity_score', 'N/A')}/100`")
            lines.append(f"  - Destroyed: {dist.get('Destroyed', 0.0)}% | Major: {dist.get('Major-Damage', 0.0)}% | Minor: {dist.get('Minor-Damage', 0.0)}%")
        return "\n".join(lines)

    # Flood query
    if any(k in query_lower for k in ["flood", "water", "inundation"]):
        lines = ["### 🌊 Flood & Inundation Status:"]
        for zid, r in reports.items():
            fl = r.get("flood_assessment", {})
            pcts = fl.get("class_pixel_percentages", {})
            lines.append(f"\n**Zone `{zid}`**:")
            lines.append(f"  - Severity: `{fl.get('flood_severity', 'N/A')}`")
            lines.append(f"  - Road Flooded: {pcts.get('Road-Flooded', 0.0)}% | Building Flooded: {pcts.get('Building-Flooded', 0.0)}% | Water: {pcts.get('Water', 0.0)}%")
        return "\n".join(lines)

    # Fallback to general zone query
    for zid, r in reports.items():
        if zid.lower() in query_lower:
            return f"### Report for Zone {zid}:\n\n" + r.get("sitrep_markdown", "No markdown SitRep found.")

    return "Query did not match specific filters. Try asking about: 'status', 'road access', 'building damage', 'flooding', or a specific zone ID."


def main():
    config = load_config()
    reports_dir = PROJECT_ROOT / config.get("outputs", {}).get("reports_dir", "outputs/reports")
    reports = find_latest_reports(reports_dir)

    print("=" * 60)
    print("📡 Disaster Multi-Agent AI — Interactive Command CLI")
    print("=" * 60)
    print(f"Loaded {len(reports)} zone assessment reports from {reports_dir}")
    print("Type your questions (e.g., 'What is the road access status?', 'Show damage breakdown', 'overview', or 'exit')\n")

    while True:
        try:
            query = input("Commander > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting CLI.")
                break
            ans = answer_query(query, reports)
            print(f"\n{ans}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break


if __name__ == "__main__":
    main()
