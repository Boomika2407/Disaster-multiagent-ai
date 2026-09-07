import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

try:
    from agents.flood_segmentation.infer import FloodSegmentationAgent
    from agents.building_damage.infer import BuildingDamageAgent
    from agents.geospatial_reasoning.agent import GeospatialReasoningAgent
except ImportError:
    from flood_segmentation.infer import FloodSegmentationAgent
    from building_damage.infer import BuildingDamageAgent
    from geospatial_reasoning.agent import GeospatialReasoningAgent


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class CoordinatorAgent:
    """
    Coordinator Agent (Agent 4 / Central Orchestrator).
    - Coordinates execution across Flood Segmentation (Agent 1), Building Damage (Agent 2),
      and Geospatial Reasoning (Agent 3).
    - Checks thresholds, validates data consistency, and detects critical edge cases.
    - Prompts Claude (Anthropic LLM) or fallback heuristic template to synthesize
      an operational Situational Report (SitRep) for emergency response command.
    """
    def __init__(self, use_llm: bool = True):
        self.config = load_config()
        self.use_llm = use_llm

        print("[Coordinator] Initializing specialist vision & reasoning agents...")
        self.flood_agent = FloodSegmentationAgent()
        self.building_agent = BuildingDamageAgent()
        self.geo_agent = GeospatialReasoningAgent()

        # LLM client setup
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or self.config.get("api", {}).get("anthropic_api_key")
        self.model_name = self.config.get("api", {}).get("model", "claude-sonnet-4-6")
        self.anthropic_client = None

        if self.use_llm and self.api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                print(f"[Coordinator] Anthropic LLM client active ({self.model_name}).")
            except Exception as e:
                print(f"[Coordinator] Warning: Could not initialize Anthropic client: {e}. Using fallback rule-based synthesis.")
                self.anthropic_client = None
        else:
            print("[Coordinator] Running with rule-based operational report synthesizer (no API key required).")

        self.reports_dir = PROJECT_ROOT / self.config.get("outputs", {}).get("reports_dir", "outputs/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_zone(
        self,
        zone_id: str,
        pre_image_path: str,
        post_image_path: str,
        metadata: dict = None
    ) -> dict:
        """
        Executes end-to-end multi-agent assessment for a specified geographic zone.
        """
        print(f"\n==================================================")
        print(f"[Coordinator] Starting Assessment for Zone: {zone_id}")
        print(f"==================================================")

        # 1. Run Flood Segmentation (Agent 1) on Post-disaster image
        print(f"[Coordinator] -> Invoking Flood Segmentation Agent...")
        flood_results = self.flood_agent.predict(post_image_path)

        # 2. Run Building Damage Assessment (Agent 2) on Pre/Post pair
        print(f"[Coordinator] -> Invoking Building Damage Assessment Agent...")
        damage_results = self.building_agent.predict(pre_image_path, post_image_path)

        # 3. Run Geospatial Reasoning (Agent 3) on extracted masks
        print(f"[Coordinator] -> Invoking Geospatial Reasoning Agent...")
        geo_results = self.geo_agent.evaluate_accessibility(
            flood_mask=flood_results["mask"],
            damage_mask=damage_results["mask"],
            zone_id=zone_id
        )

        # Compile structured data summary for synthesis
        zone_summary = {
            "zone_id": zone_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "flood_assessment": {
                "flood_severity": flood_results["flood_severity"],
                "class_pixel_percentages": {
                    k: round(v, 2) for k, v in flood_results["class_pixel_percentages"].items()
                }
            },
            "building_damage_assessment": {
                "dominant_damage_level": damage_results["dominant_damage_level"],
                "damage_severity_score": damage_results["damage_severity_score"],
                "damage_percentages": {
                    k: round(v, 2) for k, v in damage_results["damage_percentages"].items()
                },
                "building_damage_distribution": {
                    k: round(v, 2) for k, v in damage_results["building_damage_distribution"].items()
                },
                "total_building_pixels": damage_results["total_building_pixels"]
            },
            "geospatial_logistics": geo_results
        }

        # 4. Synthesize Situational Report (SitRep)
        print(f"[Coordinator] -> Synthesizing Situational Report (SitRep)...")
        sitrep_markdown = self._synthesize_report(zone_summary)
        zone_summary["sitrep_markdown"] = sitrep_markdown

        # 5. Save Report artifacts
        report_json_path = self.reports_dir / f"{zone_id}_assessment.json"
        report_md_path = self.reports_dir / f"{zone_id}_sitrep.md"

        # Save JSON (strip non-serializable fields if any)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(zone_summary, f, indent=2)

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(sitrep_markdown)

        print(f"[Coordinator] Assessment complete!")
        print(f"  -> JSON Report: {report_json_path}")
        print(f"  -> Markdown SitRep: {report_md_path}")

        return {
            "summary": zone_summary,
            "sitrep": sitrep_markdown,
            "flood_mask": flood_results["mask"],
            "damage_mask": damage_results["mask"],
            "report_paths": {
                "json": str(report_json_path),
                "markdown": str(report_md_path)
            }
        }

    def _synthesize_report(self, summary_data: dict) -> str:
        """Generates executive SitRep via LLM or structured template."""
        if self.anthropic_client:
            try:
                return self._synthesize_with_llm(summary_data)
            except Exception as e:
                print(f"[Coordinator] LLM API call failed ({e}). Falling back to template generator.")

        return self._synthesize_template(summary_data)

    def _synthesize_with_llm(self, summary_data: dict) -> str:
        prompt = f"""You are the Disaster Response AI Coordinator. Analyze the multi-agent aerial reconnaissance data for the affected zone and synthesize a concise, high-priority Situational Report (SitRep) for Emergency First Responders and Incident Commanders.

DATA FROM SPECIALIST AGENTS:
{json.dumps(summary_data, indent=2)}

Please write the Situational Report in clean GitHub Markdown with the following sections:
# 🚨 SITUATIONAL REPORT: Zone {summary_data['zone_id']}
## 1. Executive Summary & Threat Level
## 2. Flood & Inundation Analysis
## 3. Structural & Building Damage Impact
## 4. Road Accessibility & Ingress/Egress Status
## 5. Recommended Logistics & Transit Modes
## 6. Priority Action Items for First Responders

Keep the report professional, precise, data-backed, and immediately actionable for field commanders.
"""
        response = self.anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _synthesize_template(self, summary_data: dict) -> str:
        z_id = summary_data["zone_id"]
        flood = summary_data["flood_assessment"]
        dmg = summary_data["building_damage_assessment"]
        geo = summary_data["geospatial_logistics"]

        bldg_flooded = flood["class_pixel_percentages"].get("Building-Flooded", 0.0)
        road_flooded = flood["class_pixel_percentages"].get("Road-Flooded", 0.0)
        water_coverage = flood["class_pixel_percentages"].get("Water", 0.0)

        transit_list = "\n".join([f"- **{t}**" for t in geo.get("recommended_transit", [])])

        report = f"""# 🚨 SITUATIONAL REPORT: Zone {z_id}
*Generated by Disaster Multi-Agent AI System at {summary_data['timestamp']}*

---

## 1. Executive Summary & Threat Level
- **Overall Inundation Severity:** `{flood['flood_severity'].upper()}`
- **Building Damage Severity Index:** `{dmg['damage_severity_score']}/100` (Dominant: **{dmg['dominant_damage_level']}**)
- **Road Network Status:** `{geo['access_status']}` (Access Score: **{geo['accessibility_score']}/100**)

---

## 2. Flood & Inundation Analysis
- **Flooded Roadways:** `{road_flooded}%` of area
- **Flooded Buildings:** `{bldg_flooded}%` of area
- **Standing Water Coverage:** `{water_coverage}%`
- **Assessment:** {
    'Catastrophic flooding requiring immediate water rescue deployments.' if flood['flood_severity'] == 'severe'
    else 'Localized flooding observed. Secondary drainage routes required.' if flood['flood_severity'] == 'partial'
    else 'No significant standing flood water detected in this sector.'
}

---

## 3. Structural & Building Damage Impact
- **Total Building Footprint Analyzed:** `{dmg['total_building_pixels']} px`
- **Damage Class Breakdown:**
  - Destroyed: `{dmg['building_damage_distribution'].get('Destroyed', 0.0)}%`
  - Major Damage: `{dmg['building_damage_distribution'].get('Major-Damage', 0.0)}%`
  - Minor Damage: `{dmg['building_damage_distribution'].get('Minor-Damage', 0.0)}%`
  - Undamaged / Intact: `{dmg['building_damage_distribution'].get('No-Damage', 0.0)}%`
- **Structural Hazard:** `{geo['hazards']['structural_rubble_hazard']}` (Rubble road blockage risk: `{geo['hazards']['potential_rubble_blockage']}`)

---

## 4. Road Accessibility & Ingress/Egress Status
- **Road Network Flooding:** `{geo['road_metrics']['flooded_road_percentage']}%` flooded / `{geo['road_metrics']['clear_road_percentage']}%` clear
- **Access Rating:** **{geo['access_status']}**
- **Landing Zone (LZ) Feasibility:** `{'VIABLE' if geo['logistics']['landing_zone_viable'] else 'UNSAFE / OBSTRUCTED'}`
- **Staging Ground Availability:** `{'AVAILABLE' if geo['logistics']['staging_area_available'] else 'LIMITED'}`

---

## 5. Recommended Logistics & Transit Modes
{transit_list}

---

## 6. Priority Action Items for First Responders
1. {'Deploy watercraft and UAV search teams immediately.' if flood['flood_severity'] == 'severe' or geo['access_status'] == 'Blocked / Impassable' else 'Establish standard vehicle ingress corridors.'}
2. {'Cordon off collapsed structures and clear debris from primary transit routes.' if dmg['damage_severity_score'] > 40 else 'Perform routine door-to-door welfare checks.'}
3. {'Designate helicopter evacuation zone in identified open grass sectors.' if geo['logistics']['landing_zone_viable'] else 'Establish alternate amphibious extraction point.'}
"""
        return report


if __name__ == "__main__":
    agent = CoordinatorAgent(use_llm=False)
    print("CoordinatorAgent initialized successfully.")
