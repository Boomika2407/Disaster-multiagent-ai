import os
import yaml
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class GeospatialReasoningAgent:
    """
    Geospatial Reasoning Agent (Agent 3).
    Analyzes flood segmentation masks and building damage masks to derive:
    - Road accessibility status (Clear, Partial Access, Blocked / Impassable)
    - Viable transportation modes (Standard vehicles, High-clearance trucks, Amphibious/Boats, Air-only)
    - Route impassability and danger zones
    - Staging / landing zone feasibility
    - Overall accessibility score
    """
    def __init__(self):
        self.config = load_config()
        self.road_thresholds = self.config.get("thresholds", {}).get(
            "road_access", {"blocked_percent": 50.0, "partial_percent": 10.0}
        )
        self.floodnet_classes = self.config.get("floodnet", {}).get("classes", [
            "Background", "Building-Flooded", "Building-Non-Flooded",
            "Road-Flooded", "Road-Non-Flooded", "Water", "Tree", "Vehicle", "Pool", "Grass"
        ])

    def evaluate_accessibility(
        self,
        flood_mask: np.ndarray,
        damage_mask: np.ndarray = None,
        zone_id: str = "Unknown"
    ) -> dict:
        """
        Evaluates road network and physical accessibility from masks.

        Args:
            flood_mask: np.ndarray [H, W] with FloodNet class IDs (0..9).
                        Key IDs: 3 = Road-Flooded, 4 = Road-Non-Flooded, 5 = Water
            damage_mask: Optional np.ndarray [H, W] with xBD damage class IDs (0..4).
            zone_id: Identifier for the geographic zone/tile.

        Returns:
            Structured dictionary with road condition, access ratings, and logistics guidance.
        """
        if flood_mask is None:
            raise ValueError("flood_mask cannot be None")

        h, w = flood_mask.shape[:2]
        total_pixels = h * w

        # Extract flood classes
        # 3: Road-Flooded, 4: Road-Non-Flooded, 5: Water
        road_flooded_px = int(np.sum(flood_mask == 3))
        road_non_flooded_px = int(np.sum(flood_mask == 4))
        water_px = int(np.sum(flood_mask == 5))
        total_road_px = road_flooded_px + road_non_flooded_px

        road_coverage_pct = (total_road_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        if total_road_px > 0:
            flooded_road_pct = (road_flooded_px / total_road_px) * 100.0
            clear_road_pct = (road_non_flooded_px / total_road_px) * 100.0
        else:
            flooded_road_pct = 0.0
            clear_road_pct = 0.0

        # Determine Road Access Status
        blocked_thresh = self.road_thresholds.get("blocked_percent", 50.0)
        partial_thresh = self.road_thresholds.get("partial_percent", 10.0)

        if total_road_px == 0:
            access_status = "No Visible Roads"
            severity = "Unknown"
        elif flooded_road_pct >= blocked_thresh:
            access_status = "Blocked / Impassable"
            severity = "Critical"
        elif flooded_road_pct >= partial_thresh:
            access_status = "Partial Access"
            severity = "Moderate"
        else:
            access_status = "Clear / Accessible"
            severity = "Low"

        # Determine Viable Transportation Modes
        transport_modes = []
        if access_status == "Clear / Accessible":
            transport_modes.extend(["Standard Wheeled Vehicles", "Emergency Ambulances", "Heavy Utility Trucks"])
        elif access_status == "Partial Access":
            transport_modes.extend([
                "High-Clearance 4x4 / All-Terrain Vehicles",
                "Medium Rescue Trucks",
                "Shallow Water Boats (Localized)"
            ])
        elif access_status == "Blocked / Impassable":
            transport_modes.extend([
                "Amphibious Rescue Vehicles (AAV / ARGO)",
                "Flat-Bottom Rescue Boats",
                "Aerial Insertion / Drone Logistics (UAV)"
            ])
        else:
            transport_modes.append("Aerial Survey / Helicopters Only")

        # Debris & structural obstruction risk from damage mask if provided
        structural_hazard_level = "Low"
        rubble_road_blockage = False
        if damage_mask is not None:
            # Classes: 3 = Major Damage, 4 = Destroyed
            destroyed_px = int(np.sum((damage_mask == 3) | (damage_mask == 4)))
            if destroyed_px > (0.05 * total_pixels):
                structural_hazard_level = "High"
                rubble_road_blockage = True
            elif destroyed_px > (0.01 * total_pixels):
                structural_hazard_level = "Moderate"

        # Assess potential Landing / Staging Zones (open grass/dry ground)
        # Class 9 = Grass in FloodNet
        grass_px = int(np.sum(flood_mask == 9))
        grass_pct = (grass_px / total_pixels) * 100.0
        landing_zone_viable = grass_pct >= 5.0 and (flooded_road_pct < 80.0)

        # Accessibility Index (0 = totally impassable, 100 = completely clear)
        if total_road_px > 0:
            accessibility_score = max(0.0, min(100.0, 100.0 - flooded_road_pct))
        else:
            accessibility_score = 50.0  # neutral if no road detected

        return {
            "zone_id": zone_id,
            "access_status": access_status,
            "accessibility_score": round(accessibility_score, 1),
            "severity_level": severity,
            "road_metrics": {
                "total_road_pixels": total_road_px,
                "flooded_road_percentage": round(flooded_road_pct, 2),
                "clear_road_percentage": round(clear_road_pct, 2),
                "road_area_coverage_pct": round(road_coverage_pct, 2),
            },
            "recommended_transit": transport_modes,
            "hazards": {
                "standing_water_flood_risk": "High" if flooded_road_pct > 25.0 else "Low",
                "structural_rubble_hazard": structural_hazard_level,
                "potential_rubble_blockage": rubble_road_blockage,
            },
            "logistics": {
                "landing_zone_viable": landing_zone_viable,
                "staging_area_available": grass_pct >= 10.0,
            }
        }

    def generate_accessibility_map(
        self,
        flood_mask: np.ndarray,
        base_image: np.ndarray = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Generates a highlighted road status overlay map:
        - Red: Flooded / Blocked road segments
        - Green: Clear / Navigable road segments
        - Blue: Standing water / Water bodies
        """
        h, w = flood_mask.shape[:2]
        if base_image is None:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            overlay = cv2.resize(base_image, (w, h)).copy()

        road_color_map = np.zeros((h, w, 3), dtype=np.uint8)
        # Class 3: Road Flooded -> BGR (0, 0, 255) Red
        road_color_map[flood_mask == 3] = (0, 0, 255)
        # Class 4: Road Non-Flooded -> BGR (0, 255, 0) Green
        road_color_map[flood_mask == 4] = (0, 255, 0)
        # Class 5: Water -> BGR (255, 128, 0) Cyan/Blue
        road_color_map[flood_mask == 5] = (255, 128, 0)

        active_roads = (flood_mask == 3) | (flood_mask == 4) | (flood_mask == 5)
        for c in range(3):
            overlay[:, :, c] = np.where(
                active_roads,
                (1 - alpha) * overlay[:, :, c] + alpha * road_color_map[:, :, c],
                overlay[:, :, c]
            )

        return overlay.astype(np.uint8)


if __name__ == "__main__":
    agent = GeospatialReasoningAgent()
    print("GeospatialReasoningAgent initialized successfully.")

    # Smoke test with synthetic flood mask
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    # 20% clear roads, 30% flooded roads, 10% grass
    dummy_mask[100:150, :] = 4  # Road Non-Flooded
    dummy_mask[200:275, :] = 3  # Road Flooded
    dummy_mask[300:350, :] = 9  # Grass

    result = agent.evaluate_accessibility(dummy_mask, zone_id="Test-Zone-101")
    print(f"Status: {result['access_status']}")
    print(f"Accessibility Score: {result['accessibility_score']}")
    print(f"Flooded Road %: {result['road_metrics']['flooded_road_percentage']}%")
    print(f"Recommended Transit: {result['recommended_transit']}")
    print(f"Landing Zone Viable: {result['logistics']['landing_zone_viable']}")
