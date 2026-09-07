import argparse
import sys
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.coordinator.agent import CoordinatorAgent


def main():
    parser = argparse.ArgumentParser(description="Disaster Multi-Agent AI — Zone Assessment Runner")
    parser.add_argument("--zone", type=str, default="zone_001", help="Zone identifier (e.g. zone_001)")
    parser.add_argument("--pre", type=str, required=True, help="Path to pre-disaster image")
    parser.add_argument("--post", type=str, required=True, help="Path to post-disaster image")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM synthesis and use template generator")
    args = parser.parse_args()

    pre_path = Path(args.pre)
    post_path = Path(args.post)

    if not pre_path.exists():
        print(f"Error: Pre-disaster image not found at {pre_path}")
        sys.exit(1)
    if not post_path.exists():
        print(f"Error: Post-disaster image not found at {post_path}")
        sys.exit(1)

    coordinator = CoordinatorAgent(use_llm=not args.no_llm)
    results = coordinator.analyze_zone(
        zone_id=args.zone,
        pre_image_path=str(pre_path),
        post_image_path=str(post_path),
    )

    print("\n" + "=" * 50)
    print("SITUATIONAL REPORT PREVIEW")
    print("=" * 50)
    print(results["sitrep"])


if __name__ == "__main__":
    main()
