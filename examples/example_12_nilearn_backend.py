"""Generate a nilearn-based brain image for a coordinate
======================================================
"""

from coord2region.pipeline import run_pipeline


def main() -> None:
    results = run_pipeline(
        inputs=[[30, -22, 50]],
        input_type="coords",
        outputs=["images"],
        image_backend="nilearn",
    )
    print("Nilearn image saved to", results[0].images["nilearn"])


if __name__ == "__main__":
    main()
