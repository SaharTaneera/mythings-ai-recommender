"""MyThings AI Recommender package."""


def main() -> None:
    """Run the FastAPI service through the installed command-line entry point."""
    import uvicorn

    uvicorn.run(
        "mythings_ai_recommender.main:app",
        host="127.0.0.1",
        port=8000,
    )
