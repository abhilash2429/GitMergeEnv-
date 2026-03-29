"""Compatibility shim exposing the FastAPI app at the repository root."""

from server.app import app, main


if __name__ == "__main__":
    main()
