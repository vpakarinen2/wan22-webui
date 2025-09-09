from app.config.settings import settings
from app.main import create_app


def main() -> None:
    app = create_app()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=settings.port, show_error=True)


if __name__ == "__main__":
    main()
