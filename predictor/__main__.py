from aiohttp import web
import click


@click.command()
@click.option("--port", default=8080)
def start_server(port: int=8080):
    app = web.Application()
    web.run_app(app, port=port)


if __name__ == "__main__":
    from predictor.greeting import greeting
    print(greeting)
    start_server()