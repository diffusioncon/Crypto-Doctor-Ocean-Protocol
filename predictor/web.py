import asyncio

from aiohttp import web, ClientRequest, ClientResponse
import crypten.communicator as comm

from MPC_identities import PREDICTOR, PATIENT
from predictor.greeting import greeting

async def index(request: ClientRequest) -> ClientResponse:
    rank = comm.get().get_rank()
    if rank == PREDICTOR:
        rank = "PREDICTOR"
    elif rank == PATIENT:
        rank = "PATIENT"

    content = greeting
    content += f"<p>{rank}</p>"
    content = "<pre>"+content+"</pre>"

    return web.Response(text=content, content_type='text/html')


def runner_async(runner, port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, 'localhost', port)
    loop.run_until_complete(site.start())
    loop.run_forever()


def start_runner_thread(runner, port):
    from threading import Thread
    thread = Thread(target=runner_async, args=(runner, port), daemon=True)
    thread.start()