import asyncio
import io
from queue import Queue

from PIL import Image
from aiohttp import web, ClientRequest, ClientResponse, ClientSession
import torch
import torchvision.transforms as transforms
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


def get_image_transform(size: int):

    transform = transforms.Compose(
        [
            transforms.Resize(int(size*1.1)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform

class ImageQueue:
    def __init__(self, img_size=32):
        self.queue = Queue(maxsize=2)
        self.img_size = img_size


class ImageQueuePatient(ImageQueue):

    def __init__(self, img_size: int,  relay_to: str):
        self.relay_to = relay_to
        super().__init__(img_size)

    async def image(self, request: ClientRequest) -> ClientResponse:

        data = await request.post()
        encoded_img = data["file"].file.read()
        stream = io.BytesIO(encoded_img)
        img = Image.open(stream)
        trans = get_image_transform(self.img_size)
        t = trans(img)
        self.queue.put(t)

        async def send():
            async with ClientSession() as session:
                async with session.post(url=self.relay_to) as resp:
                    pass

        asyncio.ensure_future(send())
        return web.json_response(dict(OK="queued img", shape=list(t.shape)))

class ImageQueueDoctor(ImageQueue):

    async def image(self, request: ClientRequest) -> ClientResponse:
        t = torch.empty(3, self.img_size, self.img_size)
        self.queue.put(t)
        return web.json_response(dict(OK="queued empty img"))



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