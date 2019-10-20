import logging

try:
    import crypten
    import crypten.mpc as mpc
    import crypten.communicator as comm
except ImportError:
    print("Could not import CrypTen, visit https://github.com/facebookresearch/CrypTen for install instructions")
    exit(1)

import torch
import torchvision.models as models
from aiohttp import web
import click

from model import LeNet
from MPC_identities import PATIENT, PREDICTOR
from predictor.web import index, start_runner_thread, ImageQueue

crypten.init()



def init_app(relay_predictor_host, img_size):
    app = web.Application()

    if comm.get().get_rank() == PREDICTOR:
        q = ImageQueue(None, img_size=img_size)
    else:
        q = ImageQueue(relay_to=relay_predictor_host + "/image", img_size=img_size)

    app.add_routes([web.get("/", index),
                    web.post("/image", q.image)])
    return app, q.queue


def start_web_app(port0, img_size):
    app, q = init_app(f"http://localhost:{port0 + 1}", img_size)
    runner = web.AppRunner(app)

    # Todo: get other app
    if comm.get().get_rank() == PATIENT:
        start_runner_thread(runner, port0)
    else:
        start_runner_thread(runner, port0+1) #PREDICTOR

    return q


@mpc.run_multiprocess(world_size=2)
def run_prediction(port0: int = 8080,
                   model_name: str = "LeNet",
                   model_file: str = "lenet_trained.pth"):

    rank = comm.get().get_rank()

    # create empty model
    if hasattr(models, model_name):
        dummy_model = getattr(models, model_name)(pretrained=False, num_classes=2)
        img_size = 224
    elif model_name == "LeNet":
        dummy_model = LeNet(num_classes=2)
        img_size = 32
    else:
        raise NotImplementedError(f"No model named {model_name} available")
    print(f"{rank} LOADED empty")

    # start web interface
    queue = start_web_app(port0, img_size)

    # Load pre-trained model to PREDICTOR
    # For demo purposes, we don't pass model_name to PATIENT, although it would
    # be ignored in crypten.load
    if rank == PREDICTOR:
        model = crypten.load(model_file, dummy_model=dummy_model, src=PREDICTOR)
        print(f"{rank} LOADED model")
    else:
        model = crypten.load(None, dummy_model=dummy_model, src=PREDICTOR)
        print(f"{rank} BROADCAST model")

    # Encrypt model from PREDICTOR or dummy
    dummy_input = torch.zeros((1, 3, 32, 32))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=PREDICTOR)
    print(f"{rank} ENCRYPTED {private_model.encrypted}")

    # Load image to PATIENT.. dummy_input for now
    data_enc = crypten.cryptensor(dummy_input)

    # classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_enc)

    # print output
    output = output_enc.get_plain_text()
    print(f"{rank} OUTPUT {output})")

    while True:
        import time
        t = queue.get()
        print(f"{rank} INPUT {t.shape}, mean: {t.mean().item()}")
        time.sleep(1)


@click.command()
@click.option("--port", default=8080)
@click.option("--log", default="DEBUG")
def run_service(port: int=8080, log: str="DEBUG"):
    logging.basicConfig(level=getattr(logging, log.upper()))
    run_prediction(port)

if __name__ == "__main__":
    from predictor.greeting import greeting
    print(greeting)
    print(f"{PREDICTOR}=PREDICTOR")
    print(f"{PATIENT}=PATIENT")
    run_service()

