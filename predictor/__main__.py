import logging
from typing import NamedTuple, Optional

from model import LeNet

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

from MPC_identities import PATIENT, PREDICTOR
crypten.init()


@mpc.run_multiprocess(world_size=2)
def run_prediction(model_name: str = "LeNet",
                   model_file: str = "lenet_trained.pth"):

    rank = comm.get().get_rank()

    # create empty model
    if hasattr(models, model_name):
        dummy_model = getattr(models, model_name)(pretrained=False, num_classes=2)
    elif model_name == "LeNet":
        dummy_model = LeNet(num_classes=2)
    else:
        raise NotImplementedError(f"No model named {model_name} available")
    print(f"{rank} LOADED empty")


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

    # Todo: create (web)queue
    if rank == PATIENT:
        import time
        print(f"{rank} SLEEP")
        time.sleep(1)

    # Load image to PATIENT.. dummy_input for now
    data_enc = crypten.cryptensor(dummy_input)

    # classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_enc)

    # print output
    output = output_enc.get_plain_text()
    print(f"{rank} OUTPUT {output})")


@click.command()
@click.option("--port", default=8080)
@click.option("--log", default="DEBUG")
def start_server(port: int=8080):
    app = web.Application()
    web.run_app(app, port=port)


if __name__ == "__main__":
    from predictor.greeting import greeting
    print(greeting)
    #start_server()
    logging.basicConfig(level=logging.INFO)
    run_prediction()