import logging
from datetime import datetime

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
from predictor.web import index, start_runner_thread, ImageQueueDoctor, ImageQueuePatient

crypten.init()


def init_app(relay_predictor_host, img_size):
    app = web.Application()

    if comm.get().get_rank() == PREDICTOR:
        q = ImageQueueDoctor(img_size=img_size)
    else:
        q = ImageQueuePatient( img_size=img_size, relay_to=relay_predictor_host + "/image")

    app.add_routes([web.get("/", index),
                    web.post("/image", q.image)])
    return app, q


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
    image_queue = start_web_app(port0, img_size)
    input_queue = image_queue.queue

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

    # Load image to PATIENT.. dummy_input for testing
    data_enc = crypten.cryptensor(dummy_input)

    # classify the encrypted data
    private_model.eval()
    output_enc = private_model(data_enc)

    # print output
    output = output_enc.get_plain_text()
    print(f"{rank} TEST OUTPUT {output})")

    with torch.no_grad():
        while True:
            tensor_image_or_empty, input_time = input_queue.get()
            encrpyted_image = crypten.cryptensor(tensor_image_or_empty.unsqueeze(0), src=PATIENT)
            output_enc = private_model(encrpyted_image)
            output = output_enc.get_plain_text()
            probabilities = torch.softmax(output, dim=1)[0]

            prediction_is_cancer = probabilities[1].cpu().item()
            predictor_is_cancer = prediction_is_cancer > 0.5
            print(f"{rank} PRED {prediction_is_cancer:.2f}% cancer @ {input_time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"(image mean: {tensor_image_or_empty.mean().item()})")

            if rank == PREDICTOR:
                q_predictor: ImageQueueDoctor = image_queue

                yn = input(f'{rank} --> to real Doctor: is this prediction correct[y/N]?')
                if yn == "y":
                    doctor_is_cancer = predictor_is_cancer
                else:
                    doctor_is_cancer = not predictor_is_cancer
                decision_time = datetime.now()
                print(f"{rank} Thanks, you'were saying there {'IS' if doctor_is_cancer else 'is NO'} CANCER")

                csv_string = f"{decision_time.strftime('%Y-%m-%dT%H:%M:%S')},{int(doctor_is_cancer)}"
                print(f"{rank} appending: " + csv_string)


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
    print()
    run_service()

