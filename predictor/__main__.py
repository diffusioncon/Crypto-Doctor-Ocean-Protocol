import sys
from typing import Optional

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

from model import LeNet,BigLeNet
from MPC_identities import PATIENT, PREDICTOR
from predictor.web import index, start_runner_thread, ImageQueueDoctor, ImageQueuePatient

crypten.init()

def print_stderr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def init_app(relay_predictor_host, img_size):
    app = web.Application()

    if comm.get().get_rank() == PREDICTOR:
        q = ImageQueueDoctor(img_size=img_size)
    else:
        q = ImageQueuePatient( img_size=img_size, relay_to=relay_predictor_host + "/image")

    app.add_routes([web.get("/", index),
                    web.post("/image", q.image)])

    if comm.get().get_rank() == PREDICTOR:
        app.add_routes([web.get("/decision", q.decision),
                        web.post("/make_decision", q.make_decision)])

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
                   model_file: Optional[str] = None):

    rank = comm.get().get_rank()

    # create empty model
    if hasattr(models, model_name):
        dummy_model = getattr(models, model_name)(pretrained=False, num_classes=2)
        img_size = 224
    elif model_name == "LeNet":
        dummy_model = LeNet(num_classes=2)
        img_size = 32
    elif model_name == "BigLeNet":
        dummy_model = BigLeNet(num_classes=2)
        img_size = 64
    else:
        raise NotImplementedError(f"No model named {model_name} available")
    print_stderr(f"{rank} LOADED empty")

    if model_file is None:
        model_file = f"models/{model_name}.pth"

    # start web interface
    image_queue = start_web_app(port0, img_size)
    input_queue = image_queue.queue

    # Load pre-trained model to PREDICTOR
    # For demo purposes, we don't pass model_name to PATIENT, although it would
    # be ignored in crypten.load
    if rank == PREDICTOR:
        model = crypten.load(model_file, dummy_model=dummy_model, src=PREDICTOR)
        print_stderr(f"{rank} LOADED model")
    else:
        model = crypten.load(None, dummy_model=dummy_model, src=PREDICTOR)
        print_stderr(f"{rank} BROADCAST model")

    # Encrypt model from PREDICTOR or dummy
    dummy_input = torch.zeros((1, 3, img_size, img_size))
    private_model = crypten.nn.from_pytorch(model, dummy_input)
    private_model.encrypt(src=PREDICTOR)
    print_stderr(f"{rank} ENCRYPTED {private_model.encrypted}")

    # Load image to PATIENT.. dummy_input for testing
    data_enc = crypten.cryptensor(dummy_input)

    # classify the encrypted data
    with torch.no_grad():
        private_model.eval()
        output_enc = private_model(data_enc)

    # print_stderr output
    output = output_enc.get_plain_text()
    print_stderr(f"{rank} TEST OUTPUT {output})")

    if rank == PREDICTOR:
        print_stderr(f"providing .csv header")
        print("input_time,decision_time,doctor_is_cancer,predictor_is_cancer")
    with torch.no_grad():
        while True:
            tensor_image_or_empty, input_time = input_queue.get()
            encrpyted_image = crypten.cryptensor(tensor_image_or_empty.unsqueeze(0), src=PATIENT)
            output_enc = private_model(encrpyted_image)
            output = output_enc.get_plain_text()
            probabilities = torch.softmax(output, dim=1)[0]

            prediction_is_cancer = probabilities[1].cpu().item()
            print_stderr(f"{rank} PRED {prediction_is_cancer:.2f}% cancer @ {input_time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"(image mean: {tensor_image_or_empty.mean().item()})")

            if rank == PREDICTOR:
                q_predictor: ImageQueueDoctor = image_queue
                q_predictor.current_pred_cancer = prediction_is_cancer
                answer_queue = q_predictor.answer_queue

                print_stderr(f"{rank} Waiting for final decision on http://localhost:{port0 + 1}/decision")
                doctor_is_cancer, decision_time = answer_queue.get()
                print_stderr(f"{rank} Thanks, you were saying there {'IS' if doctor_is_cancer else 'is NO'} CANCER")

                # input_time,decision_time,doctor_is_cancer,predictor_is_cancer
                csv_string = f"{input_time.strftime('%Y-%m-%dT%H:%M:%S')},{decision_time.strftime('%Y-%m-%dT%H:%M:%S')},{int(doctor_is_cancer)},{float(prediction_is_cancer)}"
                print_stderr(f"{rank} appending: " + csv_string)
                print(csv_string)


@click.command()
@click.option("--port", default=8080, help="patient port of localhost service (doctor service runs on +1)")
@click.option("--model-name", default="LeNet", help="Model architecture to use (either exists in torchvision, like resnet18, or custom LeNet or BigLeNet")
@click.option("--model-file", default=None, type=str, help="Weights of a pretrained model that only PREDICTOR has access to, if None, defaults to models/{model_name}.pth")
def run_service(port: int=8080,
                model_name: str="LeNet",
                model_file: Optional[str]=None):
    print_stderr(f"As PATIENT, post your image file to: http://localhost:{port}/image")
    print_stderr('e.g.: `curl -X POST -F "file=@test/Y22.jpg" localhost:8080/image`')
    print_stderr(f"As DOCTOR, make your final decion at: http://localhost:{port+1}/decision")
    print_stderr()
    print_stderr(f"{PREDICTOR}=PREDICTOR")
    print_stderr(f"{PATIENT}=PATIENT")
    print_stderr()
    run_prediction(port, model_name, model_file)

if __name__ == "__main__":
    from predictor.greeting import greeting
    print_stderr(greeting)
    run_service()

