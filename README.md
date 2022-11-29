# ML System

<div align="center">
<a href="https://ibb.co/74mgy6T"> <kbd> <img src="https://i.ibb.co/tXrZCWN/architecture-drawio-1.png" alt="architecture-drawio-1" border="0"> </kbd> </a>
</div>

## Compile the app

On Linux type;

```
./configure
```

When asked about python version, type which python you want to run with your app. Available choices are

- python3.5
- python3.6
- python3.7
- python3.8

Default option is python3.8. If this sounds good, just hit enter.

After successfully configuring, a binary named 'aitrace' should appear in your current directory. Run it:

```
./aitrace <args>
```

To see a list of all available flags type:

```
./aitrace --help
```
Or:

```
./aitrace --helpfull
```

## Run Smoke Tests

Run a simple smoke test by specifying a config file and a workspace directory:
```
./aitrace --config model_zoo/smoke_test.pbtxt --workspace_dir workspace
```
This will train from scratch and then sample the model described in `model_zoo/smoke_test.pbtxt`

Or, to test if the app is deployed correctly use the docker image found in `./docker`. Inside this folder, configure docker dependencies with `./configure`. Then run `./docker_build.sh` to deploy the app on a fresh Ubuntu container.

## Running from Checkpoint

There is a working workspace that contains an officially trained and well-working ML model. This is the `final_workspace` in the root directory. To use it, simply use the above command setting `--workspace_dir final_workspace`.

## How to Interact with the Server

This app can perform neural network sampling with an Input/Output server queueing system:

- 2 thread-safe queues are implemented: `in_queue` and `out_queue`.
- `in_queue` stores all pending to-predict requests and `out_queue` contains all published predictions ready to be collected.
- A daemon thread listens for `POST` and `GET` requests.
- The main thread pops requests from `in_queue` and sends them to the neural network. Finished predictions are published in `out_queue`.
- The server supports three operations:
  1) `PUT` request at `http://<HOST>:<PORT>/json_file`. In this case, a JSON file with to-be-predicted inputs has to be included in the request.
  2) `GET` request at `http://<HOST>:<PORT>/get_predictions`. A serialized JSON file containing all predictions <ins>containing unchanged all the fields originally found in this input entry.</ins>
  3) `GET` request at `http://<HOST>:<PORT>/get_backlog`. All published JSON entries that are collected via 2), are stored in a safe backlog. In case the client fails, every JSON entry predicted in the session can be retrieved again.
- The sampler supports `full` or `step` sampling. In `full`, the model tries to predict all the next steps and attendance times the visitor will go through, and return a sequence of steps and the final predicted label. In `step`, only the next step is predicted and the current visiting label at that given state.
- The server will return the field `predicted_visit` for `full` sampling and `next_id`, `next_attendance_time` fields for `step`. In both cases `visiting_label` will be returned as well.

The JSON file provided to AITrace app must comply to the following rules in order to be parsed successfully:
1) It must be a list of dictionaries. Each dictionary is one visitor entry.
2) Each visitor entry must have a `input_feed` entry.
3) The `input_feed` entry represents the current state of the visitor in the museum as a list of pairs of integers. It must, therefore, be a list of lists.
4) Both pair members must be integers. The first indicates the exhibit id and the second the attendance time on this id.
5) `input_feed` must always start with the pair [0, 0]. This is the meta token the neural network understands as `START`.
6) Exhibit ID `1` is reserved for `EXIT`. Count exhibit IDs and museum traveling space starting from 2 onwards.

Other than that, each dictionary entry of the JSON file can contain any number, type and variety of fields. These will remain unchanged and will be included in the returned dictionary with the predictions. That happens for two reasons: A) To ensure consistency among the input feeds and the predictions and B) To save effort from the client in re-merging provided inputs with collected outputs.

`POST` Example:
```
[
  {
    "my_field_1" : 5,
    "my_field_2" : "test",
    "input_feed" : [[0, 0], [2, 50]]
  },
  {
    "my_field_1" : 8,
    "my_field_2" : "nick",
    "input_feed" : [[0, 0]]
  }
]
```
`POST` request example with `curl`:
```
curl -X POST http://<HOST>:<PORT>/json_file \
         --header "Content-Type: application/json" \
         -d @/home/AITrace_prvt/ML/AITrace/backend/test.json
```

will yield with `GET` when sampler is set to `prediction_type=step`:
```
[
  {
    "my_field_1" : 5,
    "my_field_2" : "test",
    "input_feed" : [[0, 0], [2, 50]],
    "next_id"    : 4,
    "next_attendance_time: 17,
    `visiting_label`     : "GRASSHOPPER"
  },
  {
    "my_field_1" : 8,
    "my_field_2" : "nick",
    "input_feed" : [[0, 0]],
    "next_id"    : 3,
    "next_attendance_time: 20,
    `visiting_label`     : "ANT"
  }
]
```
`GET` request example with `curl`:
```
curl -X GET http://<HOST>:<PORT>/get_predictions
```

Backlog `GET` request example:
```
curl -X GET http://<HOST>:<PORT>/get_backlog
```

## Deploy with Docker

This app comes in a pre-compiled docker container that can be downloaded and used as-is.
All you have to do is pull `fivosts/aitrace:latest` and run.

On Linux:

1) [Optional]: Specify the public IP of your container. This is useful if you need to know before-hand the IP address of it.

```
sudo docker network create \
    --ipv6 \
    --driver='bridge' \
    --subnet=82.103.188.0/29 \
    --gateway=82.103.188.1 \
    --subnet=2a00:9080:9:69::/64 \
    --gateway=2a00:9080:9:69::1 \
    aitrace_network
```
You can use any subnet and gateway you want.

2) Run the container

```
sudo docker run --network aitrace_network fivosts/aitrace:latest
```

This will pull the pre-compiled container and execute it.

<span style="color:red">**Congrats!**</span> You have the AITrace neural network up and running. Send requests at port 8080 (or specify otherwise at `model_zoo/smoke_test.pbtxt`) and communicate with the network as described in the previous subsection.
