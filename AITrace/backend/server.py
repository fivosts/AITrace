import portpicker
import subprocess
import json
import queue
import socketserver
import socket
import multiprocessing
import flask
import copy
import waitress

from eupy.native import logger as l

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.in_queue  = None
    self.out_queue = None
    self.backlog   = None
    self.sample_method  = None
    self.default_method = None
    self.temperature    = None
    self.default_temp   = None
    return

  def set_queues(self, in_queue, out_queue, sample_method, temperature):
    self.in_queue  = in_queue
    self.out_queue = out_queue
    self.sample_method  = sample_method
    self.default_method = copy.deepcopy(sample_method.value)
    self.temperature    = temperature
    self.default_temp   = copy.deepcopy(temperature.value)
    self.backlog   = []
    return

handler = FlaskHandler()

@app.route('/json_file', methods=['POST'])
def input_json_file():
  """
  Collect a json file, serialize and place into queue for NN prediction.

  Example command:
    curl -X POST http://localhost:PORT/json_file \
         --header "Content-Type: application/json" \
         -d @/home/fivosts/AITrace_prvt/ML/AITrace/backend/test.json
  """
  json_input = flask.request.json
  if not isinstance(json_input, list):
    return "ERROR: JSON Input has to be a list of dictionaries. One for each entry.\n", 400
  for idx, inp in enumerate(json_input):
    if 'input_feed' not in inp:
      return "ERROR: 'input_feed' entry not found in entry index {}\n".format(idx), 400
    for ent_idx, ent in enumerate(inp['input_feed']):
      if len(ent) != 2:
        return "ERROR: Input feed of entry {} has {} fields at idx {}. It should have 2 [exhibit_id, attendance_time]\n".format(
          idx, len(ent), ent_idx
        ), 400
      if not isinstance(ent[0], int) or not isinstance(ent[1], int):
        return "ERROR: Input feed idx {} of entry {} does not contain integers.\n".format(
          idx, ent_idx
        ), 400
  for entry in json_input:
    handler.in_queue.put(bytes(json.dumps(entry), encoding = "utf-8"))
  return 'OK\n', 200

@app.route('/get_predictions', methods = ['GET'])
def get_out_queue():
  """
  Publish all the predicted results of the out_queue.
  Before flushing the out_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/get_predictions
  """
  ret = []
  while not handler.out_queue.empty():
    cur = handler.out_queue.get()
    ret.append(json.loads(cur))
  handler.backlog += ret
  return bytes(json.dumps(ret), encoding="utf-8"), 200

@app.route('/get_backlog', methods = ['GET'])
def get_backlog():
  """
  In case a client side error has occured, proactively I have stored
  the whole backlog in memory. To retrieve it, call this method.

  Example command:
    curl -X GET http://localhost:PORT/get_backlog
  """
  return bytes(json.dumps(handler.backlog), encoding = "utf-8"), 200

@app.route('/submit', methods=['POST'])
def submit():
  """
  Submit paragraph prompt.
  """
  action = ""
  for x in flask.request.form:
    if x == "Single step" or x == "Full prediction":
      action = x
      break
  try:
    handler.temperature.value = float(flask.request.form['temperature'])
    if handler.temperature.value <= 0.0:
      return "Temperature must be greater than 0 to avoid division by zero, among other problems.", 311
  except Exception as e:
    return "Type error in temperature: {}, {}".format(flask.request.form['temperature'], repr(e)), 312
  try:
    if action == "Single step":
      handler.sample_method.value = True
      user_input = flask.request.form['text']
      user_input = [x.split() for x in user_input.split('\r')]
      if len(user_input) == 0:
        handler.sample_method.value = handler.default_method
        handler.temperature.value = handler.default_temp
        return "Error: User input is empty.", 301
      for idx, l in enumerate(user_input):
        if len(l) != 2:
          handler.sample_method.value = handler.default_method
          handler.temperature.value = handler.default_temp
          return "Error in line {}: Entry must have two fields visited_id, attendance_time. Found {}".format(idx, len(l)), 302
        v, t = l[0], l[1]
        try:
          v = int(v)
          t = int(t)
          if int(v) == 1:
            handler.sample_method.value = handler.default_method
            handler.temperature.value = handler.default_temp
            return "Error in line: {}: ID 1 represents 'EXIT'. Do not use it in your input feed.".format(idx), 303
        except Exception:
          handler.sample_method.value = handler.default_method
          handler.temperature.value = handler.default_temp
          return "Error in line {}: Fields must be integers.".format(idx), 304
      user_input = [[int(x[0]), (int(x[1]))] for x in user_input]
      handler.in_queue.put(bytes(json.dumps(
        {
          'input_feed': user_input
        }
      ), encoding = 'utf-8'))
      pred = json.loads(handler.out_queue.get())
      handler.sample_method.value = handler.default_method
      handler.temperature.value = handler.default_temp
      return flask.render_template("step_result.html", data = pred)
    elif action == "Full prediction":
      handler.sample_method.value = False
      user_input = flask.request.form['text']
      user_input = [x.split() for x in user_input.split('\r')]
      if len(user_input) == 0:
        handler.sample_method.value = handler.default_method
        handler.temperature.value = handler.default_temp
        return "Error: User input is empty.", 301
      for idx, l in enumerate(user_input):
        if len(l) != 2:
          handler.sample_method.value = handler.default_method
          handler.temperature.value = handler.default_temp
          return "Error in line {}: Entry must have two fields visited_id, attendance_time. Found {}".format(idx, len(l)), 302
        v, t = l[0], l[1]
        try:
          v = int(v)
          t = int(t)
          if int(v) == 1:
            handler.sample_method.value = handler.default_method
            handler.temperature.value = handler.default_temp
            return "Error in line: {}: ID 1 represents 'EXIT'. Do not use it in your input feed.".format(idx), 303
        except Exception:
          handler.sample_method.value = handler.default_method
          handler.temperature.value = handler.default_temp
          return "Error in line {}: Fields must be integers.".format(idx), 304
      user_input = [[int(x[0]), (int(x[1]))] for x in user_input]
      handler.in_queue.put(bytes(json.dumps(
        {
          'input_feed': user_input
        }
      ), encoding = 'utf-8'))
      pred = json.loads(handler.out_queue.get())
      handler.sample_method.value = handler.default_method
      handler.temperature.value = handler.default_temp
      return flask.render_template("full_result.html", data = pred)
  except Exception as e:
    handler.sample_method.value = handler.default_method
    handler.temperature.value = handler.default_temp
    return "{}\nThe above exception has occured. Copy paste to Foivos.".format(repr(e)), 404

@app.route('/')
def index():
  """
  Main page of graphical environment.
  """
  return flask.render_template("index.html")

def serve(in_queue: queue.Queue,
          out_queue: queue.Queue,
          sample_method: multiprocessing.Value,
          temperature: multiprocessing.Value,
          port: int = None
          ):
  try:
    if port is None:
      port = portpicker.pick_unused_port()
    handler.set_queues(in_queue, out_queue, sample_method, temperature)
    hostname = subprocess.check_output(
      ["hostname", "-i"],
      stderr = subprocess.STDOUT,
    ).decode("utf-8").replace("\n", "").split(' ')
    if len(hostname) == 2:
      ips = "ipv4: {}, ipv6: {}".format(hostname[1], hostname[0])
    else:
      ips = "ipv4: {}".format(hostname[0])
    l.getLogger().warn("Server Public IP: {}".format(ips))
    waitress.serve(app, host = '0.0.0.0', port = port)
    l.getLogger().info("Access at http://{}:{}".format(hostname[1], port))
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return
