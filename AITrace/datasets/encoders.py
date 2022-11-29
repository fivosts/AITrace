import datetime
import typing
import math

from proto import dataset_pb2

from eupy.native import logger as l

visiting_labels = {
  "ANT"         : 0,
  "FISH"        : 1,
  "BUTTERFLY"   : 2,
  "GRASSHOPPER" : 3,
  "UNK"         : 4,
}

ids_to_labels = {
  0: "ANT",
  1: "FISH",
  2: "BUTTERFLY",
  3: "GRASSHOPPER",
  4: "UNK",
}

terminals = {
  'enters' : 0,
  'leaves' : 1,
}

def EncodeLocations(data: typing.List[typing.Tuple[str, int]]) -> typing.List[typing.Tuple[int, int]]:
  """
  Encode locations into unique integer ids.
  """
  f_idx = 2
  for idx, dp in enumerate(data):
    if dp:
      exh, _ = dp
      if exh not in terminals:
        terminals[exh] = f_idx
        f_idx += 1
      data[idx][0] = terminals[exh]
  return data

def EncodeTimeIntervals(data: typing.List[typing.Tuple[str, datetime.datetime, datetime.datetime]]) -> typing.List[typing.Tuple[str, int]]:
  """
  Convert start and end time to duration.
  """
  new_data = []
  for dp in data:
    try:
      exh, s, e = dp
    except ValueError as e:
      new_data.append([])
      continue

    start, end = datetime.datetime.strptime(s, "%H:%M:%S"), datetime.datetime.strptime(e, "%H:%M:%S")
    diff = int((end-start).total_seconds())
    new_data.append([exh, diff])
  return new_data