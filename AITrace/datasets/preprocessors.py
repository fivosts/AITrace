import datetime
import typing
import math

from proto import dataset_pb2

from eupy.native import logger as l

def IsolateHechtVisitors(text: str) -> typing.List[str]:
  """
  Within Hecht museum data, parse log file and
  split all visitor entries into separate text entities.
  """
  visitors = []
  start_keyword = "**************************    visitor: "
  line_iter = text.split('\n')
  l_idx = 0
  while l_idx < len(line_iter):
    cur_line = line_iter[l_idx]
    cur_vis  = []
    if start_keyword in cur_line:
      cur_vis.append(cur_line)
      l_idx += 1
      while l_idx < len(line_iter) and start_keyword not in line_iter[l_idx]:
        cur_vis.append(line_iter[l_idx])
        l_idx += 1
      visitors.append('\n'.join(cur_vis))
    else:
      l_idx += 1
  return visitors

def NormalizeHechtDateTime(text: typing.Union[typing.List[str], str]) -> typing.Union[typing.List[str], str]:
  """
  Date and time shouldn't play a role in visitors' behavior.
  In datapoints, visiting times are normalized to 00:00:00.
  """
  if isinstance(text, list):
    return [NormalizeHechtDateTime(v) for v in text]
  else:
    lines = text.split('\n')
    time_offset = None
    for idx, l in enumerate(lines):
      if l.startswith("start time ") and " end time: " in l:
        stripped_line = l.split(' ')
        st, et = datetime.datetime.strptime(stripped_line[2], "%H:%M:%S"), datetime.datetime.strptime(stripped_line[5], "%H:%M:%S")
        if not time_offset:
          time_offset = datetime.timedelta(hours = st.hour, minutes = st.minute, seconds = st.second)
        st, et = st - time_offset, et - time_offset
        stripped_line[2], stripped_line[5] = st.strftime("%H:%M:%S"), et.strftime("%H:%M:%S")
        lines[idx] = ' '.join(stripped_line)
    return '\n'.join(lines)

def SplitHechtDataFields(
  text: typing.Union[typing.List[typing.List[str]], str]
)    -> typing.Union[typing.List[typing.Dict[str, str]], typing.Dict[str, str]]:
  if isinstance(text, list):
    return [SplitHechtDataFields(v) for v in text]
  else:
    data = {
      'id'             : None,
      'locations'      : [],
      'presentations'  : [],
      'after_first_merge'    : [],
      'after_second_merge'   : [],
      '#presentations' : None,
      '#locations'     : None,
      '#unique_locations'     : None,
      '#presentations_w_locations' : None,
      'avg_time_per_location'       : None,
      'total_time'                 : None,
      'movement_type'              : None
    }
    l_idx = 0
    lines = text.split('\n')
    while l_idx < len(lines):
      cur = lines[l_idx]
      if "before merge position list:" in cur:
        l_idx += 1
        while l_idx < len(lines) and "start time" in lines[l_idx]:
          fields = lines[l_idx].split(' ')
          data['locations'].append("{},{},{}".format(fields[7], fields[2], fields[5]))
          l_idx += 1
      elif "before merge presentation list:" in cur:
        l_idx += 1
        while l_idx < len(lines) and "start time" in lines[l_idx]:
          fields = lines[l_idx].split(' ')
          data['presentations'].append("{},{},{}".format(fields[7], fields[2], fields[5]))
          l_idx += 1
      elif "**********     after first merge" in cur:
        l_idx += 1
        while l_idx < len(lines) and "start time" in lines[l_idx]:
          fields = lines[l_idx].split(' ')
          data['after_first_merge'].append("{},{},{}".format(fields[7], fields[2], fields[5]))
          l_idx += 1
      elif "**********     after second merge" in cur:
        l_idx += 1
        while l_idx < len(lines) and "start time" in lines[l_idx]:
          fields = lines[l_idx].split(' ')
          data['after_second_merge'].append("{},{},{}".format(fields[7], fields[2], fields[5]))
          l_idx += 1
      elif "General Statistics:" in cur:
        l_idx += 1
        data['id'] = int(lines[l_idx].split(': ')[-1])
        l_idx += 1
        data['#presentations'] = int(lines[l_idx].split(': ')[-1])
        l_idx += 1
        data['#locations'] = int(lines[l_idx].split(': ')[-1])
        data['#unique_locations'] = -1 # Unused for Hecht
        l_idx += 1
        data['#presentations_w_locations'] = int(lines[l_idx].split(': ')[-1])
        l_idx += 1
        if data['id'] in {45}:
          data['avg_time_per_location'] = int(lines[l_idx].split(': ')[-1].replace("No act", " No act").split(' ')[0])
        else:
          data['avg_time_per_location'] = int(lines[l_idx].split(': ')[-1])
        l_idx += 3
        if data['id'] in {11, 347, 355, 401}:
          l_idx += 2
        if data['id'] in {45}:
          l_idx += 4
        data['total_time'] = int(lines[l_idx].split(' ')[2])
        l_idx += 2
        data['movement_type'] = str(lines[l_idx].split(': ')[-1])
        l_idx += 1
      else:
        l_idx += 1
    return data

  time_offset = datetime.timedelta(hours = st.hour, minutes = st.minute, seconds = st.second)

def AddHechtMuseumArea(
  text: typing.Union[typing.List[typing.Dict[str, str]], typing.Dict[str, str]]
)    -> typing.Union[typing.List[typing.Dict[str, str]], typing.Dict[str, str]]:
  """
  When there are time-frame gaps, insert 'museum_area' token.
  """
  if isinstance(text, list):
    return [AddHechtMuseumArea(t) for t in text]
  else:
    changed_keys = {"locations", "presentations", "after_first_merge", "after_second_merge"}
    for k, v in text.items():
      if k in changed_keys:
        lidx, ridx = 0, 1
        while ridx < len(v):
          t1_str, t2_str = v[lidx].split(',')[-1], v[lidx].split(',')[1]
          t1, t2 = datetime.datetime.strptime(t1_str, "%H:%M:%S"), datetime.datetime.strptime(t2_str, "%H:%M:%S")
          if t2 - t1 > datetime.timedelta():
            v = v[:ridx] + ["<MuseumArea>,{},{}".format(t1.strftime("%H:%M:%S"), t2.strftime("%H:%M:%S"))] + v[ridx:]
            lidx += 1
            ridx += 1
          lidx += 1
          ridx += 1
    return text

def AddHechtEntersLeaves(
  text: typing.Union[typing.List[typing.Dict[str, str]], typing.Dict[str, str]]
)    -> typing.Union[typing.List[typing.Dict[str, str]], typing.Dict[str, str]]:
  """
  Add 'enters' and 'leaves' entry.
  """
  if isinstance(text, list):
    return [AddHechtEntersLeaves(t) for t in text]
  else:
    changed_keys = {"locations", "presentations", "after_first_merge", "after_second_merge"}
    for k, v in text.items():
      if k in changed_keys:
        text[k] = ['enters,00:00:00,00:00:00'] + text[k] + ['leaves,00:00:00,00:00:00']
    return text

def IsolateAITraceActions(text: str) -> typing.List[typing.Dict[str, str]]:
  """
  Takes raw csv text as an input and isolates csv actions to list of post-processed data.
  """
  lines = text.split('\n')
  lidx = 0
  while "Time,Media,file,path,Total,length,FPS,Subject,Behavior,Behavioral,category,Comment,Status" not in lines[lidx]:
    lidx += 1
  lines = lines[lidx+1:]
  out = []
  for line in lines:
    if line != '':
      l = line.split(',')
      out.append(
        {
          'id'        : l[-3],
          'action'    : l[-2],
          'status'    : l[-1],
          'timestamp' : l[0],
        }
      )
  return out

def ResolveAITraceActions(actions: typing.List[typing.Dict[str, str]]) -> typing.Dict[str, typing.List[typing.Tuple[str, float, float]]]:
  """
  Receives a set of raw actions for random visitor ids
  and returns the full time-series behavior of a visitor.
  """
  visitors = {}
  base_times = {}

  # Fix "enters"
  for action in actions:
    if action['id'] not in visitors:
      visitors[action['id']] = []
      base_times[action['id']] = [math.inf, 0.0]
    if action['action'] == "enters":
      if len(visitors[action['id']]) == 0 or visitors[action['id']][0][0] != "enters":
        visitors[action['id']].append([ 'enters', float(action['timestamp']), float(action['timestamp']) ])
    base_times[action['id']] = [min(base_times[action['id']][0], float(action['timestamp'])), max(base_times[action['id']][1], float(action['timestamp']))]

  # Fix exhibit visits
  for action in actions:
    if action['action'] != "enters" and action['action'] != "leaves":
      if action['status'] == "START":
        visitors[action['id']].append([action['action'], float(action['timestamp']), None])
      elif action['status'] == "STOP":
        visitors[action['id']][-1][-1] = float(action['timestamp'])
      else:
        raise ValueError(action['status'])
    base_times[action['id']] = [min(base_times[action['id']][0], float(action['timestamp'])), max(base_times[action['id']][1], float(action['timestamp']))]

  # Fix "leaves"
  for action in actions:
    if action['action'] == "leaves":
      if len(visitors[action['id']]) == 0:
       visitors[action['id']].append([ 'leaves', float(action['timestamp']), float(action['timestamp'])])
      elif visitors[action['id']][-1][0] != "leaves":
       visitors[action['id']].append([ 'leaves', float(action['timestamp']), float(action['timestamp'])])
      else:
        visitors[action['id']][-1] = [ 'leaves', float(action['timestamp']), float(action['timestamp'])]
    base_times[action['id']] = [min(base_times[action['id']][0], float(action['timestamp'])), max(base_times[action['id']][1], float(action['timestamp']))]

  # Sanitize times
  for visitor, actions in visitors.items():
    if visitors[visitor][0][0] != "enters":
      l.getLogger().warn("\'enters\' entry not found for visitor {}.".format(visitor))
      visitors[visitor] = [[ 'enters', base_times[visitor][0], base_times[visitor][0] ]] + visitors[visitor]
    if visitors[visitor][-1][0] != "leaves":
      l.getLogger().warn("\'leaves\' entry not found for visitor {}.".format(visitor))
      visitors[visitor].append([ 'leaves', base_times[visitor][1], base_times[visitor][1]])
    if visitors[visitor][0][1] > base_times[visitor][0]:
      l.getLogger().warn("\'enters\' time registered is after minimum time found for visitor {}".format(visitor))
      visitors[visitor][0][1] = base_times[visitor][0]
      visitors[visitor][0][-1] = base_times[visitor][0]
    if visitors[visitor][-1][-1] < base_times[visitor][1]:
      l.getLogger().warn("\'leaves\' time registered is before maximum time found for visitor {}".format(visitor))
      visitors[visitor][-1][-1] = base_times[visitor][1]
      visitors[visitor][-1][1] = base_times[visitor][1]
    for idx, action in enumerate(visitors[visitor]):
      if idx != 0 and idx != len(visitors[visitor]) - 1:
        if action[1] is None:
          visitors[visitor][idx][1] = visitors[visitor][idx-1][-1]
        if action[-1] is None:
          visitors[visitor][idx][-1] = visitors[visitor][idx+1][1]

    for idx, act in enumerate(visitors[visitor]):
      visitors[visitor][idx][1] = float(act[1]) - base_times[visitor][0]
      visitors[visitor][idx][2] = float(act[2]) - base_times[visitor][0]
  return visitors

def AddAITraceMuseumArea(visitors: typing.Dict[str, typing.List[typing.Tuple[str, float, float]]]) -> typing.List[typing.Dict]:
  """
  Add intermediate <museumArea> tokens when there is time between exhibits.
  """
  for vid, actions in visitors.items():
    lidx, ridx = 0, 1
    while ridx < len(actions):
      # print(actions[lidx])
      t1, t2 = actions[lidx][-1], actions[ridx][1]
      if t2 > t1:
        actions = actions[:ridx] + [["<MuseumArea>", t1, t2]] + actions[ridx:]
        ridx += 1
        lidx += 1
      ridx += 1
      lidx += 1
    visitors[vid] = actions
  return visitors

def ReformAITraceFormat(visitors: typing.Dict[str, typing.List[typing.Tuple[str, float, float]]]) -> typing.List[typing.Dict]:
  """
  Convert visitor entries to appropriate ContentFiles format.
  """
  datapoints = []
  data = {
    'id'             : None,
    'locations'      : [],
    'presentations'  : [],
    'after_first_merge'    : [],
    'after_second_merge'   : [],
    '#presentations'    : None,
    '#locations'        : None,
    '#unique_locations' : None,
    '#presentations_w_locations' : None,
    'avg_time_per_location'       : None,
    'total_time'                 : None,
    'movement_type'              : None
  }

  for vid, actions in visitors.items():
    datapoints.append(
      {
        'id': vid,
        'locations': ["{},{},{}".format(exh_id, (datetime.timedelta(seconds = int(start)) + datetime.datetime.strptime("0", "%S")).strftime("%H:%M:%S"), (datetime.timedelta(seconds = int(stop)) + datetime.datetime.strptime("0", "%S")).strftime("%H:%M:%S")) for exh_id, start, stop in actions],
        'presentations': [],
        'after_first_merge': [],
        'after_second_merge': [],
        '#presentations': 0,
        '#locations': len([x for x in actions if x[0] != "enters" and x[0] != "leaves" and x[0] != "<MuseumArea>"]),
        '#unique_locations': len(set([x[0] for x in actions if x[0] != "enters" and x[0] != "leaves" and x[0] != "<MuseumArea>"])),
        '#presentations_w_locations': 0,
        'avg_time_per_location': int(([x for x in actions if x[0] != "enters" and x[0] != "leaves" and x[0] != "<MuseumArea>"][-1][-1] - [x for x in actions if x[0] != "enters" and x[0] != "leaves" and x[0] != "<MuseumArea>"][0][1]) / len([x for x in actions if x[0] != "enters" and x[0] != "leaves" and x[0] != "<MuseumArea>"])),
        'total_time': int(actions[-1][-1] - actions[0][1]),
        'movement_type': "UNK",
      }
    )
  return datapoints

def getPreprocessors(config: typing.Any) -> typing.List[typing.Callable]:
  if isinstance(config, dataset_pb2.HechtDataset):
    return [
      IsolateHechtVisitors,
      NormalizeHechtDateTime,
      SplitHechtDataFields, # Positions, presentations, merge1, merge2, presentationfrequency, general stats.
      AddHechtMuseumArea,
      AddHechtEntersLeaves,
    ]
  elif isinstance(config, dataset_pb2.AITraceDataset):
    return [
      IsolateAITraceActions,
      ResolveAITraceActions,
      AddAITraceMuseumArea,
      ReformAITraceFormat,
    ]
  else:
    raise NotImplementedError(type(config))
  return
