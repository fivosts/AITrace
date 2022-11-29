"""This file defines a database for pre-preprocessed content files."""
import contextlib
import datetime
import hashlib
import os
import pathlib
import subprocess
import tempfile
import time
import typing
import humanize
import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from proto import dataset_pb2
from proto import internal_pb2
from util import crypto
from backend import sqlutil
from datasets import preprocessors

from absl import flags
from eupy.native import logger as l
from eupy.hermes import client

FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = "meta"

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)


class ContentFile(Base):
  __tablename__ = "preprocessed_contentfiles"

  id: int = sql.Column(sql.Integer, primary_key=True)
  # Checksum of the preprocessed file.
  sha256             : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Relative path of the input file within the content files.
  input_relpath: str = sql.Column(sql.String(3072), nullable=False, unique=False)
  # Visiting ID as per the logfiles
  visitor_id         : int = sql.Column(sql.Integer, nullable = False)
  # Exhibit positions visited
  locations          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Presentations attended
  presentations      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # First merged list of positions + presentations
  after_first_merge  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Second merged list of positions + presentations
  after_second_merge : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Num of locations visited.
  num_locations               : int = sql.Column(sql.Integer, nullable = False)
  # Num of unique locations visited.
  num_unique_locations        : int = sql.Column(sql.Integer, nullable = False)
  # Num of presentations attented
  num_presentations           : int = sql.Column(sql.Integer, nullable = False)
  # Num of presentations + locations
  num_presentations_locations : int = sql.Column(sql.Integer, nullable = False)
  # Average time spent per location
  avg_time_per_location       : int = sql.Column(sql.Integer, nullable = False)
  # Total time spent in museum
  total_time                  : int = sql.Column(sql.Integer, nullable = False)
  # Visiting type label
  movement_type : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Datetime
  date_added    : datetime.datetime = sql.Column(sql.DateTime, nullable = False, default = datetime.datetime.utcnow)

  @classmethod
  def FromContentFile(
    cls,
    contentfile_root : pathlib.Path,
    relpath          : pathlib.Path,
    preprocessors    : typing.List[typing.Callable],
  ) -> "ContentFile":
    """Instantiate a ContentFile."""
    preprocessing_succeeded = False
    # try:
    with open(contentfile_root / relpath) as f:
      try:
        input_text = f.read()
      except Exception:
        input_text = "/*corrupted file format*/"
    for pr in preprocessors:
      input_text = pr(input_text)

    return [ cls(
      sha256         = crypto.sha256_str(str([str(x) for x in v.values()])),
      input_relpath  = relpath,
      visitor_id     = v['id'],
      locations      = '\n'.join(v['locations']),
      presentations  = '\n'.join(v['presentations']),
      after_first_merge  = '\n'.join(v['after_first_merge']),
      after_second_merge = '\n'.join(v['after_second_merge']),
      num_locations      = v['#locations'],
      num_unique_locations = v['#unique_locations'],
      num_presentations  = v['#presentations'],
      num_presentations_locations = v['#presentations_w_locations'],
      avg_time_per_location       = v['avg_time_per_location'],
      total_time                  = v['total_time'],
      movement_type               = v['movement_type'],
      date_added                  = datetime.datetime.utcnow(),
    ) for v in input_text ]

class PreprocessedContentFiles(sqlutil.Database):
  """A database of pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(PreprocessedContentFiles, self).__init__(
      url, Base, must_exist=must_exist
    )

  def Create(self, config: dataset_pb2.HechtDataset):
    with self.Session() as session:
      if not self.IsDone(session):
        self.Import(session, config)
        self.SetDone(session)
        session.commit()

    # Logging output.
    num_input_files = session.query(ContentFile).count()

    set_mail = "Content: {} files.\n".format(
              humanize.intcomma(num_input_files),
            )
    l.getLogger().info(
      "Content: {} files.".format(
              humanize.intcomma(num_input_files),
            ), mail_level = 4
    )
    if FLAGS.notify_me:
      client.getClient().send_message("AITrace:preprocessed", set_mail)
    return

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == "done").first():
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key="done", value="yes"))

  def Import(self,
             session: sqlutil.Session,
             config: typing.Union[dataset_pb2.HechtDataset, dataset_pb2.AITraceDataset]
             ) -> None:
    with self.GetContentFileRoot(config) as contentfile_root:
      relpaths = set(self.GetImportRelpaths(contentfile_root, config))
      done = set(
        [x[0] for x in session.query(ContentFile.input_relpath)]
      )
      todo = relpaths - done
      l.getLogger().info(
        "Preprocessing {} of {} content files".format(
                humanize.intcomma(len(todo)),
                humanize.intcomma(len(relpaths)),
            )
      )

      bar = progressbar.ProgressBar(max_value=len(todo))
      c = 0
      last_commit = time.time()
      wall_time_start = time.time()
      try:
        for job in bar(todo):
          pr_list = ContentFile.FromContentFile(
            contentfile_root = contentfile_root,
            relpath = job,
            preprocessors = preprocessors.getPreprocessors(config)
          )
          for pr_file in pr_list:
            wall_time_end = time.time()
            pr_file.wall_time_ms = int(
              (wall_time_end - wall_time_start) * 1000
            )
            wall_time_start = wall_time_end
            session.add(pr_file)
            if wall_time_end - last_commit > 10:
              session.commit()
              last_commit = wall_time_end

      except Exception as e:
        raise e

  @contextlib.contextmanager
  def GetContentFileRoot(self, config: dataset_pb2.HechtDataset) -> pathlib.Path:
    """Get the path of the directory containing content files.

    If the corpus is a local directory, this simply returns the path. Otherwise,
    this method creates a temporary copy of the files which can be used within
    the scope of this context.

    Args:
      config: The corpus config proto.

    Returns:
      The path of a directory containing content files.
    """
    if config.HasField("local_directory"):
      yield pathlib.Path(ExpandConfigPath(config.local_directory))
    elif config.HasField("local_tar_archive"):
      with tempfile.TemporaryDirectory(prefix="AITrace_dataset_") as d:
        start_time = time.time()
        if isinstance(config, dataset_pb2.HechtDataset):
          cmd = ["unzip", "-qq", str(ExpandConfigPath(config.local_tar_archive)), "-d", d]
          subprocess.check_call(cmd)
          with tempfile.TemporaryDirectory(prefix="AITrace_dataset_") as d2:
            cmd2 = ["unzip", "-qq", str(pathlib.Path(d) / "museum_data" / "visitors_logs.zip"), "-d", d2]
            subprocess.check_call(cmd2)

            l.getLogger().info(
              "Unpacked {} in {} ms".format(
                      ExpandConfigPath(config.local_tar_archive).name,
                      humanize.intcomma(int((time.time() - start_time) * 1000)),
                  )
            )
            yield pathlib.Path(d2)
        elif isinstance(config, dataset_pb2.AITraceDataset):
          cmd = ["unzip", "-qq", str(ExpandConfigPath(config.local_tar_archive)), "-d", d]
          subprocess.check_call(cmd)
          l.getLogger().info(
            "Unpacked {} in {} ms".format(
                    ExpandConfigPath(config.local_tar_archive).name,
                    humanize.intcomma(int((time.time() - start_time) * 1000)),
                )
          )
          yield pathlib.Path(d)
        else:
          raise NotImplementedError
    else:
      raise NotImplementedError

  @property
  def size(self) -> int:
    """Return the total number of files in the pre-processed corpus.
    """
    with self.Session() as session:
      return session.query(ContentFile).count()

  def GetImportRelpaths(
    self,
    contentfile_root: pathlib.Path,
    config: dataset_pb2.HechtDataset,
  ) -> typing.List[str]:
    """Get relative paths to all files in the content files directory.

    Args:
      contentfile_root: The root of the content files directory.

    Returns:
      A list of paths relative to the content files root.

    Raises:
      EmptyHechtDatasetException: If the content files directory is empty.
    """

    if isinstance(config, dataset_pb2.HechtDataset):
      return [str(contentfile_root / "out.log")]
    elif isinstance(config, dataset_pb2.AITraceDataset):
      return [str(contentfile_root / "AITrace_museum.csv")]
    else:
      raise NotImplementedError

def ExpandConfigPath(path: str) -> pathlib.Path:
  return pathlib.Path(os.path.expandvars(path)).expanduser().absolute()

def GetFileSha256(path: pathlib.Path) -> str:
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()
