"""This file defines a database for encoded content files."""
import datetime
import functools
import multiprocessing
import pickle
import time
import typing
import pathlib
import humanize
import numpy as np
import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from datasets import preprocessed
from datasets import encoders
from proto import internal_pb2
from util import monitors
from backend import sqlutil

from absl import flags

from eupy.native import logger as l

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class Meta(Base):
  """Meta table for encoded content files database."""

  __tablename__ = "meta"

  key: str = sql.Column(sql.String(1024), primary_key = True)
  value: str = sql.Column(sql.String(1024), nullable = False)

class EncodedContentFile(Base):
  """A single encoded content file."""

  __tablename__ = "encoded_contentfiles"

  # The ID of the ContentFile.
  id: int = sql.Column(sql.Integer, primary_key=True)
  # Time series of museum visitor.
  data: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Target label of visiting type.
  label: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Number of tokens in sequence
  linecount: int = sql.Column(sql.Integer, nullable=False)
  # The number of milliseconds encoding took.
  encoding_time_ms: int = sql.Column(sql.Integer, nullable=False)
  # Encoding is parallelizable, so the actual wall time of encoding may be much
  # less than the sum of all encoding_time_ms. This column counts the effective
  # number of "real" milliseconds during encoding between the last encoded
  # result and this result coming in. The idea is that summing this column
  # provides an accurate total of the actual time spent encoding an entire
  # corpus. Will be <= encoding_time_ms.
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromPreprocessed(cls, preprocessed_cf: preprocessed.ContentFile) -> "EncodedContentFile":
    """Instantiate an EncodedContentFile from a preprocessed file.

    Args:
      preprocessed_cf: A ContentFile instance.
      tokenizer: The tokenizer to encode using.
      eof: An end-of-file marker which is concatenated to the encoded sequence.

    Returns:
      An EncodedContentFile instance.
    """
    start_time = time.time()
    data = encoders.EncodeLocations(
           encoders.EncodeTimeIntervals(
           ([x.split(',') for x in preprocessed_cf.locations.split('\n')])))
    label = encoders.visiting_labels[preprocessed_cf.movement_type]
    encoding_time_ms = int((time.time() - start_time) * 1000)
    return EncodedContentFile(
      id               = preprocessed_cf.id,
      data             = '\n'.join([','.join([str(y) for y in x]) for x in data]),
      label            = str(label),
      linecount        = len(data),
      encoding_time_ms = encoding_time_ms,
      wall_time_ms     = encoding_time_ms,  # The outer-loop may change this.
      date_added       = datetime.datetime.utcnow(),
    )

def EncoderWorker(job: internal_pb2.EncoderWorker,) -> typing.Optional[EncodedContentFile]:
  try:
    return EncodedContentFile.FromPreprocessed(job)
  except Exception as e:
    raise e

class EncodedContentFiles(sqlutil.Database):
  """A database of encoded pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    self.encoded_path = pathlib.Path(url.replace("sqlite:///", "")).parent
    super(EncodedContentFiles, self).__init__(url, Base, must_exist=must_exist)
    self.time_attendance_monitor = monitors.NormalizedFrequencyMonitor(self.encoded_path, "time_attendance_distribution")
    self.exhibit_id_monitor      = monitors.NormalizedFrequencyMonitor(self.encoded_path, "exhibit_id_distribution")
    return

  def Create(self, p: preprocessed.PreprocessedContentFiles) -> bool:
    """Populate the encoded contentfiles database.

    Args:
      p: A PreprocessedContentFiles database.
    Returns:
      True if work was done, else False.

    Raises:
      EmptyCorpusException: If the PreprocessedContentFiles database has
        no files.
    """
    with self.Session() as session:
      if not self.IsDone(session):
        self.Import(session, p)
        self.SetDone(session)
        session.commit()

    # Logging output.
    num_files = session.query(EncodedContentFile).count()
    total_walltime, total_time, = session.query(
      func.sum(EncodedContentFile.wall_time_ms),
      func.sum(EncodedContentFile.encoding_time_ms),
    ).first()
    l.getLogger().info("Encoded {} files in {} ms ({:.2f}x speedup)"
                        .format(
                            humanize.intcomma(num_files),
                            humanize.intcomma(total_walltime),
                            total_time / total_walltime,
                          ), mail_level = 4
                      )
    return

  @property
  def size(self):
    """Return the total number of files in the encoded corpus."""
    with self.Session() as session:
      return session.query(EncodedContentFile).count()

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == "done").first():
      return True
    return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key="done", value="yes"))

  def Import(self, session: sqlutil.Session, preprocessed_db: preprocessed.PreprocessedContentFiles) -> None:
    with preprocessed_db.Session() as p_session:
      query = p_session.query(preprocessed.ContentFile).filter(~preprocessed.ContentFile.id.in_(session.query(EncodedContentFile.id).all()),)
      total_jobs = query.count()
      l.getLogger().info("Encoding {} of {} preprocessed files"
                          .format(
                              humanize.intcomma(total_jobs),
                              humanize.intcomma(p_session.query(preprocessed.ContentFile).count())
                          )
                        )
      bar = progressbar.ProgressBar(max_value=total_jobs)
      chunk, idx = 2000000, 0
      last_commit = time.time()
      wall_time_start = time.time()
      while idx < total_jobs:
        try:
          batch = query.limit(chunk).offset(idx).all()
          pool = multiprocessing.Pool()
          for encoded_cf in pool.imap_unordered(functools.partial(EncoderWorker), batch):
            wall_time_end = time.time()
            if encoded_cf:
              encoded_cf.wall_time_ms = int(
                (wall_time_end - wall_time_start) * 1000
              )
              session.add(encoded_cf)
              self.time_attendance_monitor.register([int(t.split(',')[-1]) for t in encoded_cf.data.split('\n')])
              self.exhibit_id_monitor.register     ([int(t.split(',')[ 0]) for t in encoded_cf.data.split('\n')])
            wall_time_start = wall_time_end
            if wall_time_end - last_commit > 10:
              session.commit()
              last_commit = wall_time_end
            idx += 1
            bar.update(idx)
          pool.close()
          self.time_attendance_monitor.plot()
          self.exhibit_id_monitor.plot()
        except KeyboardInterrupt as e:
          pool.terminate()
          self.time_attendance_monitor.plot()
          self.exhibit_id_monitor.plot()
          raise e
        except Exception as e:
          l.getLogger().error(e)
          pool.terminate()
          self.time_attendance_monitor.plot()
          self.exhibit_id_monitor.plot()
          raise e
    session.commit()
    return
