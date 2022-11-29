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

class Sample(Base):
  """A single encoded content file."""

  __tablename__ = "visiting_sample"

  # The ID of the ContentFile.
  id       : int = sql.Column(sql.Integer, primary_key=True)
  # Time series of museum visitor.
  sha256    : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Target label of visiting type.
  label    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Exhibit ids visited with attendance times.
  exhibits : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Num of ids visited
  num_exhibits        : int = sql.Column(sql.Integer, nullable = False)
  # Num of unique exhibits visited
  num_unique_exhibits : int = sql.Column(sql.Integer, nullable = False)
  # Total time attended 
  total_attendance    : int = sql.Column(sql.Integer, nullable = False)
  # Date added.
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id: int,
               sha256: str,
               label: str,
               exhibits: typing.List[typing.Tuple[int, int]]
               ) -> "Sample":
    return Sample(
      id                  = id,
      sha256              = sha256,
      label               = label,
      exhibits            = "\n".join(["{},{}".format(i, t) for i, t in exhibits]),
      num_exhibits        = len([x for x in exhibits if x[0] not in {0, 1}]),
      num_unique_exhibits = len(set([x for x, _ in exhibits if x not in {0, 1}])),
      total_attendance    = sum([y for _, y in exhibits]),
      date_added          = datetime.datetime.utcnow(),
    )

class SamplesDatabase(sqlutil.Database):
  """A database of encoded pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False):
    super(SamplesDatabase, self).__init__(url, Base, must_exist=must_exist)
    return

  @property
  def count(self):
    """Return the total number of files in the encoded corpus."""
    with self.Session() as session:
      return session.query(Sample).count()
