import os
import numpy as np
import pathlib
import time
import typing
import tempfile
import subprocess
import humanize
import checksumdir

from datasets import preprocessed
from datasets import encoded
from proto import dataset_pb2
from util import pbutil
from util import crypto
from util import commit

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

def AssertConfigIsValid(config: typing.Union[dataset_pb2.HechtDataset, dataset_pb2.AITraceDataset]) -> dataset_pb2.Dataset:
  """Assert that config proto is valid.

  Args:
    config: A Dataset proto.

  Returns:
    The Dataset proto.

  Raises:
    UserError: If the config is invalid.
  """
  try:
    if not isinstance(config, dataset_pb2.HechtDataset) and not isinstance(config, dataset_pb2.AITraceDataset):
      raise TypeError(config, type(config))
    return config
  except pbutil.ProtoValueError as e:
    raise e


class Dataset(object):
  """Representation of a training corpus.

  Please note corpus instances should be treated as immutable. Upon
  instantiation, a corpus's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: dataset_pb2.Dataset, base_dir: pathlib.Path):
    """Instantiate a corpus from a proto config.

    If this is a new corpus, a number of files will be created, which may
    take some time.

    Args:
      config: A Dataset message.

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: In case the corpus is not found, or config contains invalid
        options.
      EmptyDatasetException: In case the corpus contains no data.
    """
    if not isinstance(config, dataset_pb2.Dataset):
      raise TypeError(f"Config must be a Dataset proto. Received: '{type(config).__name__}'")

    # Make a local copy of the configuration.
    if isinstance(config, dataset_pb2.Dataset):
      if config.HasField("hecht_dataset"):
        self.config = dataset_pb2.HechtDataset()
        self.config.CopyFrom(AssertConfigIsValid(config.hecht_dataset))
      elif config.HasField("ai_trace_dataset"):
        self.config = dataset_pb2.AITraceDataset()
        self.config.CopyFrom(AssertConfigIsValid(config.ai_trace_dataset))

    self._created = False

    print()
    # An in-memory cache of the encoded contentfiles indices arrays.
    # Set and used in GetTrainingData().
    self._indices_arrays: typing.Optional[typing.List[np.array]] = None

    (base_dir / "dataset").mkdir(parents = True, exist_ok = True)
    self.content_id = ResolveContentId(self.config)
    # Database of pre-processed files.
    preprocessed_id = ResolvePreprocessedId(self.content_id, self.config)
    (base_dir / "dataset" / "preprocessed" / preprocessed_id).mkdir(exist_ok = True, parents = True)
    preprocessed_db_path = base_dir / "dataset" / "preprocessed" / preprocessed_id / "preprocessed.db"

    self.preprocessed = preprocessed.PreprocessedContentFiles(
      f"sqlite:///{preprocessed_db_path}"
    )
    # Create symlink to contentfiles.
    symlink = (pathlib.Path(self.preprocessed.url[len("sqlite:///") :]).parent / "contentfiles")
    if not symlink.is_symlink():
      if self.config.HasField("local_directory"):
        os.symlink(
          str(ExpandConfigPath(self.config.local_directory)),
          symlink,
        )
      elif self.config.HasField("local_tar_archive"):
        os.symlink(
          str(ExpandConfigPath(self.config.local_tar_archive)),
          symlink,
        )
    # Data of encoded pre-preprocessed files.
    encoded_id = ResolveEncodedId(self.content_id, self.config)
    (base_dir / "dataset" / "encoded" / encoded_id).mkdir(exist_ok = True, parents = True)
    db_path = base_dir / "dataset" / "encoded" / encoded_id / "encoded.db"
    self.encoded = encoded.EncodedContentFiles(f"sqlite:///{db_path}")
    symlink = (pathlib.Path(self.encoded.url[len("sqlite:///") :]).parent / "preprocessed")
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(
          pathlib.Path(self.preprocessed.url[len("sqlite:///") :]).parent,
          pathlib.Path(self.encoded.url[len("sqlite:///") :]).parent,
          ),
        symlink,
      )
    self.hash = encoded_id
    self.cache_path = base_dir / "dataset" / "encoded" / encoded_id
    self.cache_path.mkdir(exist_ok = True, parents = True)
    commit.saveCommit(self.cache_path)
    commit.saveCommit(self.cache_path.parent.parent / "preprocessed" / preprocessed_id)
    l.getLogger().info("Pre-processed corpus hash: {}".format(preprocessed_id))
    l.getLogger().info("Encoded corpus hash: {}".format(encoded_id))
    return

  def GetShortSummary(self) -> str:
    corpus_size = humanize.naturalsize(self.encoded.token_count)
    return (
      f"{corpus_size} token corpus with {self.vocab_size}-element vocabulary"
    )

  def Create(self) -> None:
    """Create the corpus files.
  
      Raises:
      EmptyDatasetException: If there are no content files, or no successfully
        pre-processed files.
    """
    self._created = True
    l.getLogger().info("Content ID: {}".format(self.content_id))

    preprocessed_lock_path = (
      pathlib.Path(self.preprocessed.url[len("sqlite:///") :]).parent / "LOCK"
    )
    self.preprocessed.Create(self.config)
    if not self.preprocessed.size:
      raise ValueError(
        f"Pre-processed corpus contains no files: '{self.preprocessed.url}'"
      )
    encoded_lock_path = (
      pathlib.Path(self.encoded.url[len("sqlite:///") :]).parent / "LOCK"
    )
    start_time      = time.time()
    self.encoded.Create(self.preprocessed)
    return

  def getData(self) -> typing.List[typing.Tuple[str, str]]:
    """Return pairs of inputs-outputs for visitors.

    Returns:
      The preprocessed data.
    """
    with self.preprocessed.Session() as session:
      return [(x.locations, x.label) for x in session.query(preprocessed.ContentFile)]

  def getEncodedData(self) -> typing.List[typing.Tuple[str, str]]:
    """Return pairs of inputs-outputs for visitors.

    Returns:
      The final encoded data.
    """
    with self.encoded.Session() as session:
      return [(x.data, x.label) for x in session.query(encoded.EncodedContentFile)]

  def GetNumContentFiles(self) -> int:
    """Get the number of contentfiles which were pre-processed."""
    with self.preprocessed.Session() as session:
      return session.query(preprocessed.ContentFile).count()

  @property
  def size(self) -> int:
    """Return the size of the atomized corpus."""
    with self.encoded.Session() as session:
      return session.query(
        sql.func.sum(encoded.EncodedContentFile.tokencount)
      ).one()

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Dataset):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)

def ExpandConfigPath(path: str, path_prefix: str = None) -> pathlib.Path:
  if "HOME" not in os.environ:
    os.environ["HOME"] = str(pathlib.Path("~").expanduser())
  return (
    pathlib.Path(os.path.expandvars((path_prefix or "") + path))
    .expanduser()
    .absolute()
  )


def ResolveContentId(config: dataset_pb2.Dataset) -> str:
  """Compute the hash of the input contentfiles.

  This function resolves the unique sha1 checksum of a set of content files.

  Args:
    config: The corpus config proto.
  Returns:
    A hex encoded sha1 string.
  """
  start_time = time.time()
  if config.HasField("local_directory"):
    local_directory = ExpandConfigPath(
      config.local_directory
    )
    content_id = crypto.sha256_str(str(local_directory))
  elif config.HasField("local_tar_archive"):
    content_id = GetHashOfArchiveContents(
      ExpandConfigPath(config.local_tar_archive)
    )
  else:
    raise NotImplementedError("Unsupported Dataset.contentfiles field value")
  l.getLogger().warning(
    "Resolved Content ID {} in {} ms.".format(
          content_id,
          humanize.intcomma(int((time.time() - start_time) * 1000)),
        )
  )
  return content_id


def ResolvePreprocessedId(content_id: str,
                          config: dataset_pb2.Dataset
                          ) -> str:
  """Compute the hash of a corpus of preprocessed contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the preprocessor pipeline.
  """
  # out Dataset class.
  return crypto.sha1_list(content_id)

def ResolveEncodedId(content_id: str,
                     config: dataset_pb2.Dataset
                     ) -> str:
  """Compute the hash of a corpus of preprocessed and encoded contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the config proto.
  """
  config_without_contentfiles = type(config)()
  config_without_contentfiles.CopyFrom(config)
  # Clear the contentfiles field, since we use the content_id to uniquely
  # identify the input files. This means that corpuses with the same content
  # files delivered through different means (e.g. two separate but identical
  # directories) have the same hash.
  config_without_contentfiles.ClearField("contentfiles")
  return crypto.sha1_list(
    content_id, config_without_contentfiles.SerializeToString()
  )


def GetHashOfArchiveContents(archive: pathlib.Path) -> str:
  """Compute the checksum of the contents of a directory.

  Args:
    archive: Path of the archive.

  Returns:
    Checksum of the archive.

  Raises:
    UserError: If the requested archive does not exist, or cannot be unpacked.
  """
  with tempfile.TemporaryDirectory(prefix="aitrace_corpus_") as d:
    tar = ["unzip", "-qq", archive, "-d", d]
    try:
      # pv_proc = subprocess.Popen(pv, stdout = subprocess.PIPE)
      subprocess.check_call(tar)
    except subprocess.CalledProcessError:
      raise ValueError(f"Archive unpack failed: '{archive}'")
    return checksumdir.dirhash(d, "sha1")
