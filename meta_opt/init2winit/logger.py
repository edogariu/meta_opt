from init2winit.utils import metric_writers, gfile, pd, logging, checkpoint, flax_checkpoints, exists, json
from meta_opt.experiment import ExperimentConfig

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  logging.exception('Unable to import wandb.')
  wandb = None

class CustomMetricLogger(object):
  """Used to log all measurements during training.

  Note: Writes are not atomic, so files may become corrupted if preempted at
  the wrong time.
  """

  def __init__(self,
               csv_path='',
               json_path='',
               pytree_path='',
               events_dir=None,
               experiment_cfg: ExperimentConfig = None,
               **logger_kwargs):
    """Create a recorder for metrics, as CSV or JSON.


    Args:
      csv_path: A filepath to a CSV file to append to.
      json_path: An optional filepath to a JSON file to append to.
      pytree_path: Where to save trees of numeric arrays.
      events_dir: Optional. If specified, save tfevents summaries to this
        directory.
      **logger_kwargs: Optional keyword arguments, whose only valid parameter
        name is an optional XM WorkUnit used to also record metrics to XM as
        MeasurementSeries.
    """
    self._use_wandb = experiment_cfg.use_wandb
    self._measurements = {}
    self._csv_path = csv_path
    self._json_path = json_path
    self._pytree_path = pytree_path
    if logger_kwargs:
      if len(logger_kwargs.keys()) > 1 or 'xm_work_unit' not in logger_kwargs:
        raise ValueError(
            'The only logger_kwarg that should be passed to MetricLogger is '
            'xm_work_unit.')
      self._xm_work_unit = logger_kwargs['xm_work_unit']
    else:
      self._xm_work_unit = None

    self._tb_metric_writer = None
    if events_dir:
      self._tb_metric_writer = metric_writers.create_default_writer(events_dir)
    if wandb is not None and self._use_wandb:
        wandb.init(
            dir=events_dir, tags=['init2winit'])
        wandb.config.update(experiment_cfg)

  def append_scalar_metrics(self, metrics):
    """Record a dictionary of scalar metrics at a given step.

    Args:
      metrics: a Dict of metric names to scalar values. 'global_step' is the
        only required key.
    """
    try:
      with gfile.GFile(self._csv_path) as csv_file:
        measurements = pd.read_csv(csv_file)
        measurements = pd.concat([measurements, pd.DataFrame([metrics])])
    except (pd.errors.EmptyDataError, gfile.FileError) as e:
      measurements = pd.DataFrame([metrics], columns=sorted(metrics.keys()))
      if isinstance(e, pd.errors.EmptyDataError):
        # TODO(ankugarg): Identify root cause for the corrupted file.
        # Most likely it's preemptions or file-write error.
        logging.info('Measurements file is empty. Create a new one, starting '
                     'with metrics from this step.')
    # TODO(gdahl,gilmer): Should this be an atomic file?
    with gfile.GFile(self._csv_path, 'w') as csv_file:
      measurements.to_csv(csv_file, index=False)
    if self._xm_work_unit:
      for name, value in metrics.items():
        if name not in self._measurements:
          self._measurements[name] = self._xm_work_unit.get_measurement_series(
              label=name)
        self._measurements[name].create_measurement(
            objective_value=value, step=metrics['global_step'])

    if self._tb_metric_writer:
        self._tb_metric_writer.write_scalars(
            step=int(metrics['global_step']), scalars=metrics)
        # This gives a 1-2% slowdown in steps_per_sec on cifar-10 with batch
        # size 512. We could only flush at the end of training to optimize this.
        self._tb_metric_writer.flush()
        if wandb is not None and self._use_wandb:
            wandb.log(metrics)

  def finish(self) -> None:
    if wandb is not None and self._use_wandb:
      wandb.finish()

  def write_pytree(self, pytree, prefix='training_metrics'):
    """Record a serializable pytree to disk, overwriting any previous state.

    Args:
      pytree: Any serializable pytree
      prefix: The prefix for the checkpoint.  Save path is
        self._pytree_path/prefix
    """
    state = dict(pytree=pytree)
    checkpoint.save_checkpoint(
        self._pytree_path,
        step='',
        state=state,
        prefix=prefix,
        max_to_keep=None)

  def append_pytree(self, pytree, prefix='training_metrics'):
    """Append and record a serializable pytree to disk.

    The pytree will be saved to disk as a list of pytree objects. Everytime
    this function is called, it will load the previous saved state, append the
    next pytree to the list, then save the appended list.

    Args:
      pytree: Any serializable pytree.
      prefix: The prefix for the checkpoint.
    """
    # Read the latest (and only) checkpoint if it exists, then append the new
    # state to it before saving back to disk.
    try:
      old_state = flax_checkpoints.restore_checkpoint(
          self._pytree_path, target=None, prefix=prefix)
    except ValueError:
      old_state = None
    # Because we pass target=None, checkpointing will return the raw state
    # dict, where 'pytree' is a dict with keys ['0', '1', ...] instead of a
    # list.
    if old_state:
      state_list = old_state['pytree']
      state_list = [state_list[str(i)] for i in range(len(state_list))]
    else:
      state_list = []
    state_list.append(pytree)

    self.write_pytree(state_list)

  def append_json_object(self, json_obj):
    """Append a json serializable object to the json file."""

    if not self._json_path:
      raise ValueError('Attempting to write to a null json path')
    if exists(self._json_path):
      with gfile.GFile(self._json_path) as json_file:
        json_objs = json.loads(json_file.read())
      json_objs.append(json_obj)
    else:
      json_objs = [json_obj]
    # TODO(gdahl,gilmer): Should this be an atomic file?
    with gfile.GFile(self._json_path, 'w') as json_file:
      json_file.write(json.dumps(json_objs))