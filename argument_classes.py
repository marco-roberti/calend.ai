from dataclasses import dataclass, field
from typing import Optional, Union

from torch.optim.optimizer import Optimizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, SchedulerType, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_constant_schedule, add_start_docstrings


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to "
                    "`max_target_length`. This argument is also used to override the ``max_length`` param of "
                    "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
@add_start_docstrings(Seq2SeqTrainingArguments.__doc__)
class Seq2SeqTrainingArgumentsWithScheduler(Seq2SeqTrainingArguments):
    lr_scheduler_end: float = field(
        default=1e-7, metadata={'help': 'The final value of the polynomial learning rate decay.'}
    )
    lr_scheduler_power: float = field(
        default=1.0, metadata={'help': 'The power factor of the polynomial learning rate decay.'}
    )
    lr_scheduler_cycles: float = field(
        default=-1,
        metadata={'help': '[lr_scheduler_type=cosine] number of waves (defaults is 0.5, i.e. decrease from the max '
                          'value to 0). [lr_scheduler_type=cosine_with_restarts] The number of hard restarts to use '
                          '(default is 1).'}
    )

    @property
    def num_cycles(self):
        if self.lr_scheduler_cycles == -1:
            if self.lr_scheduler_type == 'cosine':
                return 0.5
            if self.lr_scheduler_type == 'cosine_with_restarts':
                return 1
        else:
            return self.lr_scheduler_cycles


class Seq2SeqTrainerWithSchedulerArgs(Seq2SeqTrainer):

    def get_scheduler(
            self,
            name: Union[str, SchedulerType],
            optimizer: Optimizer,
            num_training_steps: Optional[int] = None,
    ):
        """
        Unified API to get any scheduler from its name.

        Args:
            name (:obj:`str` or `:obj:`SchedulerType`):
                The name of the scheduler to use.
            optimizer (:obj:`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_training_steps (:obj:`int`, `optional`):
                The number of training steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
        """
        name = SchedulerType(name)
        self.args: Seq2SeqTrainingArgumentsWithScheduler

        if name == SchedulerType.CONSTANT:
            return get_constant_schedule(optimizer)

        # All other schedulers require `num_warmup_steps`
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        if name == SchedulerType.LINEAR:
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        if name == SchedulerType.COSINE:
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, self.args.num_cycles)  # 0.5
        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, self.args.num_cycles)  # 1
        if name == SchedulerType.POLYNOMIAL:
            return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                                             self.args.lr_scheduler_end, self.args.lr_scheduler_power)

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = self.get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_training_steps=num_training_steps,
            )
