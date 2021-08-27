from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List, Tuple, Callable

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, SchedulerType, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_constant_schedule, add_start_docstrings, PreTrainedModel, TrainingArguments, DataCollator, \
    PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.deepspeed import is_deepspeed_zero3_enabled


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
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
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum numbers of tokens to generate, ignore the current number of tokens. Use either "
                    "max_new_tokens or max_length but not both, they serve the same purpose."
        },
    )
    min_length: Optional[int] = field(
        default=10,
        metadata={"help": "The minimum length of the sequence to be generated."},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum target length to use when predicting with the generate method."},
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."},
    )
    early_stopping: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to stop the beam search when at least num_beams sentences are finished per batch or not."
        },
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or "
                    "higher are kept for generation."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to "
                    "encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the "
                    "model to produce longer sequences."
        },
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."},
    )
    encoder_no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "If set to int > 0, all ngrams of that size that occur in the encoder_input_ids cannot occur in "
                    "the decoder_input_ids."
        },
    )
    num_beam_groups: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of groups to divide num_beams into in order to ensure diversity among different groups of "
                    "beams."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "This value is subtracted from a beam's score if it generates a token same as any beam from other "
                    "group at a particular time. Note that diversity_penalty is only effective if group beam search is "
                    "enabled."
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
        default=1024,
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
class Seq2SeqTrainingArgumentsRefined(Seq2SeqTrainingArguments):
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


class Seq2SeqTrainerRefined(Seq2SeqTrainer):

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 max_new_tokens=None, min_length=10, max_length=512, num_beams=1, do_sample=False, early_stopping=False,
                 temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, length_penalty=1.0,
                 no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0, num_beam_groups=1, diversity_penalty=0.0):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers)
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty

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
        self.args: Seq2SeqTrainingArgumentsRefined

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

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "max_new_tokens": self.max_new_tokens,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "encoder_no_repeat_ngram_size": self.encoder_no_repeat_ngram_size,
            "num_beam_groups": self.num_beam_groups,
            "diversity_penalty": self.diversity_penalty
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels
